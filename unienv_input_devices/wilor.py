"""Monocular-RGB hand tracking as an observation-only UniEnv WorldNode.

This node wraps the WiLoR-mini inference pipeline (YOLO hand detection + MANO
3D pose estimation) and exposes its per-frame outputs as a UniEnv observation
node. It is a pure sensor: ``action_space`` is ``None`` and any action passed to
:meth:`set_next_action` is silently ignored.

Coordinate frames (documented exhaustively because the downstream retargeting
pipelines are frame-sensitive):

- **Camera frame** (used by ``keypoints_3d`` and ``wrist_pose``): a standard
  pinhole camera frame with **x right, y down, z forward**, units in meters.
  This is the frame WiLoR's ``pred_cam_t_full`` and ``pred_keypoints_3d`` are
  natively expressed in once ``pred_cam_t_full`` is added to the wrist-rooted
  MANO-local keypoints. The node emits this frame **raw** — no extrinsics, no
  operator-rig offsets, no OPERATOR2MANO rotation.

- **MANO-local frame** (used by ``keypoints_3d_local``): wrist-rooted, meters,
  as returned by WiLoR's ``pred_keypoints_3d`` (the wrist keypoint is at the
  origin). This is the articulation-only signal: it is invariant to wrist
  translation, so a noisy ``pred_cam_t_full`` cannot corrupt finger curl
  estimates. Use this for retargeting joint angles; use ``keypoints_3d`` for
  absolute 3D positions.

- **Pixel frame** (used by ``keypoints_2d``): image coordinates in pixels,
  origin at the top-left, x right, y down. Range is generous (0..8192) to
  accommodate any reasonable webcam resolution.

- **21-keypoint order**: wrist (0), thumb (1-4), index (5-8), middle (9-12),
  ring (13-16), little (17-20) — identical to the MediaPipe convention, so the
  output feeds retargeting pipelines expecting mediapipe-style (21, 3) arrays
  directly.

Integration notes
-----------------
The integration is adapted from the MIT-licensed AnyTeleop community fork
(https://github.com/RalphFH/AnyTeleop). The fork applies hardcoded translation
offsets (``z -= 0.6``, ``y -= 0.2``) and a fixed ``OPERATOR2MANO`` rotation
that are specific to that teleoperation rig; this node **does not** port them.
We emit raw camera-frame data and let downstream nodes apply any rig-specific
transforms.

WiLoR model weights are released under **CC-BY-NC-ND** and the MANO hand model
assets are **non-commercial**. Both auto-download from HuggingFace on first
use of the WiLoR pipeline. This package vendors none of those assets — the
user installs ``wilor_mini`` separately (see README) and accepts the upstream
licensing terms.
"""

from typing import Dict, Literal, Optional

import numpy as np

from unienv_interface.world import WorldNode, RealWorld, World
from unienv_interface.backends import ComputeBackend, BArrayType, BDeviceType, BDtypeType, BRNGType
from unienv_interface.backends.numpy import (
    NumpyComputeBackend,
    NumpyArrayType,
    NumpyDeviceType,
    NumpyDtypeType,
    NumpyRNGType,
)
from unienv_interface.space import DictSpace, BoxSpace


class WiLoRHandNode(WorldNode[
    None, Dict[str, NumpyArrayType], None,
    NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
]):
    """Observation-only WorldNode streaming monocular-RGB hand tracking from WiLoR-mini.

    The node reads BGR frames from a single UVC webcam via OpenCV, runs the
    WiLoR-mini pipeline (YOLO hand detection + MANO 3D pose regression), and
    caches the latest per-hand result as the observation. ``action_space`` is
    ``None`` (observation-only); any action passed to :meth:`set_next_action`
    is silently ignored.

    Coordinate frames (see the module docstring for the full treatment):

    - ``keypoints_3d_local`` (21, 3): MANO-local, wrist-rooted, meters — the
      articulation-only signal.
    - ``keypoints_3d`` (21, 3): camera frame (x right, y down, z forward),
      meters — ``keypoints_3d_local + pred_cam_t_full``.
    - ``keypoints_2d`` (21, 2): pixel coordinates, top-left origin.
    - ``wrist_pose`` (4, 4): camera-frame homogeneous transform; rotation from
      ``global_orient`` (axis-angle) via
      ``scipy.spatial.transform.Rotation.from_rotvec``, translation from
      ``pred_cam_t_full``.
    - ``hand_detected`` (1,): 1.0 if a hand of the configured handedness was
      found in the latest read, else 0.0.

    When ``connect=False`` (or no hand has been detected yet) the cached
    observation is a zero dict of the correct shapes/dtypes, so the node can
    be constructed and driven through the full lifecycle without ``torch``,
    ``cv2``, ``scipy`` or ``wilor_mini`` installed. When a frame is read but no
    matching hand is found, the previous keypoints/pose are held and
    ``hand_detected`` is set to 0.0 (documented hold-last behavior).
    """

    # The first WorldEnv.reset routes through the reload flow, so after_reload
    # must also refresh the initial observation (mirrors the sibling adaptors).
    after_reset_priorities = {0}
    after_reload_priorities = {0}
    # Observation-only: nothing to send pre-step, so no pre_environment_step set.
    post_environment_step_priorities = {0}

    # Generous bounds for homogeneous-transform entries and 3D positions (meters).
    _TRANSFORM_BOUND = 1e6
    # Pixel-coordinate bounds (any reasonable webcam resolution fits).
    _PIXEL_HIGH = 8192.0
    # Default focal length used by upstream WiLoR-mini (see pipeline source).
    _DEFAULT_FOCAL_LENGTH = 5000.0

    def __init__(
        self,
        world: Optional[RealWorld] = None,
        name: str = "wilor_hand",
        camera_id: int = 0,
        *,
        hand: Literal["left", "right"] = "right",
        focal_length: float = 5000.0,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[int] = None,
        fourcc: Optional[str] = None,
        device: Optional[str] = None,
        connect: bool = True,
        control_timestep: Optional[float] = 0.04,  # 25Hz
        update_timestep: Optional[float] = 0.04,  # frame-read frequency
    ):
        # Set WorldNode-related attributes first so backend/device properties work.
        self.name = name
        self.camera_id = camera_id
        self.hand = hand
        self.focal_length = float(focal_length)
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self._device_input = device  # stored separately; `device` is a read-only property
        self.connect = bool(connect)

        # Hardware handles — only populated when connect=True. Initialized FIRST
        # (before any validation that can raise) so close()/__del__ remain safe
        # even if __init__ raises partway through.
        self._cap = None       # cv2.VideoCapture
        self._pipe = None      # WiLorHandPose3dEstimationPipeline
        self._torch_device = None  # resolved torch.device (str form)

        if self.hand not in ("left", "right"):
            raise ValueError(
                f"WiLoRHandNode: `hand` must be 'left' or 'right', got {self.hand!r}."
            )

        if isinstance(world, World):
            assert world.backend == NumpyComputeBackend, "World backend must be NumpyComputeBackend."
            assert world.is_control_timestep_compatible(control_timestep), \
                "Control timestep must be a multiple of world timestep."
        self.world = world

        if self.connect:
            self._connect_hardware()

        # Build the observation space.
        self.observation_space = DictSpace(
            NumpyComputeBackend,
            {
                "keypoints_3d_local": BoxSpace(
                    NumpyComputeBackend,
                    low=-self._TRANSFORM_BOUND,
                    high=self._TRANSFORM_BOUND,
                    dtype=np.float32,
                    shape=(21, 3),
                ),
                "keypoints_3d": BoxSpace(
                    NumpyComputeBackend,
                    low=-self._TRANSFORM_BOUND,
                    high=self._TRANSFORM_BOUND,
                    dtype=np.float32,
                    shape=(21, 3),
                ),
                "keypoints_2d": BoxSpace(
                    NumpyComputeBackend,
                    low=0.0,
                    high=self._PIXEL_HIGH,
                    dtype=np.float32,
                    shape=(21, 2),
                ),
                "wrist_pose": BoxSpace(
                    NumpyComputeBackend,
                    low=-self._TRANSFORM_BOUND,
                    high=self._TRANSFORM_BOUND,
                    dtype=np.float32,
                    shape=(4, 4),
                ),
                "hand_detected": BoxSpace(
                    NumpyComputeBackend,
                    low=0.0,
                    high=1.0,
                    dtype=np.float32,
                    shape=(1,),
                ),
            },
        )
        # Observation-only node: no action space.
        self.action_space = None

        self.control_timestep = control_timestep  # Control timestep in seconds
        self.update_timestep = update_timestep

        # Allocate the zero observation once so the very first get_observation
        # (and the reload-flow first reset) always returns a non-None dict.
        self._current_observation: Dict[str, NumpyArrayType] = self._zero_observation()

    # ========== Hardware connection ==========

    def _connect_hardware(self) -> None:
        """Open the webcam and construct the WiLoR pipeline (lazy heavy imports)."""
        # --- torch (needed for device resolution + pipeline dtype) ---
        try:
            import torch  # noqa: PLC0415 — lazy import
        except ImportError as e:  # pragma: no cover — requires torch
            raise ImportError(
                "WiLoRHandNode(connect=True) requires PyTorch. Install it with "
                "`pip install torch` (a CUDA build is strongly recommended by the "
                "WiLoR-mini upstream; see https://pytorch.org for instructions)."
            ) from e

        if self._device_input is None:
            self._torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._torch_device = self._device_input

        # --- OpenCV (webcam capture) ---
        try:
            import cv2  # noqa: PLC0415 — lazy import
        except ImportError as e:  # pragma: no cover — requires cv2
            raise ImportError(
                "WiLoRHandNode(connect=True) requires OpenCV. Install it with "
                "`pip install opencv-python`."
            ) from e

        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            raise RuntimeError(
                f"WiLoRHandNode: cv2.VideoCapture({self.camera_id!r}) could not "
                "be opened. Check the camera index / device permissions."
            )
        if self.fourcc is not None:
            # Set the pixel format before resolution — e.g. "MJPG" drastically
            # reduces USB bandwidth, which is required for cameras attached to
            # WSL2 via usbipd (uncompressed YUYV frames arrive zeroed/green).
            if len(self.fourcc) != 4:
                raise ValueError(
                    f"WiLoRHandNode: `fourcc` must be a 4-character code like "
                    f"'MJPG' or 'YUYV', got {self.fourcc!r}."
                )
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        if self.width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.width))
        if self.height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.height))
        if self.fps is not None:
            cap.set(cv2.CAP_PROP_FPS, int(self.fps))
        self._cap = cap

        # --- WiLoR-mini pipeline ---
        try:
            from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (  # noqa: PLC0415 — lazy import
                WiLorHandPose3dEstimationPipeline,
            )
        except ImportError as e:  # pragma: no cover — requires wilor_mini
            raise ImportError(
                "WiLoRHandNode(connect=True) requires the 'wilor_mini' package, "
                "which is not on PyPI. Install it via this package's 'wilor' "
                "extra (pulls a torch>=2.6-compatible fork; see README), and "
                "install chumpy manually first:\n"
                '  pip install --no-build-isolation "chumpy @ '
                'git+https://github.com/mattloper/chumpy@master"\n'
                "A CUDA PyTorch build is strongly recommended. Model "
                "weights and MANO assets auto-download from HuggingFace on first "
                "use (CC-BY-NC-ND / non-commercial terms — see README)."
            ) from e

        self._pipe = WiLorHandPose3dEstimationPipeline(
            device=torch.device(self._torch_device),
            dtype=torch.float16,
            verbose=False,
            focal_length=self.focal_length,
        )

    # ========== Internal helpers ==========

    def _zero_observation(self) -> Dict[str, NumpyArrayType]:
        """Build the all-zeros observation dict of the correct shapes/dtypes.

        ``hand_detected`` starts at 0.0 (no hand seen yet); all geometric
        entries are zero so downstream consumers can branch on
        ``hand_detected`` before touching the keypoints/pose.
        """
        return {
            "keypoints_3d_local": np.zeros((21, 3), dtype=np.float32),
            "keypoints_3d": np.zeros((21, 3), dtype=np.float32),
            "keypoints_2d": np.zeros((21, 2), dtype=np.float32),
            "wrist_pose": np.zeros((4, 4), dtype=np.float32),
            "hand_detected": np.zeros((1,), dtype=np.float32),
        }

    def _select_detection(self, outputs) -> Optional[dict]:
        """Return the first pipeline detection matching the configured handedness.

        WiLoR's ``is_right`` is a float (1.0 for right, 0.0 for left). We match
        ``self.hand == "right"`` against ``is_right == 1.0`` and
        ``self.hand == "left"`` against ``is_right == 0.0``.
        """
        want_right = self.hand == "right"
        for out in outputs:
            is_right = float(out.get("is_right", 0.0))
            if want_right and is_right == 1.0:
                return out
            if not want_right and is_right == 0.0:
                return out
        return None

    def _build_observation_from_detection(self, out: dict) -> Dict[str, NumpyArrayType]:
        """Convert one WiLoR detection dict into the cached observation.

        Coordinate-frame notes (see module docstring for the full treatment):

        - ``pred_keypoints_3d`` is wrist-rooted MANO-local meters → stored
          verbatim as ``keypoints_3d_local``.
        - ``pred_cam_t_full`` is the camera-frame wrist translation (meters,
          x right / y down / z forward); adding it to the local keypoints
          yields camera-frame ``keypoints_3d``.
        - ``global_orient`` is a (1, 1, 3) axis-angle vector for the wrist
          rotation in camera frame; converted to a (3, 3) rotation matrix via
          ``scipy.spatial.transform.Rotation.from_rotvec`` and embedded in the
          (4, 4) ``wrist_pose`` with ``pred_cam_t_full`` as translation.
        - ``pred_keypoints_2d`` is (1, 21, 2) pixels (top-left origin).
        """
        # scipy is a wilor_mini dependency; import lazily here so the module
        # imports cleanly without it when connect=False.
        from scipy.spatial.transform import Rotation  # noqa: PLC0415 — lazy import

        preds = out["wilor_preds"]
        # Local (wrist-rooted) MANO keypoints — (1, 21, 3) -> (21, 3).
        kp_local = np.asarray(preds["pred_keypoints_3d"], dtype=np.float32).reshape(21, 3)
        # Camera-frame wrist translation — (1, 3) -> (3,).
        cam_t = np.asarray(preds["pred_cam_t_full"], dtype=np.float32).reshape(3)
        # Camera-frame keypoints = local + translation.
        kp_3d = kp_local + cam_t[None, :]
        # 2D pixel keypoints — (1, 21, 2) -> (21, 2).
        kp_2d = np.asarray(preds["pred_keypoints_2d"], dtype=np.float32).reshape(21, 2)
        # Wrist rotation from axis-angle — (1, 1, 3) -> (3,).
        global_orient = np.asarray(preds["global_orient"], dtype=np.float32).reshape(3)
        R = Rotation.from_rotvec(global_orient).as_matrix().astype(np.float32)
        wrist_pose = np.zeros((4, 4), dtype=np.float32)
        wrist_pose[:3, :3] = R
        wrist_pose[:3, 3] = cam_t
        wrist_pose[3, 3] = 1.0
        return {
            "keypoints_3d_local": kp_local.astype(np.float32, copy=False),
            "keypoints_3d": kp_3d.astype(np.float32, copy=False),
            "keypoints_2d": kp_2d.astype(np.float32, copy=False),
            "wrist_pose": wrist_pose,
            "hand_detected": np.ones((1,), dtype=np.float32),
        }

    # ========== WorldNode Implementation ==========

    @property
    def backend(self) -> ComputeBackend[NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType]:
        return NumpyComputeBackend

    @property
    def device(self) -> None:
        return None

    def post_environment_step(self, dt: float, *, priority: int = 0) -> None:
        """Read one BGR frame, run WiLoR, and cache the result as the observation.

        Lifecycle:

        - If not connected, this is a no-op (the zero observation is retained).
        - If ``cap.read()`` fails (e.g. camera disconnected), the previous
          keypoints/pose are held and ``hand_detected`` is set to 0.0.
        - If the pipeline returns no detection matching the configured
          handedness, the previous keypoints/pose are held and
          ``hand_detected`` is set to 0.0 (documented hold-last behavior).
        - Otherwise all entries are populated and ``hand_detected`` is 1.0.

        The frame convention is **BGR** (the native OpenCV order, and what
        WiLoR's own pipeline tests pass directly to ``pipe.predict``).
        """
        if not self.connect or self._cap is None or self._pipe is None:
            return

        ok, frame = self._cap.read()
        if not ok or frame is None:
            # Frame read failed — hold last geometry, mark no hand this tick.
            self._current_observation["hand_detected"] = np.zeros((1,), dtype=np.float32)
            return

        outputs = self._pipe.predict(frame)
        det = self._select_detection(outputs)
        if det is None or "wilor_preds" not in det:
            # No matching hand — hold last geometry, mark no hand this tick.
            self._current_observation["hand_detected"] = np.zeros((1,), dtype=np.float32)
            return

        self._current_observation = self._build_observation_from_detection(det)

    def after_reset(self, *, priority: int = 0, mask=None) -> None:
        self.post_environment_step(0.0, priority=priority)

    # after_reload defaults to calling after_reset in the WorldNode base class
    # (see node.py). We register its priority so the reload flow (used by the
    # first WorldEnv.reset) refreshes the observation. No override needed.

    def get_observation(self) -> Dict[str, NumpyArrayType]:
        return self._current_observation

    def set_next_action(self, action) -> None:
        """Accept and ignore any action.

        This node is observation-only (``action_space is None``); WorldEnv.step
        will not call this method because it guards on ``action_space is not
        None``. The no-op is provided so standalone WorldNode usage (which may
        call ``set_next_action`` directly) does not raise.
        """
        # Intentionally empty: observation-only node.
        return

    def close(self) -> None:
        """Best-effort teardown of the webcam capture."""
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                # Best-effort: never raise from close().
                pass
            self._cap = None
        # The WiLoR pipeline holds no OS resource we need to release explicitly.
        self._pipe = None

    # ========== Public helpers ==========

    def get_keypoints(self, local: bool = False) -> np.ndarray:
        """Return the latest (21, 3) hand keypoints as float32.

        Parameters
        ----------
        local : bool
            If ``True``, return the wrist-rooted MANO-local keypoints
            (``keypoints_3d_local``) — articulation-only, meters, invariant to
            wrist translation. If ``False`` (default), return the camera-frame
            keypoints (``keypoints_3d``): x right, y down, z forward, meters.
        """
        key = "keypoints_3d_local" if local else "keypoints_3d"
        return np.asarray(self._current_observation[key], dtype=np.float32)

    def get_wrist_pose(self) -> np.ndarray:
        """Return the latest (4, 4) camera-frame homogeneous wrist transform.

        Rotation comes from WiLoR's ``global_orient`` (axis-angle) via
        ``scipy.spatial.transform.Rotation.from_rotvec``; translation is
        ``pred_cam_t_full``. Camera frame: x right, y down, z forward, meters.
        """
        return np.asarray(self._current_observation["wrist_pose"], dtype=np.float32)

    def is_hand_detected(self) -> bool:
        """True if a hand of the configured handedness was found in the latest read."""
        return bool(self._current_observation["hand_detected"][0] >= 0.5)

    def get_keypoints_2d(self) -> np.ndarray:
        """Return the latest (21, 2) pixel keypoints (top-left origin, x right, y down)."""
        return np.asarray(self._current_observation["keypoints_2d"], dtype=np.float32)