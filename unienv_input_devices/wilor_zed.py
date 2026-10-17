"""WiLoR hand tracking fused with ZED stereo depth, as an observation-only node.

This module provides :class:`WiLoRZedHandNode`, a subclass of
:class:`unienv_input_devices.wilor.WiLoRHandNode` that consumes frames from a
**ZED stereo camera** instead of a UVC webcam and corrects WiLoR's weakest
output — the metric wrist translation — with the ZED's metric depth map.

Why this exists
---------------
WiLoR (like every monocular hand-mesh model) regresses a *weak-perspective*
camera: finger articulation (``keypoints_3d_local`` / ``keypoints_3d_wrist``)
is accurate and temporally stable, but the camera-frame wrist translation
``pred_cam_t_full`` is scale-ambiguous in Z (documented ~5-15 cm error in
``docs/wilor.md``). The wrist translation is exactly what drives the arm EEF
in the teleop retargeting path, so a ZED's calibrated stereo depth is used to
re-anchor it:

1. The ZED's left rectified RGB feeds WiLoR (RGB↔depth are pixel-aligned, and
   the true intrinsics ``fx`` replaces the hand-tuned ``focal_length``).
2. WiLoR's 2D wrist keypoint is deprojected through the depth map (robust
   neighborhood median) to a metric 3D wrist position.
3. If the correction magnitude is within ``max_depth_correction`` (outlier
   guard for occluded/invalid depth), the wrist translation and the
   camera-frame ``keypoints_3d`` are shifted by the correction delta.
   Articulation signals (``keypoints_3d_local`` / ``keypoints_3d_wrist``) are
   never touched.

The ``zed_camera`` argument is duck-typed: any object whose
``get_observation()`` returns a dict with

- ``left_rgb``: uint8 (H, W, 3) RGB,
- ``left_depth``: float32 (H, W) meters (0 or NaN = invalid),
- ``left_intrinsics``: float32 (3, 3) pinhole intrinsics of the left stream,

works — e.g. ``unienv_zed.ZedCameraNode``. The node ticks the camera
(``post_environment_step``) before each read, exactly like ``cap.read()``
grabs a fresh webcam frame in :class:`WiLoRHandNode` — so do **not** register
the same ZED node in the world env when it serves as this node's capture
device. Ownership stays with the caller; :meth:`close` never closes the ZED
camera. All frame conventions (camera frame x right / y down / z forward,
MediaPipe-21 keypoint order) are inherited unchanged from
:class:`WiLoRHandNode`; see that module's docstring.

Licensing: WiLoR weights are CC-BY-NC-ND and MANO assets are non-commercial
(see ``wilor.py``); this module vendors neither.
"""

from typing import Dict, Literal, Optional

import numpy as np

from unienv_interface.world import RealWorld
from unienv_interface.backends.numpy import NumpyArrayType

from .wilor import WiLoRHandNode


class WiLoRZedHandNode(WiLoRHandNode):
    """WiLoR hand tracking on ZED frames with metric depth-corrected wrist.

    Parameters
    ----------
    zed_camera
        A connected ZED camera node (see module docstring for the observation
        contract). Required; use :class:`WiLoRHandNode` for UVC webcams.
    max_depth_correction
        Maximum allowed depth-based wrist-translation correction in meters
        (default 0.15). Larger corrections are rejected as depth outliers and
        the monocular estimate is kept for that frame.
    focal_length
        Placeholder only — when ``connect=True`` it is overridden by the true
        horizontal focal length from the ZED intrinsics before the WiLoR
        pipeline is constructed. Pass an explicit value only to force a
        specific focal (e.g. when feeding non-rectified frames).

    All other parameters (``hand``, ``device``, ``connect``,
    ``control_timestep``, ``update_timestep``, ...) match
    :class:`WiLoRHandNode`. ``camera_id``/``width``/``height``/``fps``/
    ``fourcc`` do not exist here — the ZED node owns its own streaming
    configuration.
    """

    # Depth sampling parameters for the wrist pixel neighborhood.
    _DEPTH_HALF_WINDOW = 3  # 7x7 pixel window around the wrist keypoint
    _DEPTH_MIN_M = 0.1  # ZED minimum range guard (also rejects 0=invalid)
    _DEPTH_MAX_M = 5.0

    def __init__(
        self,
        world: Optional[RealWorld] = None,
        name: str = "wilor_zed_hand",
        zed_camera=None,
        *,
        hand: Literal["left", "right", "both"] = "right",
        max_depth_correction: float = 0.15,
        focal_length: Optional[float] = None,
        device: Optional[str] = None,
        connect: bool = True,
        control_timestep: Optional[float] = 0.04,  # 25Hz
        update_timestep: Optional[float] = 0.04,  # frame-read frequency
    ):
        if zed_camera is None:
            raise ValueError(
                "WiLoRZedHandNode requires a `zed_camera` (any object whose "
                "get_observation() returns left_rgb/left_depth/left_intrinsics, "
                "e.g. unienv_zed.ZedCameraNode). For UVC webcams use "
                "WiLoRHandNode instead."
            )
        self.zed_camera = zed_camera
        self.max_depth_correction = float(max_depth_correction)
        self._zed_intrinsics: Optional[np.ndarray] = None
        # True intrinsics win unless the caller explicitly forces a focal.
        self._force_focal_length = focal_length is not None
        super().__init__(
            world=world,
            name=name,
            camera_id=0,  # unused — no UVC capture in this node
            hand=hand,
            focal_length=(
                float(focal_length)
                if focal_length is not None
                else WiLoRHandNode._DEFAULT_FOCAL_LENGTH
            ),
            width=None,
            height=None,
            fps=None,
            fourcc=None,
            device=device,
            connect=connect,
            control_timestep=control_timestep,
            update_timestep=update_timestep,
        )

    def _metadata_source(self) -> Optional[str]:
        """The ZED camera feeding the node (its node ``name``), not the unused ``camera_id``."""
        return str(getattr(self.zed_camera, "name", None) or "zed")

    # ========== Hardware connection ==========

    def _connect_hardware(self) -> None:
        """Read ZED intrinsics and construct the WiLoR pipeline (no webcam)."""
        try:
            import torch  # noqa: PLC0415 — lazy import
        except ImportError as e:  # pragma: no cover — requires torch
            raise ImportError(
                "WiLoRZedHandNode(connect=True) requires PyTorch. Install it with "
                "`pip install torch` (a CUDA build is strongly recommended)."
            ) from e

        if self._device_input is None:
            self._torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self._torch_device = self._device_input

        # Pinhole intrinsics of the ZED's left (rectified) stream; also gives
        # the true fx for WiLoR's weak-perspective scale.
        obs = self.zed_camera.get_observation()
        K = np.asarray(obs["left_intrinsics"], dtype=np.float64).reshape(3, 3)
        self._zed_intrinsics = K
        if not self._force_focal_length:
            self.focal_length = float(K[0, 0])

        # --- WiLoR-mini pipeline ---
        try:
            from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (  # noqa: PLC0415 — lazy import
                WiLorHandPose3dEstimationPipeline,
            )
        except ImportError as e:  # pragma: no cover — requires wilor_mini
            raise ImportError(
                "WiLoRZedHandNode(connect=True) requires the 'wilor_mini' "
                "package. Install it via this package's 'wilor' extra "
                "(--no-build-isolation is needed for chumpy's legacy setup.py; "
                "see docs/wilor.md). Model weights and MANO assets "
                "auto-download from HuggingFace on first use (CC-BY-NC-ND / "
                "non-commercial terms — see README)."
            ) from e

        self._cap = None  # no UVC capture in this node
        self._pipe = WiLorHandPose3dEstimationPipeline(
            device=torch.device(self._torch_device),
            dtype=torch.float16,
            verbose=False,
            focal_length=self.focal_length,
        )

    # ========== Depth correction ==========

    @classmethod
    def _sample_depth(cls, depth: np.ndarray, u: float, v: float) -> Optional[float]:
        """Median valid depth (meters) in a small window around pixel (u, v).

        Returns ``None`` when no valid sample exists (all 0/NaN/out of range),
        in which case the caller keeps the monocular estimate.
        """
        h, w = depth.shape[:2]
        x, y = int(round(u)), int(round(v))
        x0, x1 = max(0, x - cls._DEPTH_HALF_WINDOW), min(w, x + cls._DEPTH_HALF_WINDOW + 1)
        y0, y1 = max(0, y - cls._DEPTH_HALF_WINDOW), min(h, y + cls._DEPTH_HALF_WINDOW + 1)
        window = depth[y0:y1, x0:x1].reshape(-1)
        valid = window[
            np.isfinite(window)
            & (window >= cls._DEPTH_MIN_M)
            & (window <= cls._DEPTH_MAX_M)
        ]
        if valid.size == 0:
            return None
        return float(np.median(valid))

    def _apply_depth_correction(
        self,
        hand_obs: Dict[str, NumpyArrayType],
        hand: str,
        depth: Optional[np.ndarray],
    ) -> Dict[str, NumpyArrayType]:
        """Re-anchor the wrist translation to the metric ZED depth, in place.

        Deprojects the wrist 2D keypoint with the ZED intrinsics and the
        median local depth. If the correction exceeds ``max_depth_correction``
        (or no valid depth exists), the monocular estimate is kept. Only
        ``wrist_pose`` translation and camera-frame ``keypoints_3d`` are
        shifted — the articulation signals are translation-invariant already.
        """
        if depth is None or self._zed_intrinsics is None:
            return hand_obs

        kp_2d = np.asarray(hand_obs[self._obs_key(hand, "keypoints_2d")], dtype=np.float32)
        z = self._sample_depth(depth, float(kp_2d[0, 0]), float(kp_2d[0, 1]))
        if z is None:
            return hand_obs

        K = self._zed_intrinsics
        wrist_metric = np.array(
            [
                (float(kp_2d[0, 0]) - K[0, 2]) * z / K[0, 0],
                (float(kp_2d[0, 1]) - K[1, 2]) * z / K[1, 1],
                z,
            ],
            dtype=np.float32,
        )
        pose_key = self._obs_key(hand, "wrist_pose")
        cam_t = np.asarray(hand_obs[pose_key][:3, 3], dtype=np.float32).copy()
        delta = wrist_metric - cam_t
        if float(np.linalg.norm(delta)) > self.max_depth_correction:
            return hand_obs

        hand_obs[pose_key][:3, 3] = wrist_metric
        kp3d_key = self._obs_key(hand, "keypoints_3d")
        hand_obs[kp3d_key] = (
            np.asarray(hand_obs[kp3d_key], dtype=np.float32) + delta[None, :]
        ).astype(np.float32)
        return hand_obs

    # ========== WorldNode Implementation ==========

    def post_environment_step(self, dt: float, *, priority: int = 0) -> None:
        """Grab a ZED frame, run WiLoR on it, and depth-correct the wrist.

        Same lifecycle semantics as :meth:`WiLoRHandNode.post_environment_step`
        (hold-last geometry per hand, ``hand_detected`` gating), except the
        frame comes from ``zed_camera.get_observation()``: ``left_rgb`` is
        converted RGB→BGR for the WiLoR pipeline and ``left_depth`` re-anchors
        the wrist translation when valid.
        """
        if not self.connect or self._pipe is None:
            return

        # Grab a fresh ZED frame+depth before reading (the camera node's
        # get_observation() only returns its last grab).
        self.zed_camera.post_environment_step(dt)
        obs = self.zed_camera.get_observation()
        rgb = np.asarray(obs["left_rgb"])
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
            # Frame read failed — hold last geometry, mark no hands this tick.
            for h in self._active_hands:
                self._current_observation[self._obs_key(h, "hand_detected")] = np.zeros((1,), dtype=np.float32)
            return
        # WiLoR expects BGR (native OpenCV order); the ZED node emits RGB.
        frame = np.ascontiguousarray(rgb[:, :, ::-1])
        depth = obs.get("left_depth")
        depth = None if depth is None else np.asarray(depth, dtype=np.float32)

        outputs = self._pipe.predict(frame)
        for h in self._active_hands:
            det = self._select_detection(outputs, h)
            if det is None or "wilor_preds" not in det:
                # No matching hand — hold this hand's geometry, mark it absent.
                self._current_observation[self._obs_key(h, "hand_detected")] = np.zeros((1,), dtype=np.float32)
                continue
            hand_obs = self._build_observation_from_detection(det, h)
            hand_obs = self._apply_depth_correction(hand_obs, h, depth)
            # Merge (not replace) so the other hand's held geometry survives.
            self._current_observation.update(hand_obs)
