"""Apple Vision Pro (AVP) hand/head tracking as an observation-only UniEnv WorldNode.

This node exposes the raw, unfiltered tracking stream coming from the AVP
``VisionProStreamer`` (hand-tracking gRPC on port 12345). It performs no
calibration, filtering or retargeting — those concerns belong in downstream
nodes. The :func:`convert_vp_to_mediapipe` helper is provided as a convenience
for retargeting pipelines that expect MediaPipe-style (21, 3) keypoints.
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

# VisionPro 25-joint layout -> MediaPipe 21-landmark layout. Faithful port of
# /tmp/tianji_teleop/wuji-retargeting/example/input_devices/visionpro.py.
VP_TO_MEDIAPIPE = (
    0, 1, 2, 3, 4,
    6, 7, 8, 9,
    11, 12, 13, 14,
    16, 17, 18, 19,
    21, 22, 23, 24,
)


def convert_vp_to_mediapipe(fingers_mat: np.ndarray) -> np.ndarray:
    """Map VisionPro (25, 4, 4) finger transforms to MediaPipe-style (21, 3) keypoints.

    Takes the [:3, 3] translation column of each selected finger transform, so the
    keypoints inherit the finger transforms' frame — wrist-local (meters), not
    world frame. The output is float32. This is a faithful port of the reference
    implementation in ``wuji-retargeting/example/input_devices/visionpro.py``.
    """
    fingers_mat = np.asarray(fingers_mat)
    mediapipe_pose = np.zeros((21, 3), dtype=np.float32)
    for mp_idx, vp_idx in enumerate(VP_TO_MEDIAPIPE):
        mediapipe_pose[mp_idx] = fingers_mat[vp_idx][:3, 3]
    return mediapipe_pose.astype(np.float32, copy=False)


class AVPTrackerNode(WorldNode[
    None, Dict[str, NumpyArrayType], None,
    NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
]):
    """Observation-only WorldNode streaming Apple Vision Pro hand/head tracking.

    The node is a pure sensor: ``action_space`` is ``None`` and any action passed
    to :meth:`set_next_action` is silently ignored. Observations are the raw AVP
    transforms (meters, X-right / Y-forward / Z-up after the ``avp_stream``
    YUP2ZUP conversion) plus pinch distances and wrist rolls. ``head``,
    ``left_wrist`` and ``right_wrist`` are world-frame (4, 4) transforms, while
    ``left_fingers`` / ``right_fingers`` are (25, 4, 4) transforms in the
    **wrist-local frame** (joint 0 is the wrist itself, at the origin). Compose
    ``wrist @ finger`` to obtain world-frame finger poses.

    When ``connect=False`` (or no frame has arrived yet) the cached observation is
    a zero dict of the correct shapes/dtypes, so the node can be constructed and
    driven through the full lifecycle without the ``avp_stream`` SDK or hardware.
    """

    # The first WorldEnv.reset routes through the reload flow, so after_reload
    # must also refresh the initial observation (mirrors the sibling adaptors).
    after_reset_priorities = {0}
    after_reload_priorities = {0}
    # Observation-only: nothing to send pre-step, so no pre_environment_step set.
    post_environment_step_priorities = {0}

    # Fixed hand-tracking gRPC endpoint on the AVP.
    _GRPC_PORT = 12345

    # Generous bounds for homogeneous transform entries (positions in meters).
    _TRANSFORM_BOUND = 1e6
    _PINCH_HIGH = 1.0  # meters
    _ROLL_LOW = -np.pi
    _ROLL_HIGH = np.pi

    def __init__(
        self,
        world: Optional[RealWorld] = None,
        name: str = "avp",
        ip: Optional[str] = None,
        *,
        record: bool = False,
        connect: bool = True,
        control_timestep: Optional[float] = 0.04,  # 25Hz
        update_timestep: Optional[float] = 0.04,  # background read frequency
    ):
        # Set WorldNode-related attributes first so backend/device properties work.
        self.name = name
        self.ip = ip
        self.record = bool(record)
        self.connect = bool(connect)

        # Hardware handle — only populated when connect=True. Initialized early so
        # close()/__del__ remain safe even if __init__ raises partway through.
        self._streamer = None
        # Whether at least one frame has ever been received from the streamer.
        self._has_frame = False

        if self.connect and ip is None:
            raise ValueError(
                "AVPTrackerNode(connect=True) requires an `ip` (the AVP's LAN IP "
                "address running the Tracking Streamer app)."
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
                "head": BoxSpace(
                    NumpyComputeBackend,
                    low=-self._TRANSFORM_BOUND,
                    high=self._TRANSFORM_BOUND,
                    dtype=np.float32,
                    shape=(4, 4),
                ),
                "left_wrist": BoxSpace(
                    NumpyComputeBackend,
                    low=-self._TRANSFORM_BOUND,
                    high=self._TRANSFORM_BOUND,
                    dtype=np.float32,
                    shape=(4, 4),
                ),
                "right_wrist": BoxSpace(
                    NumpyComputeBackend,
                    low=-self._TRANSFORM_BOUND,
                    high=self._TRANSFORM_BOUND,
                    dtype=np.float32,
                    shape=(4, 4),
                ),
                "left_fingers": BoxSpace(
                    NumpyComputeBackend,
                    low=-self._TRANSFORM_BOUND,
                    high=self._TRANSFORM_BOUND,
                    dtype=np.float32,
                    shape=(25, 4, 4),
                ),
                "right_fingers": BoxSpace(
                    NumpyComputeBackend,
                    low=-self._TRANSFORM_BOUND,
                    high=self._TRANSFORM_BOUND,
                    dtype=np.float32,
                    shape=(25, 4, 4),
                ),
                "left_pinch_distance": BoxSpace(
                    NumpyComputeBackend,
                    low=0.0,
                    high=self._PINCH_HIGH,
                    dtype=np.float32,
                    shape=(1,),
                ),
                "right_pinch_distance": BoxSpace(
                    NumpyComputeBackend,
                    low=0.0,
                    high=self._PINCH_HIGH,
                    dtype=np.float32,
                    shape=(1,),
                ),
                "left_wrist_roll": BoxSpace(
                    NumpyComputeBackend,
                    low=self._ROLL_LOW,
                    high=self._ROLL_HIGH,
                    dtype=np.float32,
                    shape=(1,),
                ),
                "right_wrist_roll": BoxSpace(
                    NumpyComputeBackend,
                    low=self._ROLL_LOW,
                    high=self._ROLL_HIGH,
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
        """Open the VisionProStreamer (lazy import of the avp_stream SDK)."""
        try:
            from avp_stream import VisionProStreamer  # noqa: PLC0415 — lazy import
        except ImportError as e:  # pragma: no cover — requires hardware/SDK
            raise ImportError(
                "The 'avp_stream' package is required to connect to an Apple "
                "Vision Pro Tracking Streamer. Install it with: "
                'pip install "unienv-input-devices[avp]"'
            ) from e

        self._streamer = VisionProStreamer(ip=self.ip, record=self.record)

    # ========== Internal helpers ==========

    def _zero_observation(self) -> Dict[str, NumpyArrayType]:
        """Build the all-zeros observation dict of the correct shapes/dtypes."""
        return {
            "head": np.zeros((4, 4), dtype=np.float32),
            "left_wrist": np.zeros((4, 4), dtype=np.float32),
            "right_wrist": np.zeros((4, 4), dtype=np.float32),
            "left_fingers": np.zeros((25, 4, 4), dtype=np.float32),
            "right_fingers": np.zeros((25, 4, 4), dtype=np.float32),
            "left_pinch_distance": np.zeros((1,), dtype=np.float32),
            "right_pinch_distance": np.zeros((1,), dtype=np.float32),
            "left_wrist_roll": np.zeros((1,), dtype=np.float32),
            "right_wrist_roll": np.zeros((1,), dtype=np.float32),
        }

    def _build_observation(self, latest: Optional[dict]) -> Dict[str, NumpyArrayType]:
        """Convert a raw ``streamer.latest`` dict into the cached observation.

        If ``latest`` is ``None`` (no frame yet), the previously cached observation
        is preserved — callers should handle that case before invoking this.
        """
        # Squeeze the leading batch dim of the (1, 4, 4) head/wrist transforms.
        head = np.asarray(latest["head"], dtype=np.float32).reshape(4, 4)
        left_wrist = np.asarray(latest["left_wrist"], dtype=np.float32).reshape(4, 4)
        right_wrist = np.asarray(latest["right_wrist"], dtype=np.float32).reshape(4, 4)
        # Finger transforms are (25, 4, 4) already — copy as-is.
        left_fingers = np.array(latest["left_fingers"], dtype=np.float32, copy=True)
        right_fingers = np.array(latest["right_fingers"], dtype=np.float32, copy=True)
        # Scalars -> shape-(1,) arrays.
        left_pinch = np.asarray(latest["left_pinch_distance"], dtype=np.float32).reshape(1)
        right_pinch = np.asarray(latest["right_pinch_distance"], dtype=np.float32).reshape(1)
        left_roll = np.asarray(latest["left_wrist_roll"], dtype=np.float32).reshape(1)
        right_roll = np.asarray(latest["right_wrist_roll"], dtype=np.float32).reshape(1)
        return {
            "head": head,
            "left_wrist": left_wrist,
            "right_wrist": right_wrist,
            "left_fingers": left_fingers,
            "right_fingers": right_fingers,
            "left_pinch_distance": left_pinch,
            "right_pinch_distance": right_pinch,
            "left_wrist_roll": left_roll,
            "right_wrist_roll": right_roll,
        }

    # ========== WorldNode Implementation ==========

    @property
    def backend(self) -> ComputeBackend[NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType]:
        return NumpyComputeBackend

    @property
    def device(self) -> None:
        return None

    def post_environment_step(self, dt: float, *, priority: int = 0) -> None:
        """Read the latest AVP frame and cache it as the current observation.

        If no frame has arrived yet (``streamer.latest`` is ``None``) the previous
        cached observation is retained (zeros initially). When not connected, this
        is effectively a no-op that keeps the zero observation.
        """
        if not self.connect or self._streamer is None:
            return
        latest = self._streamer.latest
        if latest is None:
            return
        self._has_frame = True
        self._current_observation = self._build_observation(latest)

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
        """Best-effort teardown of the streamer."""
        if self._streamer is not None:
            try:
                self._streamer.cleanup()
            except Exception:
                # Best-effort: never raise from close().
                pass
            self._streamer = None

    # ========== Public helpers ==========

    def get_wrist_pose(self, side: Literal["left", "right"]) -> np.ndarray:
        """Return the (4, 4) homogeneous wrist transform from the cached observation."""
        key = f"{side}_wrist"
        return np.asarray(self._current_observation[key], dtype=np.float32)

    def get_finger_keypoints(self, side: Literal["left", "right"]) -> np.ndarray:
        """Return the (25, 3) translation columns of the cached finger transforms.

        These are wrist-local coordinates (meters); joint 0 is the wrist at the
        origin. Compose with the wrist transform for world-frame keypoints.
        """
        key = f"{side}_fingers"
        fingers = np.asarray(self._current_observation[key], dtype=np.float32)
        return fingers[:, :3, 3].astype(np.float32, copy=False)

    def get_pinch_distance(self, side: Literal["left", "right"]) -> float:
        """Return the latest pinch distance (meters) for the given hand."""
        key = f"{side}_pinch_distance"
        return float(self._current_observation[key][0])

    def is_streaming(self) -> bool:
        """True if at least one frame has ever been received from the streamer."""
        return self._has_frame