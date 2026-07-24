"""Smoke test for the UniEnv input-devices adaptor (Apple Vision Pro node).

Exercises the :class:`unienv_input_devices.AVPTrackerNode` node **directly**
(constructed with ``connect=False``) without going through ``WorldEnv`` or
``RealWorld``. This is required because the ``unienv`` package published on
PyPI predates a recent local core fix (``RealWorld.reset`` kwargs), so the
node is driven through its lifecycle hooks manually.
"""

import numpy as np
import pytest

import unienv_input_devices
from unienv_input_devices import AVPTrackerNode, convert_vp_to_mediapipe


EXPECTED_OBS_KEYS = {
    "head",
    "left_wrist",
    "right_wrist",
    "left_fingers",
    "right_fingers",
    "left_pinch_distance",
    "right_pinch_distance",
    "left_wrist_roll",
    "right_wrist_roll",
}

EXPECTED_SHAPES_DTYPES = {
    "head": ((4, 4), np.float32),
    "left_wrist": ((4, 4), np.float32),
    "right_wrist": ((4, 4), np.float32),
    "left_fingers": ((25, 4, 4), np.float32),
    "right_fingers": ((25, 4, 4), np.float32),
    "left_pinch_distance": ((1,), np.float32),
    "right_pinch_distance": ((1,), np.float32),
    "left_wrist_roll": ((1,), np.float32),
    "right_wrist_roll": ((1,), np.float32),
}


def test_avp_tracker_node_smoke():
    node = AVPTrackerNode(connect=False)
    try:
        # Observation-only node: no action space.
        assert node.action_space is None

        # Observation space has the 9 expected keys with the right shapes.
        obs_space = node.observation_space
        assert set(obs_space.spaces.keys()) == EXPECTED_OBS_KEYS
        for key, (shape, dtype) in EXPECTED_SHAPES_DTYPES.items():
            sub = obs_space.spaces[key]
            assert sub.shape == shape, f"{key}: {sub.shape} != {shape}"
            assert sub.dtype == dtype, f"{key}: {sub.dtype} != {dtype}"

        # after_reset runs and yields a populated observation dict.
        node.after_reset()
        obs = node.get_observation()
        assert obs is not None
        assert set(obs.keys()) == EXPECTED_OBS_KEYS
        for key, (shape, dtype) in EXPECTED_SHAPES_DTYPES.items():
            arr = np.asarray(obs[key])
            assert arr.shape == shape, f"{key}: {arr.shape} != {shape}"
            assert arr.dtype == dtype, f"{key}: {arr.dtype} != {dtype}"

        # post_environment_step runs without error.
        node.post_environment_step(0.04)
    finally:
        # close() must run cleanly even when never connected.
        node.close()


@pytest.mark.parametrize(
    "fingers",
    [
        np.zeros((25, 4, 4), dtype=np.float64),
        np.random.RandomState(0).randn(25, 4, 4).astype(np.float64),
    ],
)
def test_convert_vp_to_mediapipe(fingers):
    out = convert_vp_to_mediapipe(fingers)
    assert isinstance(out, np.ndarray)
    assert out.shape == (21, 3)
    assert out.dtype == np.float32


def test_public_api_exposed():
    assert hasattr(unienv_input_devices, "AVPTrackerNode")
    assert hasattr(unienv_input_devices, "convert_vp_to_mediapipe")


# ====================== WiLoRHandNode smoke tests ======================

from unienv_input_devices import WiLoRHandNode  # noqa: E402

WILORED_EXPECTED_OBS_KEYS = {
    "keypoints_3d_local",
    "keypoints_3d",
    "keypoints_2d",
    "wrist_pose",
    "hand_detected",
}

WILORED_EXPECTED_SHAPES_DTYPES = {
    "keypoints_3d_local": ((21, 3), np.float32),
    "keypoints_3d": ((21, 3), np.float32),
    "keypoints_2d": ((21, 2), np.float32),
    "wrist_pose": ((4, 4), np.float32),
    "hand_detected": ((1,), np.float32),
}


def test_wilor_hand_node_smoke():
    # connect=False must NOT require torch/cv2/scipy/wilor_mini.
    node = WiLoRHandNode(connect=False)
    try:
        # Observation-only node: no action space.
        assert node.action_space is None

        # Observation space has the 5 expected keys with the right shapes/dtypes.
        obs_space = node.observation_space
        assert set(obs_space.spaces.keys()) == WILORED_EXPECTED_OBS_KEYS
        for key, (shape, dtype) in WILORED_EXPECTED_SHAPES_DTYPES.items():
            sub = obs_space.spaces[key]
            assert sub.shape == shape, f"{key}: {sub.shape} != {shape}"
            assert sub.dtype == dtype, f"{key}: {sub.dtype} != {dtype}"

        # after_reset runs and yields a populated observation dict.
        node.after_reset()
        obs = node.get_observation()
        assert obs is not None
        assert set(obs.keys()) == WILORED_EXPECTED_OBS_KEYS
        for key, (shape, dtype) in WILORED_EXPECTED_SHAPES_DTYPES.items():
            arr = np.asarray(obs[key])
            assert arr.shape == shape, f"{key}: {arr.shape} != {shape}"
            assert arr.dtype == dtype, f"{key}: {arr.dtype} != {dtype}"

        # hand_detected starts at 0.0 (no hand seen yet).
        assert float(obs["hand_detected"][0]) == 0.0
        assert node.is_hand_detected() is False

        # post_environment_step runs without error (no-op when not connected).
        node.post_environment_step(0.04)

        # Helpers return the correct shapes/dtypes.
        assert node.get_keypoints().shape == (21, 3)
        assert node.get_keypoints(local=True).shape == (21, 3)
        assert node.get_keypoints().dtype == np.float32
        assert node.get_wrist_pose().shape == (4, 4)
        assert node.get_wrist_pose().dtype == np.float32
        assert node.get_keypoints_2d().shape == (21, 2)
        assert node.get_keypoints_2d().dtype == np.float32
    finally:
        # close() must run cleanly even when never connected.
        node.close()


def test_wilor_hand_node_rejects_invalid_hand():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        WiLoRHandNode(connect=False, hand="both")
