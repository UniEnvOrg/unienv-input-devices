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
    "keypoints_3d_wrist",
    "keypoints_3d",
    "keypoints_2d",
    "wrist_pose",
    "hand_detected",
}

WILORED_EXPECTED_SHAPES_DTYPES = {
    "keypoints_3d_local": ((21, 3), np.float32),
    "keypoints_3d_wrist": ((21, 3), np.float32),
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
        assert node.get_keypoints_wrist().shape == (21, 3)
        assert node.get_keypoints_wrist().dtype == np.float32
    finally:
        # close() must run cleanly even when never connected.
        node.close()


def test_wilor_keypoints_wrist_frame_math():
    """keypoints_3d_wrist must satisfy kp_3d == kp_wrist @ R.T + cam_t."""
    from scipy.spatial.transform import Rotation

    rng = np.random.RandomState(0)
    kp_local = rng.randn(1, 21, 3).astype(np.float64) * 0.05
    cam_t = np.array([[0.05, 0.10, 0.42]])
    rotvec = rng.randn(1, 1, 3) * 0.7
    preds = {
        "pred_keypoints_3d": kp_local,
        "pred_cam_t_full": cam_t,
        "pred_keypoints_2d": np.zeros((1, 21, 2)),
        "global_orient": rotvec,
    }
    node = WiLoRHandNode(connect=False)
    try:
        obs = node._build_observation_from_detection({"wilor_preds": preds}, "right")
    finally:
        node.close()

    R = Rotation.from_rotvec(rotvec.reshape(3)).as_matrix()
    kp_local_2d = kp_local.reshape(21, 3)
    # Forward: wrist-frame keypoints are R.T @ kp_local (per row: kp_local @ R).
    np.testing.assert_allclose(obs["keypoints_3d_wrist"], kp_local_2d @ R, atol=1e-5)
    # Inverse: camera-frame keypoints recover from wrist frame + wrist pose.
    recon = obs["keypoints_3d_wrist"] @ R.T + obs["wrist_pose"][:3, 3]
    np.testing.assert_allclose(recon, obs["keypoints_3d"], atol=1e-5)
    # Sanity: wrist joint itself stays at the local origin in both frames.
    np.testing.assert_allclose(obs["keypoints_3d_wrist"][0], kp_local_2d[0] @ R, atol=1e-5)


def test_wilor_hand_node_rejects_invalid_hand():
    import pytest as _pytest
    with _pytest.raises(ValueError):
        WiLoRHandNode(connect=False, hand="middle")


def test_wilor_hand_node_both_hands_smoke():
    # connect=False must NOT require torch/cv2/scipy/wilor_mini.
    node = WiLoRHandNode(connect=False, hand="both")
    try:
        assert node.action_space is None

        expected_keys = {
            f"{h}_{base}"
            for h in ("left", "right")
            for base in WILORED_EXPECTED_OBS_KEYS
        }
        expected_shapes = {
            f"{h}_{base}": v
            for h in ("left", "right")
            for base, v in WILORED_EXPECTED_SHAPES_DTYPES.items()
        }

        obs_space = node.observation_space
        assert set(obs_space.spaces.keys()) == expected_keys
        for key, (shape, dtype) in expected_shapes.items():
            sub = obs_space.spaces[key]
            assert sub.shape == shape, f"{key}: {sub.shape} != {shape}"
            assert sub.dtype == dtype, f"{key}: {sub.dtype} != {dtype}"

        node.after_reset()
        obs = node.get_observation()
        assert obs is not None
        assert set(obs.keys()) == expected_keys
        for key, (shape, dtype) in expected_shapes.items():
            arr = np.asarray(obs[key])
            assert arr.shape == shape, f"{key}: {arr.shape} != {shape}"
            assert arr.dtype == dtype, f"{key}: {arr.dtype} != {dtype}"

        assert node.is_hand_detected("left") is False
        assert node.is_hand_detected("right") is False

        node.post_environment_step(0.04)

        # Helpers require an explicit hand in both-mode.
        for h in ("left", "right"):
            assert node.get_keypoints(hand=h).shape == (21, 3)
            assert node.get_keypoints(local=True, hand=h).shape == (21, 3)
            assert node.get_keypoints_wrist(hand=h).shape == (21, 3)
            assert node.get_wrist_pose(hand=h).shape == (4, 4)
            assert node.get_keypoints_2d(hand=h).shape == (21, 2)

        import pytest as _pytest
        with _pytest.raises(ValueError):
            node.get_keypoints()
        with _pytest.raises(ValueError):
            node.is_hand_detected()
        with _pytest.raises(ValueError):
            node.get_keypoints(hand="middle")
    finally:
        node.close()


def test_wilor_hand_node_single_hand_rejects_inactive_hand():
    node = WiLoRHandNode(connect=False, hand="right")
    try:
        import pytest as _pytest
        with _pytest.raises(ValueError):
            node.get_keypoints(hand="left")
    finally:
        node.close()
