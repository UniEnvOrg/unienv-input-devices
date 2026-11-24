"""Hardware-free tests for WiLoRZedHandNode (WiLoR + ZED depth fusion).

The ZED camera and the WiLoR pipeline are both mocked: a fake ZED node serves
static RGB/depth/intrinsics observations, and a fake pipeline returns canned
WiLoR detections while recording the frames it was given. No torch, cv2,
pyzed or wilor_mini is required.
"""

import numpy as np
import pytest

from unienv_input_devices import WiLoRHandNode, WiLoRZedHandNode


# 200x200 test image; principal point at the wrist pixel so deprojection is
# trivially (0, 0, z).
_H, _W = 200, 200
_FX = _FY = 500.0
_CX = _CY = 100.0
_K = np.array(
    [[_FX, 0.0, _CX], [0.0, _FY, _CY], [0.0, 0.0, 1.0]], dtype=np.float32
)

_WRIST_UV = (100.0, 100.0)
# Monocular estimate: off by a few cm in every axis (within the 0.15 m clamp).
_MONO_CAM_T = np.array([0.02, 0.03, 0.55], dtype=np.float32)
_TRUE_DEPTH = 0.5


class FakeZedCamera:
    """Duck-typed stand-in for unienv_zed.ZedCameraNode."""

    def __init__(self, depth_value=_TRUE_DEPTH):
        rgb = np.zeros((_H, _W, 3), dtype=np.uint8)
        rgb[..., 0] = 1  # R
        rgb[..., 1] = 2  # G
        rgb[..., 2] = 3  # B
        depth = np.full((_H, _W), depth_value, dtype=np.float32)
        self._obs = {
            "left_rgb": rgb,
            "left_depth": depth,
            "left_intrinsics": _K.copy(),
        }
        self.ticks = 0

    def post_environment_step(self, dt, *, priority=0):
        self.ticks += 1

    def get_observation(self):
        return self._obs


class FakePipe:
    """Stand-in for WiLorHandPose3dEstimationPipeline.predict."""

    def __init__(self, detections):
        self.detections = detections
        self.frames = []

    def predict(self, frame):
        self.frames.append(frame.copy())
        return self.detections


def _make_detection(*, is_right=1.0):
    kp_local = np.zeros((1, 21, 3), dtype=np.float32)
    kp_local[0, 1] = (0.01, 0.02, 0.03)  # deterministic non-wrist keypoint
    kp_2d = np.zeros((1, 21, 2), dtype=np.float32)
    kp_2d[0, 0] = _WRIST_UV
    return {
        "is_right": float(is_right),
        "wilor_preds": {
            "pred_keypoints_3d": kp_local,
            "pred_cam_t_full": _MONO_CAM_T.reshape(1, 3).copy(),
            "pred_keypoints_2d": kp_2d,
            "global_orient": np.zeros((1, 1, 3), dtype=np.float32),
        },
    }


def _make_node(zed, pipe, *, hand="right", max_depth_correction=0.15):
    """Construct the node without touching hardware, then inject the fakes."""
    node = WiLoRZedHandNode(
        zed_camera=zed,
        hand=hand,
        max_depth_correction=max_depth_correction,
        connect=False,
    )
    node.connect = True
    node._pipe = pipe
    node._zed_intrinsics = _K.astype(np.float64)
    return node


def test_requires_zed_camera():
    with pytest.raises(ValueError, match="zed_camera"):
        WiLoRZedHandNode(zed_camera=None, connect=False)


def test_depth_correction_reanchors_wrist():
    node = _make_node(FakeZedCamera(), FakePipe([_make_detection()]))
    node.post_environment_step(0.04)

    wrist_pose = node.get_wrist_pose()
    # Wrist pixel is at the principal point -> deprojects to (0, 0, z).
    np.testing.assert_allclose(
        wrist_pose[:3, 3], [0.0, 0.0, _TRUE_DEPTH], atol=1e-6
    )

    # Camera-frame keypoints shifted by the same delta; articulation intact.
    kp_3d = node.get_keypoints()
    kp_local = node.get_keypoints(local=True)
    np.testing.assert_allclose(kp_3d, kp_local + np.array([0.0, 0.0, _TRUE_DEPTH]), atol=1e-6)
    np.testing.assert_allclose(kp_local[1], [0.01, 0.02, 0.03], atol=1e-6)
    assert node.is_hand_detected()


def test_invalid_depth_keeps_monocular_estimate():
    node = _make_node(FakeZedCamera(depth_value=0.0), FakePipe([_make_detection()]))
    node.post_environment_step(0.04)

    np.testing.assert_allclose(node.get_wrist_pose()[:3, 3], _MONO_CAM_T, atol=1e-6)
    assert node.is_hand_detected()


def test_oversized_correction_is_rejected():
    # 5.0 m is beyond _DEPTH_MAX_M and would also exceed the clamp.
    node = _make_node(FakeZedCamera(depth_value=5.0), FakePipe([_make_detection()]))
    node.post_environment_step(0.04)
    np.testing.assert_allclose(node.get_wrist_pose()[:3, 3], _MONO_CAM_T, atol=1e-6)

    # In-range depth but beyond max_depth_correction -> still monocular.
    node = _make_node(
        FakeZedCamera(depth_value=1.0),
        FakePipe([_make_detection()]),
        max_depth_correction=0.05,
    )
    node.post_environment_step(0.04)
    np.testing.assert_allclose(node.get_wrist_pose()[:3, 3], _MONO_CAM_T, atol=1e-6)


def test_rgb_frame_is_converted_to_bgr_for_wilor():
    pipe = FakePipe([_make_detection()])
    node = _make_node(FakeZedCamera(), pipe)
    node.post_environment_step(0.04)

    frame = pipe.frames[0]
    # Fake ZED serves R=1, G=2, B=3; WiLoR must receive BGR.
    assert tuple(frame[0, 0]) == (3, 2, 1)


def test_both_hands_mode_prefix_and_hold_last():
    # Only the right hand is detected.
    node = _make_node(FakeZedCamera(), FakePipe([_make_detection(is_right=1.0)]), hand="both")
    node.post_environment_step(0.04)

    assert node.is_hand_detected("right")
    assert not node.is_hand_detected("left")
    np.testing.assert_allclose(
        node.get_wrist_pose("right")[:3, 3], [0.0, 0.0, _TRUE_DEPTH], atol=1e-6
    )
    # Undetected hand holds its (zero) geometry.
    np.testing.assert_allclose(node.get_wrist_pose("left"), np.zeros((4, 4)), atol=1e-6)


def test_no_detection_marks_absent_and_holds_geometry():
    pipe = FakePipe([])  # no detections at all
    node = _make_node(FakeZedCamera(), pipe)
    node.post_environment_step(0.04)
    assert not node.is_hand_detected()
    np.testing.assert_allclose(node.get_wrist_pose(), np.zeros((4, 4)), atol=1e-6)


def test_zed_camera_is_ticked_before_reading():
    zed = FakeZedCamera()
    node = _make_node(zed, FakePipe([_make_detection()]))
    assert zed.ticks == 0
    node.post_environment_step(0.04)
    assert zed.ticks == 1


def test_observation_space_matches_wilor_hand_node():
    zed_node = WiLoRZedHandNode(zed_camera=FakeZedCamera(), hand="both", connect=False)
    webcam_node = WiLoRHandNode(hand="both", connect=False)
    assert set(zed_node.observation_space.spaces) == set(webcam_node.observation_space.spaces)
    for key, sub in zed_node.observation_space.spaces.items():
        other = webcam_node.observation_space.spaces[key]
        assert sub.shape == other.shape
        assert sub.dtype == other.dtype


def test_observation_space_metadata_names_the_zed_camera():
    from unienv_interface.space.space_utils.metadata_utils import validate_metadata

    zed = FakeZedCamera()
    zed.name = "zed_front"
    zed_space = WiLoRZedHandNode(zed_camera=zed, hand="both", connect=False).observation_space
    webcam_space = WiLoRHandNode(hand="both", connect=False).observation_space
    validate_metadata(zed_space)
    # `==` ignores metadata: compare it explicitly. Everything but the source
    # (the ZED node's name instead of the unused UVC camera_id) is inherited.
    for key, sub in zed_space.spaces.items():
        assert sub.metadata["source"] == "zed_front"
        expected = {**webcam_space.spaces[key].metadata, "source": "zed_front"}
        assert dict(sub.metadata) == expected
    # Nameless duck-typed cameras fall back to "zed".
    unnamed = WiLoRZedHandNode(zed_camera=FakeZedCamera(), connect=False).observation_space
    assert unnamed.spaces["keypoints_3d"].metadata["source"] == "zed"


def test_close_after_failed_construction_does_not_raise():
    # __del__ -> close() runs on nodes whose __init__ raised before _cap was set
    node = object.__new__(WiLoRZedHandNode)
    node.close()


# ---------------------------------------------------------------------------
# ZED space metadata: depth units and image format
# ---------------------------------------------------------------------------

def _zed_space(*, depth_units="m", rgb_format="rgb", with_depth=True, with_rgb=True,
               source="31726771", serial_number=31726771):
    """An observation space like ``unienv_zed.build_zed_observation_space`` writes."""
    from unienv_interface.backends.numpy import NumpyComputeBackend
    from unienv_interface.space import BoxSpace, DictSpace
    from unienv_interface.space.space_utils.metadata_utils import (
        depth_metadata, image_metadata, metadata,
    )

    camera = {k: v for k, v in (("source", source), ("serial_number", serial_number)) if v is not None}
    spaces = {}
    if with_rgb:
        spaces["left_rgb"] = BoxSpace(
            NumpyComputeBackend, 0, 255, np.uint8, shape=(_H, _W, 3),
            metadata=image_metadata(format=rgb_format, side="left", **camera),
        )
    if with_depth:
        spaces["left_depth"] = BoxSpace(
            NumpyComputeBackend, 0.0, np.inf, np.float32, shape=(_H, _W),
            metadata=depth_metadata(units=depth_units, side="left", **camera),
        )
    spaces["left_intrinsics"] = BoxSpace(
        NumpyComputeBackend, -np.inf, np.inf, np.float32, shape=(3, 3),
        metadata=metadata("intrinsics", frame="camera_cv", units="px", **camera),
    )
    return DictSpace(NumpyComputeBackend, spaces)


class FakeZedCameraWithSpace(FakeZedCamera):
    """A fake ZED that also exposes an ``observation_space`` with metadata."""

    def __init__(self, space, depth_value=_TRUE_DEPTH):
        super().__init__(depth_value=depth_value)
        self.observation_space = space
        self.name = "zed_node"


@pytest.mark.parametrize(
    "units, scale",
    [("m", 1.0), ("cm", 100.0), ("mm", 1000.0), ("in", 1 / 0.0254), ("ft", 1 / 0.3048)],
)
def test_depth_is_converted_to_meters_from_zed_units(units, scale):
    zed = FakeZedCameraWithSpace(_zed_space(depth_units=units), depth_value=_TRUE_DEPTH * scale)
    node = _make_node(zed, FakePipe([_make_detection()]))
    node.post_environment_step(0.04)
    np.testing.assert_allclose(node.get_wrist_pose()[:3, 3], [0.0, 0.0, _TRUE_DEPTH], atol=1e-5)


def test_mm_depth_without_conversion_would_be_rejected():
    # Sanity check of the fixture: 500 mm read as meters is out of range, so
    # without the unit conversion the monocular estimate would be kept.
    node = _make_node(FakeZedCamera(depth_value=_TRUE_DEPTH * 1000.0), FakePipe([_make_detection()]))
    node.post_environment_step(0.04)
    np.testing.assert_allclose(node.get_wrist_pose()[:3, 3], _MONO_CAM_T, atol=1e-6)


def test_unknown_depth_units_are_rejected():
    with pytest.raises(ValueError, match="units"):
        WiLoRZedHandNode(zed_camera=FakeZedCameraWithSpace(_zed_space(depth_units="furlong")), connect=False)


def test_missing_left_rgb_is_rejected():
    with pytest.raises(ValueError, match="left_rgb"):
        WiLoRZedHandNode(zed_camera=FakeZedCameraWithSpace(_zed_space(with_rgb=False)), connect=False)


@pytest.mark.parametrize("rgb_format", ["rgba", "gray"])
def test_unsupported_rgb_format_is_rejected(rgb_format):
    with pytest.raises(ValueError, match="format"):
        WiLoRZedHandNode(zed_camera=FakeZedCameraWithSpace(_zed_space(rgb_format=rgb_format)), connect=False)


def test_bgr_frames_are_passed_through():
    zed = FakeZedCameraWithSpace(_zed_space(rgb_format="bgr"))
    pipe = FakePipe([_make_detection()])
    node = _make_node(zed, pipe)
    node.post_environment_step(0.04)
    # The fake serves channels (1, 2, 3); declared BGR, they reach WiLoR unchanged.
    assert tuple(pipe.frames[0][0, 0]) == (1, 2, 3)


def test_rgb_only_zed_space_skips_depth_correction():
    zed = FakeZedCameraWithSpace(_zed_space(with_depth=False))
    del zed._obs["left_depth"]
    node = _make_node(zed, FakePipe([_make_detection()]))
    node.post_environment_step(0.04)
    np.testing.assert_allclose(node.get_wrist_pose()[:3, 3], _MONO_CAM_T, atol=1e-6)


def test_metadata_source_comes_from_zed_rgb_metadata():
    from unienv_interface.space.space_utils.metadata_utils import validate_metadata

    zed = FakeZedCameraWithSpace(_zed_space(depth_units="mm"))
    space = WiLoRZedHandNode(zed_camera=zed, hand="both", connect=False).observation_space
    validate_metadata(space)
    assert space.metadata["source"] == "31726771"
    assert space.metadata["serial_number"] == 31726771
    for sub in space.spaces.values():
        assert sub.metadata["source"] == "31726771"
        assert sub.metadata["serial_number"] == 31726771

    # A space without a source falls back to the ZED node's name.
    zed = FakeZedCameraWithSpace(_zed_space(source=None, serial_number=None))
    space = WiLoRZedHandNode(zed_camera=zed, connect=False).observation_space
    assert space.spaces["keypoints_3d"].metadata["source"] == "zed_node"
    assert "serial_number" not in space.spaces["keypoints_3d"].metadata


@pytest.mark.parametrize("unit", ["METER", "CENTIMETER", "MILLIMETER", "INCH", "FOOT"])
def test_every_unienv_zed_unit_is_supported(unit):
    unienv_zed = pytest.importorskip("unienv_zed")
    space = unienv_zed.build_zed_observation_space(
        height=_H, width=_W, fps=30, serial_number=123, unit=unit, enable_left_depth=True,
    )
    node = WiLoRZedHandNode(zed_camera=FakeZedCameraWithSpace(space), connect=False)
    assert node._depth_scale_to_m > 0
    assert node.observation_space.spaces["keypoints_3d"].metadata["source"] == "123"
