"""Hardware-free tests for SpaceMouseNode."""

import builtins
from collections import namedtuple
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
import pytest

from unienv_input_devices import SpaceMouseNode


SpaceMouseState = namedtuple(
    "SpaceMouseState",
    "x y z roll pitch yaw buttons",
)


class FakeDevice:
    """Fake pyspacemouse device.

    Mirrors real pyspacemouse behavior: ``read()`` returns SpaceMouseState
    with axes ALREADY normalized to [-1, 1] (raw HID counts divided by the
    device spec's axis_scale happens inside pyspacemouse).
    """

    def __init__(self, button_names=("LEFT", "RIGHT"), reads=()):
        self.info = SimpleNamespace(
            button_names=tuple(button_names),
            button_specs=tuple(SimpleNamespace(name=name) for name in button_names),
        )
        self.reads = list(reads)
        self.close_calls = 0

    def read(self):
        return self.reads.pop(0) if self.reads else None

    def close(self):
        self.close_calls += 1


def _state(translation=(0, 0, 0), rotation=(0, 0, 0), buttons=(0, 0)):
    return SpaceMouseState(*translation, *rotation, list(buttons))


def _install_fake(monkeypatch, devices, listed=None):
    module = ModuleType("pyspacemouse")
    module.AxisConvention = SimpleNamespace(
        HID="hid-enum",
        HID_Z_UP="hid-z-up-enum",
        ROS="ros-enum",
        UNITY="unity-enum",
        LEGACY="legacy-enum",
    )
    module.calls = []
    module.list_calls = 0

    def open_by_path(path, *, nonblocking, axis_convention):
        module.calls.append((path, nonblocking, axis_convention))
        return devices[path]

    def list_devices():
        module.list_calls += 1
        return {} if listed is None else listed

    module.open_by_path = open_by_path
    module.get_connected_devices_by_path = list_devices
    monkeypatch.setitem(sys.modules, "pyspacemouse", module)
    return module


def test_button_name_resolution_and_validation(monkeypatch):
    derived_device = FakeDevice(("LEFT", "RIGHT", "MENU"))
    explicit_device = FakeDevice(("LEFT", "RIGHT", "MENU"))
    missing_device = FakeDevice(("LEFT", "RIGHT"))
    mismatch_device = FakeDevice(("MENU", "FIT"))
    _install_fake(
        monkeypatch,
        {
            "/derived": derived_device,
            "/explicit": explicit_device,
            "/missing": missing_device,
            "/mismatch": mismatch_device,
        },
    )

    derived = SpaceMouseNode(path="/derived")
    assert derived.button_names == ("left", "right", "menu")

    explicit = SpaceMouseNode(path="/explicit", button_names=("RIGHT", "menu"))
    assert explicit.button_names == ("right", "menu")
    assert set(explicit.observation_space.spaces) == {
        "translation", "rotation", "right_button", "menu_button"
    }

    offline = SpaceMouseNode(path="/offline", connect=False)
    assert offline.button_names == ("left", "right")

    with pytest.raises(ValueError, match="not present"):
        SpaceMouseNode(path="/missing", button_names=("MENU",))
    assert missing_device.close_calls == 1

    deferred = SpaceMouseNode(path="/mismatch", connect=False)
    with pytest.raises(ValueError, match="observation space"):
        deferred._connect_hardware()
    assert mismatch_device.close_calls == 1


def test_per_button_observations_and_getter(monkeypatch):
    device = FakeDevice(
        ("LEFT", "RIGHT", "MENU"),
        reads=[_state(buttons=(0, 1, 1))],
    )
    _install_fake(monkeypatch, {"/buttons": device})
    node = SpaceMouseNode(path="/buttons")
    node.post_environment_step(0.04)

    observation = node.get_observation()
    for name in ("left", "right", "menu"):
        value = observation[f"{name}_button"]
        assert value.shape == (1,)
        assert value.dtype == np.float32
    assert not node.get_button("LEFT")
    assert node.get_button("right")
    assert node.get_button("Menu")
    with pytest.raises(ValueError, match="unknown button"):
        node.get_button("fit")


def test_axes_passthrough_clipping_and_getters(monkeypatch):
    # pyspacemouse returns axes already normalized to [-1, 1]; the node must
    # pass them through unchanged and only clip out-of-range values.
    device = FakeDevice(reads=[_state((0.5, -2.0, 1.0), (2.0, -0.5, 0.0))])
    _install_fake(monkeypatch, {"/axes": device})
    node = SpaceMouseNode(path="/axes")
    node.post_environment_step(0.04)

    np.testing.assert_array_equal(node.get_translation(), [0.5, -1.0, 1.0])
    np.testing.assert_array_equal(node.get_rotation(), [1.0, -0.5, 0.0])
    np.testing.assert_array_equal(
        node.get_twist(),
        [0.5, -1.0, 1.0, 1.0, -0.5, 0.0],
    )
    assert node.get_translation().dtype == np.float32
    assert node.get_rotation().dtype == np.float32
    assert node.get_twist().dtype == np.float32


def test_deadband_zeroes_small_normalized_deflections(monkeypatch):
    device = FakeDevice(reads=[_state((0.14, 0.2, -0.14), (0.197, -0.203, 0.0))])
    _install_fake(monkeypatch, {"/deadband": device})
    node = SpaceMouseNode(path="/deadband", deadband=0.2)
    node.post_environment_step(0.04)

    np.testing.assert_allclose(node.get_translation(), [0.0, 0.2, 0.0])
    np.testing.assert_allclose(node.get_rotation(), [0.0, -0.203, 0.0])


def test_nonblocking_none_holds_last_observation(monkeypatch):
    device = FakeDevice(reads=[_state((0.5, 0, 0), buttons=(1, 0)), None])
    _install_fake(monkeypatch, {"/hold": device})
    node = SpaceMouseNode(path="/hold")
    node.post_environment_step(0.04)
    held = {key: value.copy() for key, value in node.get_observation().items()}

    node.post_environment_step(0.04)
    for key, expected in held.items():
        np.testing.assert_array_equal(node.get_observation()[key], expected)


def test_multiple_devices_are_independent(monkeypatch):
    first = FakeDevice(reads=[_state((0.5, 0, 0), buttons=(1, 0))])
    second = FakeDevice(reads=[_state((0, -0.5, 0), buttons=(0, 1))])
    module = _install_fake(monkeypatch, {"/first": first, "/second": second})
    first_node = SpaceMouseNode(path="/first")
    second_node = SpaceMouseNode(path="/second")

    first_node.post_environment_step(0.04)
    second_node.post_environment_step(0.04)
    np.testing.assert_array_equal(first_node.get_translation(), [0.5, 0.0, 0.0])
    np.testing.assert_array_equal(second_node.get_translation(), [0.0, -0.5, 0.0])
    assert first_node.get_button("left")
    assert second_node.get_button("right")

    first_node.close()
    assert first.close_calls == 1
    assert second.close_calls == 0
    assert not hasattr(module, "close")
    second_node.close()
    assert second.close_calls == 1


def test_connect_false_is_zero_and_does_not_import_pyspacemouse(monkeypatch):
    monkeypatch.delitem(sys.modules, "pyspacemouse", raising=False)
    imports = []
    original_import = builtins.__import__

    def recording_import(name, *args, **kwargs):
        if name == "pyspacemouse":
            imports.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", recording_import)
    node = SpaceMouseNode(path="/offline", connect=False)
    assert imports == []
    assert node.action_space is None
    assert set(node.observation_space.spaces) == {
        "translation", "rotation", "left_button", "right_button"
    }
    for value in node.get_observation().values():
        assert value.dtype == np.float32
        assert not np.any(value)
    node.after_reset()
    node.set_next_action({"ignored": True})
    node.close()
    assert imports == []


def test_axis_conventions_validation_and_discovery(monkeypatch):
    devices = {
        f"/{name}": FakeDevice()
        for name in ("ros", "hid_z_up", "hid", "unity")
    }
    listed = {"/ros": "SpaceMouse Compact"}
    module = _install_fake(monkeypatch, devices, listed=listed)

    expected = {
        "ros": "ros-enum",
        "hid_z_up": "hid-z-up-enum",
        "hid": "hid-enum",
        "unity": "unity-enum",
    }
    for name, enum_value in expected.items():
        SpaceMouseNode(path=f"/{name}", axis_convention=name)
        assert module.calls[-1] == (f"/{name}", True, enum_value)

    with pytest.raises(ValueError, match="axis_convention"):
        SpaceMouseNode(path="/invalid", axis_convention="legacy", connect=False)
    for value in (-0.01, 1.0, 1.1):
        with pytest.raises(ValueError, match="deadband"):
            SpaceMouseNode(path="/invalid", deadband=value, connect=False)

    assert SpaceMouseNode.list_devices() == listed
    assert module.list_calls == 1
