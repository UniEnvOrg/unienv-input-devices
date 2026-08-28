# SpaceMouseNode - 3Dconnexion 6-DOF input

`SpaceMouseNode` wraps a path-selected 3Dconnexion SpaceMouse as an
observation-only `WorldNode`. It reports the cap's signed translation and
rotation deflection plus one named observation per hardware button. It does
not integrate deflection into a virtual pose.

## Installation

Install the package extra:

```bash
pip install "unienv-input-devices[spacemouse]"
```

On Debian or Ubuntu, install the hidapi runtime library:

```bash
sudo apt-get install libhidapi-dev
```

Linux users also need permission to access the SpaceMouse HID device. Add a
udev rule for current 3Dconnexion vendor IDs and older Logitech-branded models:

```udev
# /etc/udev/rules.d/99-spacemouse.rules
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", MODE="0660", GROUP="plugdev", TAG+="uaccess"
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", ATTRS{idVendor}=="046d", MODE="0660", GROUP="plugdev", TAG+="uaccess"
```

Then reload the rules and reconnect the device:

```bash
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Add your user to `plugdev` if your distribution does not grant access through
`TAG+="uaccess"`:

```bash
sudo usermod -aG plugdev "$USER"
```

Log out and back in after changing group membership.

## Discovery

Device paths are the stable selector accepted by the node. Discover supported
devices before constructing it:

```python
from unienv_input_devices import SpaceMouseNode

devices = SpaceMouseNode.list_devices()
for path, name in devices.items():
    print(path, name)
```

`list_devices()` returns `{hid_path: device_name}`, e.g.
`{"/dev/hidraw3": "SpaceMouse Compact"}`. Use the path as the node's `path`
argument. The value is an opaque HID path on macOS and Windows, and should
still be passed unchanged.

## Usage

```python
from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.world import RealWorld, WorldEnv
from unienv_input_devices import SpaceMouseNode

world = RealWorld(
    NumpyComputeBackend,
    world_timestep=0.04,
    batch_size=None,
)
node = SpaceMouseNode(
    world,
    path="/dev/hidraw3",
    axis_convention="ros",
    deadband=0.05,
)
env = WorldEnv(world, node)

context, observation, info = env.reset()
while True:
    observation, reward, terminated, truncated, info = env.step(None)
    twist = node.get_twist()              # (6,) float32
    left_pressed = node.get_button("left")
```

`translation` contains `(x, y, z)` and `rotation` contains `(roll, pitch,
yaw)`. PySpaceMouse normalizes both to `[-1, 1]` internally (raw HID counts
divided by the device's axis scale, 350 for current models); the node only
applies the configured `deadband` and clips to `[-1, 1]`. Button observations
are float32 arrays with shape `(1,)`, such as `left_button` and
`right_button`.

## Axis conventions

| Value | Translation axes | Handedness |
|-------|------------------|------------|
| `"ros"` (default) | +x forward, +y left, +z up | Right-handed |
| `"hid_z_up"` | +x right, +y away from the user, +z up | Right-handed |
| `"hid"` | +x right, +y toward the user, +z down | Right-handed |
| `"unity"` | +x right, +y up, +z forward | Left-handed |

The string is mapped directly to the matching `pyspacemouse.AxisConvention`
member when the path is opened.

## Button names

With `button_names=None`, names are derived from the opened model's device
specification and normalized to lowercase. The stock model definitions expose:

| Models | Button names |
|--------|--------------|
| SpaceMouse Compact, SpaceMouse Wireless, SpaceNavigator | `left`, `right` |
| SpaceMouse Pro / Pro Wireless | `menu`, `fit`, `esc`, `alt`, `shift`, `ctrl`, `btn_1`, `btn_2`, `btn_3`, `btn_4`, `roll_cw`, `top`, `front`, `rear`, `rotation` |
| SpaceMouse Enterprise | `btn_1`, `btn_2`, `btn_3`, `btn_4`, `btn_5`, `btn_6`, `btn_7`, `btn_8`, `btn_9`, `btn_10`, `menu`, `fit`, `top`, `right`, `front`, `roll_cw`, `iso1`, `esc`, `alt`, `shift`, `ctrl`, `lock` |

Pass `button_names=(...)` to expose only a selected set. Every explicit name
must exist in the connected device specification. With `connect=False` and no
explicit names, the node uses the Compact default `("left", "right")`.

## Multiple devices

Each node owns and closes only the device object returned for its path. Two
devices can therefore be polled independently:

```python
left_controller = SpaceMouseNode(
    world,
    name="left_spacemouse",
    path="/dev/hidraw3",
)
right_controller = SpaceMouseNode(
    world,
    name="right_spacemouse",
    path="/dev/hidraw5",
)

left_twist = left_controller.get_twist()
right_twist = right_controller.get_twist()
```

Do not use PySpaceMouse's module-level `read()` or `close()` functions for
these handles; the node always uses each returned device object's methods.

## Deadband and hold-last behavior

`deadband` is applied independently to each normalized axis. Values satisfying
`abs(value) < deadband` become zero; larger values are retained and the result
is clipped to `[-1, 1]`. Valid deadbands are in `[0, 1)`.

The device is opened non-blocking. When a poll returns `None` because no new
HID report is available, the complete previous observation is retained.
