# UniEnv Input Devices

Input-device adaptors for [UniEnv](https://github.com/UniEnvOrg/UniEnv),
exposed as observation-only `WorldNode`s.

## Supported devices

| Device | Node | Extra | Docs |
|--------|------|-------|------|
| Apple Vision Pro (hand/head tracking) | `AVPTrackerNode` | `[avp]` | [docs/avp.md](docs/avp.md) |
| 3Dconnexion SpaceMouse (6-DOF input) | `SpaceMouseNode` | `[spacemouse]` | [docs/spacemouse.md](docs/spacemouse.md) |
| Monocular RGB webcam (WiLoR hand tracking) | `WiLoRHandNode` | `[wilor]` | [docs/wilor.md](docs/wilor.md) |

Each device has its own optional extra and its own README covering hardware
setup, installation, calibration, and usage — follow the links above.

## Space metadata

Every observation entry is described with unienv space metadata
(`space.metadata`): what it is (`kind`: `"pose"`, `"keypoints"`, `"teleop"`,
`"mask"`), its `units`, reference `frame` (plus an `axes` note where the frame
is device specific), `rotation` representation of 4x4 transforms
(`"matrix"`), `hand`, `source` (AVP IP, SpaceMouse HID path, camera) and
`fps` (`1 / control_timestep`). For example,
`find_spaces_by_metadata(node.observation_space, kind="keypoints")` finds the
keypoint entries of any hand tracker.

## Installation

The base package installs with no device SDKs (all nodes import their optional
dependencies lazily, so the package imports fine without them):

```bash
pip install unienv-input-devices
```

Add the extra for the device(s) you want, e.g.:

```bash
pip install "unienv-input-devices[avp]"
pip install "unienv-input-devices[spacemouse]"
pip install "unienv-input-devices[wilor]"  # + chumpy and wilor_mini from git, see docs/wilor.md
```

## Development

```bash
pip install -e ".[test]"
pytest tests/
```
