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
pip install "unienv-input-devices[wilor]" --no-build-isolation  # see docs/wilor.md
```

## Development

```bash
pip install -e ".[test]"
pytest tests/
```
