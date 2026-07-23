# UniEnv Input Devices

Apple Vision Pro (AVP) hand/head tracking adaptors for [UniEnv](https://github.com/UniEnvOrg/UniEnv),
exposed as observation-only `WorldNode`s.

Based on the [`avp_stream`](https://pypi.org/project/avp_stream/) VisionPro
streaming client.

## Installation

```bash
pip install unienv-input-devices
```

## Hardware setup

1. Put your Apple Vision Pro on the **same LAN** as the host machine (the host
   can be Linux, macOS, etc.).
2. On the Vision Pro, run the **Tracking Streamer** app. The streamer exposes a
   hand-tracking gRPC endpoint on a fixed port (`<avp-ip>:12345`).
3. Note the AVP's LAN IP address (e.g. `192.168.50.127`).
4. Install the `avp_stream` dependency on the host (it is listed as a
   dependency of this package, so `pip install unienv-input-devices` should pull
   it in).

## Usage

```python
import numpy as np
import time

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.world import RealWorld, WorldEnv
from unienv_input_devices import AVPTrackerNode

# RealWorld backed by numpy; world_timestep matches the node control timestep.
world = RealWorld(
    NumpyComputeBackend,
    world_timestep=0.04,  # 25Hz — usually set equal to the control timestep
    batch_size=None,       # None means single instance
)

# Pass the AVP's LAN IP. The node lazily imports avp_stream only on this path.
node = AVPTrackerNode(
    world,
    ip="192.168.50.127",  # your AVP's LAN IP
    connect=True,
    control_timestep=0.04,
    update_timestep=0.04,
)

env = WorldEnv(world, node)

# The first reset routes through the reload flow and seeds the initial
# observation (zeros until the first AVP frame arrives).
context, obs, info = env.reset()
print(obs.keys())  # dict_keys(['head', 'left_fingers', ... ])

while True:
    # Observation-only node: pass None (action_space is None, step ignores it).
    obs, reward, terminated, truncated, info = env.step(None)
    print(obs["left_wrist"].shape)   # (4, 4)
    print(obs["left_fingers"].shape) # (25, 4, 4)
    print(node.get_pinch_distance("left"))
    print(node.is_streaming())
```

### Running without hardware

`AVPTrackerNode(connect=False)` constructs and runs the full lifecycle without
the `avp_stream` SDK or any hardware — reads return zero-valued observations of
the correct shapes/dtypes. This is handy for development and unit tests.

## Raw-data philosophy

This node exposes **unfiltered** AVP world-frame data exactly as the
`VisionProStreamer` provides it:

- `head`, `left_wrist`, `right_wrist` — homogeneous (4, 4) transforms.
- `left_fingers`, `right_fingers` — (25, 4, 4) per-finger homogeneous transforms.
- `left_pinch_distance`, `right_pinch_distance` — pinch distances in meters.
- `left_wrist_roll`, `right_wrist_roll` — wrist roll angles in radians.

The AVP world frame is X-right, Y-forward, Z-up (after the `avp_stream`
YUP2ZUP conversion). Positions are the `[:3, 3]` translation columns of each
4x4 transform.

Calibration, filtering, coordinate retargeting, and any downstream
interpretation belong in **separate downstream nodes** — this adaptor stays a
thin, faithful sensor.

## Retargeting helper

For retargeting pipelines that expect MediaPipe-style (21, 3) hand keypoints,
the package exports :func:`convert_vp_to_mediapipe`, which maps the 25 VisionPro
finger transforms down to the 21 MediaPipe landmarks by selecting a subset of
indices and taking their translation columns:

```python
from unienv_input_devices import convert_vp_to_mediapipe

# fingers_mat: (25, 4, 4) from obs["left_fingers"]
mediapipe_keypoints = convert_vp_to_mediapipe(fingers_mat)  # (21, 3) float32
```