# AVPTrackerNode — Apple Vision Pro hand/head tracking

`AVPTrackerNode` is an observation-only `WorldNode` streaming Apple Vision Pro
hand/head tracking into UniEnv, based on the
[`avp_stream`](https://pypi.org/project/avp_stream/) VisionPro streaming client.

## Installation

```bash
pip install "unienv-input-devices[avp]"
```

## Hardware setup

1. Put your Apple Vision Pro on the **same LAN** as the host machine (the host
   can be Linux, macOS, etc.).
2. On the Vision Pro, run the **Tracking Streamer** app. The streamer exposes a
   hand-tracking gRPC endpoint on a fixed port (`<avp-ip>:12345`).
3. Note the AVP's LAN IP address (e.g. `192.168.50.127`).

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
    print(obs["left_wrist"].shape)             # (4, 4)
    print(obs["left_fingers"].shape)           # (25, 4, 4)
    print(obs["left_keypoints_3d_wrist"].shape) # (21, 3)
    print(node.get_pinch_distance("left"))
    print(node.is_streaming())
```

### Running without hardware

`AVPTrackerNode(connect=False)` constructs and runs the full lifecycle without
the `avp_stream` SDK or any hardware — reads return zero-valued observations of
the correct shapes/dtypes. This is handy for development and unit tests.

## Raw-data philosophy

This node exposes **unfiltered** AVP data exactly as the `VisionProStreamer`
provides it:

- `head`, `left_wrist`, `right_wrist` — homogeneous (4, 4) transforms in the
  AVP world frame.
- `left_fingers`, `right_fingers` — (25, 4, 4) per-finger homogeneous
  transforms in the **wrist-local frame** (joint 0 is the wrist itself, at the
  origin). Compose `wrist @ finger` to obtain world-frame finger poses.
- `left_keypoints_3d_wrist`, `right_keypoints_3d_wrist` — (21, 3) wrist-local
  keypoints in meters, MediaPipe order (wrist 0, thumb 1-4, index 5-8, middle
  9-12, ring 13-16, little 17-20): the translation columns of the mapped
  finger transforms. Same frame/ordering contract as the WiLoR node's
  `keypoints_3d_wrist` — the right input for hand IK.
- `left_keypoints_3d`, `right_keypoints_3d` — (21, 3) keypoints in the AVP
  world frame (`wrist @ keypoints_3d_wrist`), the analog of the WiLoR node's
  camera-frame `keypoints_3d`.
- `left_pinch_distance`, `right_pinch_distance` — pinch distances in meters.
- `left_wrist_roll`, `right_wrist_roll` — wrist roll angles in radians.

The AVP world frame is X-right, Y-forward, Z-up (after the `avp_stream`
YUP2ZUP conversion). Positions are the `[:3, 3]` translation columns of each
4x4 transform.

Calibration, filtering, coordinate retargeting, and any downstream
interpretation belong in **separate downstream nodes** — this adaptor stays a
thin, faithful sensor.

## Keypoint getters

The keypoint observations are also available through convenience getters that
mirror the WiLoR node's API:

```python
kp_wrist = node.get_keypoints_wrist("left")  # (21, 3) float32, wrist-local
kp_world = node.get_keypoints("left")        # (21, 3) float32, AVP world frame
```
