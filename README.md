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

---

# WiLoRHandNode — monocular-RGB hand tracking

`WiLoRHandNode` wraps the [WiLoR-mini](https://github.com/warmshao/WiLoR-mini)
inference pipeline (YOLO hand detection + MANO 3D pose regression) as an
observation-only `WorldNode`. It reads BGR frames from a single UVC webcam via
OpenCV and emits per-frame 21-keypoint hand poses in three coordinate frames
(MANO-local, camera-frame, pixel) plus a camera-frame wrist transform.

The integration is adapted from the MIT-licensed
[AnyTeleop community fork](https://github.com/RalphFH/AnyTeleop). The fork
applies hardcoded translation offsets and a fixed `OPERATOR2MANO` rotation
that are specific to that teleoperation rig; this node **does not** port them —
it emits raw camera-frame data and leaves any rig-specific transform to
downstream nodes.

## Hardware

- Any UVC webcam (laptop built-in camera works for development).
- A **CUDA GPU is strongly recommended** — WiLoR-mini inference on CPU is
  functional but slow. Upstream recommends Python 3.10 + a CUDA PyTorch build.

## Installation

```bash
# 1. Torch / OpenCV / scipy (the WiLoR runtime deps this package lists).
pip install unienv-input-devices[wilor]

# 2. WiLoR-mini itself — NOT on PyPI, install from GitHub.
pip install "git+https://github.com/warmshao/WiLoR-mini"
```

WiLoR model weights and the MANO hand-model assets auto-download from
HuggingFace on first use of the pipeline. **Licensing:** WiLoR model weights
are released under **CC-BY-NC-ND** and the MANO assets are **non-commercial**.
This package vendors none of those assets — by installing `wilor_mini` you
accept the upstream licensing terms.

## Focal-length calibration

WiLoR estimates camera translation from a weak-perspective model scaled by a
focal-length parameter. The pipeline's `focal_length` kwarg should be set to
approximately `camera_fx * 256 / max(image_width, image_height)` (the upstream
default is `5000.0`, which is reasonable for a typical webcam). Pass a
calibrated value to `WiLoRHandNode(..., focal_length=...)` for more accurate
absolute depth.

## Usage

```python
import numpy as np
import time

from unienv_interface.backends.numpy import NumpyComputeBackend
from unienv_interface.world import RealWorld, WorldEnv
from unienv_input_devices import WiLoRHandNode

# RealWorld backed by numpy; world_timestep matches the node control timestep.
world = RealWorld(
    NumpyComputeBackend,
    world_timestep=0.04,  # 25Hz — usually set equal to the control timestep
    batch_size=None,      # None means single instance
)

# The node lazily imports torch/cv2/wilor_mini only on this connect=True path.
node = WiLoRHandNode(
    world,
    camera_id=0,         # default webcam
    hand="right",        # track the right hand
    focal_length=5000.0, # upstream default; calibrate for your camera
    connect=True,
    control_timestep=0.04,
    update_timestep=0.04,
)

env = WorldEnv(world, node)

# The first reset routes through the reload flow and seeds the initial
# observation (zeros until the first WiLoR frame arrives).
context, obs, info = env.reset()
print(obs.keys())  # dict_keys(['keypoints_3d_local', 'keypoints_3d', ...])

while True:
    # Observation-only node: pass None (action_space is None, step ignores it).
    obs, reward, terminated, truncated, info = env.step(None)
    print(obs["keypoints_3d"].shape)        # (21, 3)
    print(obs["keypoints_3d_local"].shape)  # (21, 3)
    print(obs["wrist_pose"].shape)          # (4, 4)
    print(node.is_hand_detected())          # True/False
    print(node.get_keypoints(local=True))   # (21, 3) MANO-local meters
```

### Running without hardware

`WiLoRHandNode(connect=False)` constructs and runs the full lifecycle without
`torch`, `cv2`, `scipy` or `wilor_mini` installed — reads return zero-valued
observations of the correct shapes/dtypes with `hand_detected == 0.0`. This is
handy for development and unit tests.

## Observation space

All entries are `np.float32`.

| Key                  | Shape   | Frame / units                                                                 |
|----------------------|---------|-------------------------------------------------------------------------------|
| `keypoints_3d_local` | (21, 3) | MANO-local, wrist-rooted, meters (articulation-only)                          |
| `keypoints_3d`       | (21, 3) | Camera frame (x right, y down, z forward), meters = local + `pred_cam_t_full` |
| `keypoints_2d`       | (21, 2) | Pixel coordinates, top-left origin                                            |
| `wrist_pose`         | (4, 4)  | Camera-frame homogeneous transform; rotation from `global_orient` axis-angle  |
| `hand_detected`      | (1,)    | 1.0 if a hand of the configured handedness was found in the latest read      |

21-keypoint order: wrist (0), thumb (1-4), index (5-8), middle (9-12),
ring (13-16), little (17-20) — identical to the MediaPipe convention.

## Limitations

- **Monocular depth is model-estimated.** Wrist translation (`pred_cam_t_full`)
  typically carries ~5-15 cm of error; finger articulation
  (`keypoints_3d_local`) is considerably more reliable than absolute position.
  Prefer `keypoints_3d_local` for retargeting joint angles.
- **Occlusion degrades silently.** When the hand is partially out of frame the
  pipeline may still return a detection with degraded keypoints. Gate
  downstream behavior on `hand_detected` and on reprojection-error checks
  (e.g. compare `keypoints_2d` against a re-projection of `keypoints_3d`).
- **Hold-last behavior.** On a failed frame read or when no matching hand is
  found, the previous keypoints/pose are retained and `hand_detected` is set
  to 0.0 — downstream consumers must check `hand_detected` before trusting the
  geometric entries.