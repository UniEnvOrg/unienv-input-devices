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
  functional but slow.

## Installation

```bash
pip install "unienv-input-devices[wilor]" --no-build-isolation
```

The `wilor` extra pulls:

- the WiLoR runtime deps (`torch` / `opencv-python` / `scipy`),
- `wilor_mini` from a [compat fork](https://github.com/realquantumcookie/WiLoR-mini)
  of `warmshao/WiLoR-mini` (upstream pins `torch<=2.5`; the fork relaxes the
  pins, bumps `ultralytics>=8.4`, and passes explicit `weights_only=False`, so
  it runs on torch>=2.6 and Python>=3.11),
- `chumpy` from git master (0.71). PyPI's 0.70 is broken on Python>=3.11
  (`inspect.getargspec` was removed), and its legacy `setup.py` does
  `import pip`, which fails under pip's default build isolation — hence the
  `--no-build-isolation` flag, which makes `pip` importable during the build.
  `setuptools` and `wheel` must be present in your environment (they are in
  any standard conda/venv setup).

Tested with Python 3.12 + torch 2.13 (CUDA 13) + ultralytics 8.4.

WiLoR model weights and the MANO hand-model assets auto-download from
HuggingFace on first use of the pipeline. **Licensing:** WiLoR model weights
are released under **CC-BY-NC-ND** and the MANO assets are **non-commercial**.
This package vendors none of those assets — by installing `wilor_mini` you
accept the upstream licensing terms.

## Webcams behind usbipd (WSL2)

WSL2 does not expose webcams natively; attach one from Windows with
[usbipd-win](https://github.com/dorssel/usbipd-win) (`usbipd bind` +
`usbipd attach --wsl`). Over the virtual USB bus, the default uncompressed
YUYV stream exceeds the available bandwidth and frames arrive zeroed (solid
green). Request a compressed pixel format and a modest resolution via the
node's capture params:

```python
WiLoRHandNode(..., width=640, height=480, fourcc="MJPG")
```

## Two-hand tracking

Construct the node with `hand="both"` to track both hands in a single pass.
The WiLoR-mini detector emits left/right hand classes and batches all detected
hands through one model forward pass, so the second hand is nearly free.

In this mode the observation keys are prefixed per hand —
`left_keypoints_3d_local`, `left_keypoints_3d`, `left_keypoints_2d`,
`left_wrist_pose`, `left_hand_detected`, and the same five with `right_` —
and each hand has its own independent hold-last state. The helper methods take
a required `hand` argument:

```python
node = WiLoRHandNode(world, camera_id=0, hand="both", focal_length=456.0,
                     width=640, height=480, fourcc="MJPG",
                     connect=True, control_timestep=0.04, update_timestep=0.04)
...
obs, *_ = env.step(None)
if node.is_hand_detected("left"):
    left_kp = node.get_keypoints(local=True, hand="left")   # (21, 3)
if node.is_hand_detected("right"):
    right_wrist = node.get_wrist_pose(hand="right")         # (4, 4)
```

Identity comes from the detector's left/right classification, not temporal
association — during hand-over-hand occlusion the detector may drop or
mislabel a hand, so always gate on the per-hand `*_hand_detected` flag.

## Focal-length calibration

WiLoR estimates camera translation from a weak-perspective model scaled by a
focal-length parameter. The pipeline's `focal_length` kwarg should be set to
approximately `camera_fx * 256 / max(image_width, image_height)` (the upstream
default is `5000.0`, which is reasonable for a typical webcam). Pass a
calibrated value to `WiLoRHandNode(..., focal_length=...)` for more accurate
absolute depth.

A quick practical calibration: hold your open hand flat at a measured distance
`d` from the camera, record the reported wrist depth `z` with the default
focal length, then set `focal_length = 5000.0 * d / z`. With a correct value,
the wrist→middle-fingertip distance should land in the anatomical range
(~0.17–0.19 m).

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
| `keypoints_3d_local` | (21, 3) | Wrist-rooted, **camera-aligned** axes, meters (articulation-only)             |
| `keypoints_3d_wrist` | (21, 3) | True wrist frame (origin **and** axes on the wrist, MANO root frame), meters |
| `keypoints_3d`       | (21, 3) | Camera frame (x right, y down, z forward), meters = local + `pred_cam_t_full` |
| `keypoints_2d`       | (21, 2) | Pixel coordinates, top-left origin                                            |
| `wrist_pose`         | (4, 4)  | Camera-frame homogeneous transform; rotation from `global_orient` axis-angle  |
| `hand_detected`      | (1,)    | 1.0 if a hand of the configured handedness was found in the latest read      |

21-keypoint order: wrist (0), thumb (1-4), index (5-8), middle (9-12),
ring (13-16), little (17-20) — identical to the MediaPipe convention.

**Which frame for robot control?** For an arm+hand setup, drive the arm from
`wrist_pose` (full camera-frame EEF transform) and do hand IK on
`keypoints_3d_wrist` — it is invariant to wrist translation *and* rotation
(`R(global_orient).T @ keypoints_3d_local`), mirroring the AVP node's
wrist-local `*_keypoints_3d_wrist` convention. Note the wrist frame follows the MANO
root-frame convention; mapping it to your robot's EEF frame convention needs
one constant offset rotation. `keypoints_3d_local` sits between the two:
wrist-rooted but camera-aligned — invariant to translation only.

## Limitations

- **Monocular depth is model-estimated.** Wrist translation (`pred_cam_t_full`)
  typically carries ~5-15 cm of error; finger articulation
  (`keypoints_3d_local` / `keypoints_3d_wrist`) is considerably more reliable
  than absolute position. Prefer the local/wrist-frame keypoints for
  retargeting joint angles.
- **Occlusion degrades silently.** When the hand is partially out of frame the
  pipeline may still return a detection with degraded keypoints. Gate
  downstream behavior on `hand_detected` and on reprojection-error checks
  (e.g. compare `keypoints_2d` against a re-projection of `keypoints_3d`).
- **Hold-last behavior.** On a failed frame read or when no matching hand is
  found, the previous keypoints/pose are retained and `hand_detected` is set
  to 0.0 — downstream consumers must check `hand_detected` before trusting the
  geometric entries.
