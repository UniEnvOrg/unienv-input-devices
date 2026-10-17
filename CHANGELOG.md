# Changelog

All notable changes to this project will be documented in this file.

## 0.0.1a1 (2026-10-17)

- Initial public release
- Observation spaces of all nodes now carry unienv space metadata (`kind`,
  `units`, `frame`, `rotation`, `names`, `source`, `fps`, ...):
  `AVPTrackerNode` (poses / keypoints in the AVP world or wrist frame, source
  = AVP IP), `SpaceMouseNode` (normalized deflections in the device frame of
  the chosen `axis_convention`, one entry per discovered button, source = HID
  path), `WiLoRHandNode` (per-key table: `camera_cv` / wrist frames, pixel
  keypoints, `hand_detected` mask, source = camera id) and
  `WiLoRZedHandNode` (source = the ZED node's name)
- Require `unienv>=0.0.1b16` (space metadata)
- Licensed under the Apache License 2.0; `NOTICE` carries the copyright and third-party attributions.
