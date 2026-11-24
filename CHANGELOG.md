# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- `WiLoRZedHandNode` reads the ZED camera's space metadata at construction:
  `left_rgb` is required (`format` `"rgb"` or `"bgr"`, BGR frames are no
  longer flipped), and `left_depth` is converted to meters from its `units`
  (`"m"`, `"cm"`, `"mm"`, `"in"`, `"ft"`), so a `ZedCameraNode` built with a
  non-metric `unit` no longer corrupts the depth correction; other formats /
  units raise a `ValueError`. Duck-typed cameras without an
  `observation_space` keep the RGB / meters contract.
- `WiLoRZedHandNode` metadata: `source` is now the ZED's `left_rgb` `source`
  (its serial number) and `serial_number` is recorded, falling back to the
  ZED node's name when the camera space carries none.

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
- The `wilor` extra no longer pins `chumpy` / `wilor_mini` from git (PyPI rejects direct dependencies); install them separately, see `docs/wilor.md`.
