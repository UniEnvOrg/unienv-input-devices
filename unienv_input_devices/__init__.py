"""UniEnv input-device adaptors.

Currently exposes:

- :class:`AVPTrackerNode` — Apple Vision Pro hand/head tracking, plus the
  :func:`convert_vp_to_mediapipe` helper for retargeting pipelines.
- :class:`WiLoRHandNode` — monocular-RGB hand tracking via WiLoR-mini.

Both nodes are observation-only ``WorldNode``s. Their heavy dependencies
(``avp_stream`` / ``torch`` / ``cv2`` / ``scipy`` / ``wilor_mini``) are imported
lazily inside the ``connect=True`` path, so ``import unienv_input_devices``
succeeds with only ``unienv_interface`` + ``numpy`` installed.
"""

from .avp import AVPTrackerNode, convert_vp_to_mediapipe
from .wilor import WiLoRHandNode

__all__ = [
    "AVPTrackerNode",
    "convert_vp_to_mediapipe",
    "WiLoRHandNode",
]