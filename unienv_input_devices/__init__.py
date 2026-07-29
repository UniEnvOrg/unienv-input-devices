"""UniEnv input-device adaptors.

Currently exposes:

- :class:`AVPTrackerNode` — Apple Vision Pro hand/head tracking.
- :class:`WiLoRHandNode` — monocular-RGB hand tracking via WiLoR-mini.

Both nodes are observation-only ``WorldNode``s and expose MediaPipe-style
(21, 3) finger keypoints under matching key names (``keypoints_3d_wrist`` /
``keypoints_3d``). Their heavy dependencies
(``avp_stream`` / ``torch`` / ``cv2`` / ``scipy`` / ``wilor_mini``) are imported
lazily inside the ``connect=True`` path, so ``import unienv_input_devices``
succeeds with only ``unienv_interface`` + ``numpy`` installed.
"""

from .avp import AVPTrackerNode
from .wilor import WiLoRHandNode

__all__ = [
    "AVPTrackerNode",
    "WiLoRHandNode",
]