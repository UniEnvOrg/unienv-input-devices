"""UniEnv input-device adaptors.

Currently exposes:

- :class:`AVPTrackerNode` — Apple Vision Pro hand/head tracking.
- :class:`SpaceMouseNode` — 3Dconnexion SpaceMouse 6-DOF input.
- :class:`WiLoRHandNode` — monocular-RGB hand tracking via WiLoR-mini.
- :class:`WiLoRZedHandNode` — WiLoR-mini on ZED stereo frames, with the wrist
  translation re-anchored to the ZED's metric depth.

All nodes are observation-only ``WorldNode``s. The hand trackers expose
MediaPipe-style (21, 3) finger keypoints under matching key names
(``keypoints_3d_wrist`` / ``keypoints_3d``). Their optional dependencies
(``avp_stream`` / ``pyspacemouse`` / ``torch`` / ``cv2`` / ``scipy`` /
``wilor_mini``) are imported
lazily inside the ``connect=True`` path, so ``import unienv_input_devices``
succeeds with only ``unienv_interface`` + ``numpy`` installed.
"""

from .avp import AVPTrackerNode
from .spacemouse import SpaceMouseNode
from .wilor import WiLoRHandNode
from .wilor_zed import WiLoRZedHandNode

__all__ = [
    "AVPTrackerNode",
    "SpaceMouseNode",
    "WiLoRHandNode",
    "WiLoRZedHandNode",
]
