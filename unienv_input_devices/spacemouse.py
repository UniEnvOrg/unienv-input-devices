"""3Dconnexion SpaceMouse input as an observation-only UniEnv WorldNode.

The node exposes the cap's signed six-axis deflection, not an integrated pose.
PySpaceMouse's ``read()`` already normalizes axes to ``[-1, 1]`` (raw HID
counts divided by the device's ``axis_scale``), so the node only applies the
configured deadband and clips to ``[-1, 1]``. The selected PySpaceMouse axis
convention determines the meaning and signs of those axes; ``"ros"`` is the
default (x forward, y left, z up, right-handed).

Axis directions (device body frame — they rotate with the device on the desk)
------------------------------------------------------------------------------
With the default ``axis_convention="ros"`` (ROS REP-103 body frame,
right-handed), assuming the device sits with its buttons/front edge facing
the operator:

- ``translation`` (push/pull the cap): ``+x`` = push the cap **forward** (away
  from the operator), ``+y`` = push **left**, ``+z`` = **lift up**.
- ``rotation`` (right-hand rule about the translation axes): ``+roll`` = tilt
  the cap's **right edge down** (about +x), ``+pitch`` = tip the cap **top
  forward** / nose-down (about +y), ``+yaw`` = twist **counterclockwise**
  viewed from above (about +z).

This matches the usual robot base frame convention (x forward, y left, z up).

The raw ``"hid"`` convention (USB HID Usage Tables section 4.2, right-handed
with z down) instead reads: translation ``+x`` right, ``+y`` toward the
operator, ``+z`` down; ``+yaw`` = clockwise from above. ``"hid_z_up"`` keeps
HID x and flips y/z to a z-up right-handed frame. Do NOT use PySpaceMouse's
``"legacy"`` convention: its roll/pitch labels are swapped relative to the
physical axes and its yaw sign is wrong — it exists only for upstream
backward compatibility, and this node deliberately does not offer it.

Each hardware button is exposed as a separate named observation. Non-blocking
reads that return ``None`` preserve the previous observation. Constructing with
``connect=False`` produces a correctly shaped zero observation without
importing ``pyspacemouse``.
"""

from typing import Dict, Optional, Sequence

import numpy as np

from unienv_interface.backends import ComputeBackend
from unienv_interface.backends.numpy import (
    NumpyArrayType,
    NumpyComputeBackend,
    NumpyDeviceType,
    NumpyDtypeType,
    NumpyRNGType,
)
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.world import RealWorld, World, WorldNode


class SpaceMouseNode(WorldNode[
    None, Dict[str, NumpyArrayType], None,
    NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
]):
    """Observation-only WorldNode for one path-selected SpaceMouse.

    The observation contains ``translation`` and ``rotation`` float32 vectors
    of shape ``(3,)``, plus one float32 ``(1,)`` ``{name}_button`` entry for
    every resolved button name. ``action_space`` is ``None``.

    With ``button_names=None``, connected nodes derive names from the opened
    device specification. Offline nodes use ``("left", "right")`` so their
    space is deterministic. A later connection is rejected if its resolved
    button set does not match an already-built observation space.
    """

    after_reset_priorities = {0}
    after_reload_priorities = {0}
    post_environment_step_priorities = {0}

    _OFFLINE_BUTTON_NAMES = ("left", "right")
    _AXIS_CONVENTIONS = {
        "ros": "ROS",
        "hid_z_up": "HID_Z_UP",
        "hid": "HID",
        "unity": "UNITY",
    }

    def __init__(
        self,
        world: Optional[RealWorld] = None,
        name: str = "spacemouse",
        *,
        path: str,
        button_names: Optional[Sequence[str]] = None,
        axis_convention: str = "ros",
        deadband: float = 0.0,
        connect: bool = True,
        control_timestep: Optional[float] = 0.04,
        update_timestep: Optional[float] = 0.04,
    ):
        self.name = name
        self.path = str(path)
        self.axis_convention = str(axis_convention).lower()
        self.deadband = float(deadband)
        self.connect = bool(connect)

        self._device = None
        self._space_button_names: Optional[tuple[str, ...]] = None
        self._explicit_button_names = button_names is not None
        self.button_names = (
            tuple(str(button).lower() for button in button_names)
            if button_names is not None
            else self._OFFLINE_BUTTON_NAMES
        )
        self._button_indices = tuple(range(len(self.button_names)))

        if self.axis_convention not in self._AXIS_CONVENTIONS:
            valid = ", ".join(repr(value) for value in self._AXIS_CONVENTIONS)
            raise ValueError(
                f"SpaceMouseNode: `axis_convention` must be one of {valid}, "
                f"got {axis_convention!r}."
            )
        if not 0.0 <= self.deadband < 1.0:
            raise ValueError(
                "SpaceMouseNode: `deadband` must be in [0, 1), "
                f"got {deadband!r}."
            )

        if isinstance(world, World):
            assert world.backend == NumpyComputeBackend, \
                "World backend must be NumpyComputeBackend."
            assert world.is_control_timestep_compatible(control_timestep), \
                "Control timestep must be a multiple of world timestep."
        self.world = world

        if self.connect:
            self._connect_hardware()

        spaces: Dict[str, BoxSpace] = {
            "translation": BoxSpace(
                NumpyComputeBackend,
                low=-1.0,
                high=1.0,
                dtype=np.float32,
                shape=(3,),
            ),
            "rotation": BoxSpace(
                NumpyComputeBackend,
                low=-1.0,
                high=1.0,
                dtype=np.float32,
                shape=(3,),
            ),
        }
        for button in self.button_names:
            spaces[f"{button}_button"] = BoxSpace(
                NumpyComputeBackend,
                low=0.0,
                high=1.0,
                dtype=np.float32,
                shape=(1,),
            )
        self.observation_space = DictSpace(NumpyComputeBackend, spaces)
        self._space_button_names = self.button_names
        self.action_space = None

        self.control_timestep = control_timestep
        self.update_timestep = update_timestep
        self._current_observation: Dict[str, NumpyArrayType] = self._zero_observation()

    # ========== Hardware connection ==========

    def _connect_hardware(self) -> None:
        """Open this node's device handle and resolve its specification."""
        try:
            import pyspacemouse  # noqa: PLC0415 - optional dependency
        except ImportError as e:  # pragma: no cover - depends on host setup
            raise ImportError(
                "SpaceMouseNode(connect=True) requires pyspacemouse and hidapi. "
                "Install with `pip install \"unienv-input-devices[spacemouse]\"`. "
                "On Linux, also install `libhidapi-dev` and configure a "
                "udev rule granting access to the SpaceMouse hidraw device."
            ) from e

        convention = getattr(
            pyspacemouse.AxisConvention,
            self._AXIS_CONVENTIONS[self.axis_convention],
        )
        try:
            device = pyspacemouse.open_by_path(
                self.path,
                nonblocking=True,
                axis_convention=convention,
            )
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as e:
            raise RuntimeError(
                f"SpaceMouseNode: no supported SpaceMouse could be opened at "
                f"path {self.path!r}. Check the path and device permissions."
            ) from e
        if device is None:
            raise RuntimeError(
                f"SpaceMouseNode: no supported SpaceMouse could be opened at "
                f"path {self.path!r}. Check the path and device permissions."
            )

        try:
            info = self._device_info(device)
            device_button_names = self._device_button_names(device, info)

            if self._explicit_button_names:
                missing = [name for name in self.button_names if name not in device_button_names]
                if missing:
                    raise ValueError(
                        "SpaceMouseNode: explicit button name(s) not present in "
                        f"the device spec: {missing!r}."
                    )
                resolved_names = self.button_names
            else:
                resolved_names = device_button_names

            if (
                self._space_button_names is not None
                and set(resolved_names) != set(self._space_button_names)
            ):
                raise ValueError(
                    "SpaceMouseNode: connected device button names do not match "
                    "the existing observation space. Construct a new node with "
                    "matching `button_names`."
                )

            self.button_names = tuple(resolved_names)
            self._button_indices = tuple(
                device_button_names.index(name) for name in self.button_names
            )
            self._device = device
        except Exception:
            try:
                device.close()
            except Exception:
                pass
            raise

    @staticmethod
    def _device_info(device):
        """Return DeviceInfo across the common public/fallback attributes."""
        for attribute in ("info", "spec", "device_info", "_info"):
            info = getattr(device, attribute, None)
            if info is not None:
                return info
        return device

    @staticmethod
    def _device_button_names(device, info) -> tuple[str, ...]:
        names = getattr(info, "button_names", None)
        if names is None:
            names = getattr(device, "button_names", None)
        if names is None:
            specs = getattr(info, "button_specs", ())
            names = [
                spec if isinstance(spec, str) else getattr(spec, "name")
                for spec in specs
            ]
        return tuple(str(name).lower() for name in names)

    # ========== Internal helpers ==========

    def _zero_observation(self) -> Dict[str, NumpyArrayType]:
        """Build an all-zero observation with the configured keys and dtypes."""
        observation: Dict[str, NumpyArrayType] = {
            "translation": np.zeros((3,), dtype=np.float32),
            "rotation": np.zeros((3,), dtype=np.float32),
        }
        for button in self.button_names:
            observation[f"{button}_button"] = np.zeros((1,), dtype=np.float32)
        return observation

    def _normalize_axes(self, values) -> np.ndarray:
        """Apply deadband and clip; PySpaceMouse already normalized to [-1, 1]."""
        normalized = np.asarray(values, dtype=np.float32)
        normalized[np.abs(normalized) < self.deadband] = 0.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32, copy=False)

    def _build_observation(self, state) -> Dict[str, NumpyArrayType]:
        observation: Dict[str, NumpyArrayType] = {
            "translation": self._normalize_axes((state.x, state.y, state.z)),
            "rotation": self._normalize_axes((state.roll, state.pitch, state.yaw)),
        }
        for name, index in zip(self.button_names, self._button_indices):
            observation[f"{name}_button"] = np.asarray(
                [1.0 if state.buttons[index] else 0.0],
                dtype=np.float32,
            )
        return observation

    # ========== WorldNode implementation ==========

    @property
    def backend(self) -> ComputeBackend[
        NumpyArrayType, NumpyDeviceType, NumpyDtypeType, NumpyRNGType
    ]:
        return NumpyComputeBackend

    @property
    def device(self) -> None:
        return None

    def post_environment_step(self, dt: float, *, priority: int = 0) -> None:
        """Poll once, holding the previous observation if no report is ready."""
        if not self.connect or self._device is None:
            return
        state = self._device.read()
        if state is None:
            return
        self._current_observation = self._build_observation(state)

    def after_reset(self, *, priority: int = 0, mask=None) -> None:
        self.post_environment_step(0.0, priority=priority)

    def get_observation(self) -> Dict[str, NumpyArrayType]:
        return self._current_observation

    def set_next_action(self, action) -> None:
        """Accept and ignore actions because this node is observation-only."""
        return

    def close(self) -> None:
        """Best-effort closure of only this node's device handle."""
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None

    # ========== Public helpers ==========

    def get_translation(self) -> np.ndarray:
        return np.asarray(self._current_observation["translation"], dtype=np.float32)

    def get_rotation(self) -> np.ndarray:
        return np.asarray(self._current_observation["rotation"], dtype=np.float32)

    def get_twist(self) -> np.ndarray:
        return np.concatenate((self.get_translation(), self.get_rotation())).astype(
            np.float32,
            copy=False,
        )

    def get_button(self, name: str) -> bool:
        normalized = str(name).lower()
        if normalized not in self.button_names:
            raise ValueError(
                f"SpaceMouseNode: unknown button {name!r}; available buttons: "
                f"{self.button_names!r}."
            )
        return bool(self._current_observation[f"{normalized}_button"][0])

    @classmethod
    def list_devices(cls) -> dict[str, str]:
        """Return ``{hid_path: device_name}`` for connected SpaceMice.

        Wraps ``pyspacemouse.get_connected_devices_by_path()`` (v2 API), e.g.
        ``{"/dev/hidraw3": "SpaceMouse Compact"}`` — the paths are exactly
        what :class:`SpaceMouseNode` takes as its ``path`` argument.
        """
        try:
            import pyspacemouse  # noqa: PLC0415 - optional dependency
        except ImportError as e:  # pragma: no cover - depends on host setup
            raise ImportError(
                "SpaceMouseNode.list_devices() requires pyspacemouse and hidapi. "
                "Install with `pip install \"unienv-input-devices[spacemouse]\"`. "
                "On Linux, also install `libhidapi-dev` and configure a "
                "udev rule granting access to SpaceMouse hidraw devices."
            ) from e
        return dict(pyspacemouse.get_connected_devices_by_path())
