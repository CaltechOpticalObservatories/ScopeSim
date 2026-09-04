"""
A new effect module that enables multi-arm/branch instruments by allowing application of different effects of same type
to different FoV objects. For e.g. for a simultaneous 2-arm instrument (red and blue), if aperture_id=1 corresponds to
FoV objects of blue arm and aperture_id=2 to FoV objects of the red arm (and they already have non-overlapping wavelengths),
most of the existing effects in the optical train drop the FoV objects that are outside their spatio-spectral volume.
There is a need of an effect that stores different effects of the same type that apply to different FoV objects based on
a selector (e.g. aperture_id).

This module implements a SelectorWheel effect that allows the user to define multiple effects of the same type
( e.g. different aperture masks for different arms) in the wheel dictionary where each effect corresponds to a
"selector_id" value. The user can set which id to use as the "selector", for e.g. aperture_id or id of the FoV object.
"""
import importlib
from numbers import Integral

from ..utils import (check_keys, get_logger, real_colname)
from .effects import Effect
from ..optics.fov_volume_list import FovVolumeList
from ..optics.fov import FieldOfView
from ..optics.image_plane import ImagePlane
from ..detector.detector import Detector

logger = get_logger(__name__)


class SelectorWheel(Effect):
    """
    SelectorWheel class - that applies different effect objects (of the same type) based on a selector value.

    Examples:
    ---------
    ::
        name: aperture_selector_wheel
        class: SelectorWheel
        kwargs:
            selector_key: aperture_id
            wheel:
                - selector_value: 0
                  effect_class: ApertureList
                  effect_kwargs:
                      filename: "slits/slit_nir.txt"
                - selector_value: 1
                  effect_class: ApertureList
                  effect_kwargs:
                      filename: "slits/slit_vis.txt"

    In the above example, the SelectorWheel effect applies different aperture masks to FoV objects
    based on their aperture_id value. If a FoV object has aperture_id=0, it gets the NIR slit mask applied,
    while aperture_id=1 gets the VIS slit mask. This allows for multi-arm instruments to have different
    aperture masks applied in a single optical train. The effect_class specified in the wheel entries
    must be a valid Effect subclass available in scopesim.effects module.

    """

    z_order = ()
    required_keys = {"selector_key", "wheel"}

    def __init__(self, **kwargs):
        check_keys(kwargs, self.required_keys, action="error")
        super().__init__(**kwargs)

        self.wheel_effects = {}
        for wheel_entry in self.meta["wheel"]:
            selector_value = wheel_entry["selector_value"] # can be single value or list of values
            effect_class_name = wheel_entry["effect_class"]
            effect_kwargs = wheel_entry.get("effect_kwargs", {})

            # Dynamically import the effect class
            effect_module = importlib.import_module("scopesim.effects")
            effect_class = getattr(effect_module, effect_class_name)
            # Instantiate the effect and store it in the wheel_effects dictionary
            if isinstance(selector_value, list):
                for val in selector_value:
                    self.wheel_effects[val] = effect_class(cmds=self.cmds, **effect_kwargs)
            else:
                self.wheel_effects[selector_value] = effect_class(cmds=self.cmds, **effect_kwargs)

        self.z_order = self._resolve_z_order()


    def apply_to(self, obj, **kwargs):
        """Based on the selector_key's value in obj.meta, apply the corresponding effect from the wheel."""
        if isinstance(obj, FieldOfView):
            if self.meta['selector_key'] not in obj.meta.keys():
                raise ValueError(f"Selector key {self.meta['selector_key']} not found in FieldOfView meta.")
            selector_value = obj.meta[self.meta["selector_key"]]

            effect_to_apply = self.get_effect(selector_value)
            if effect_to_apply is None:
                return obj

            obj = effect_to_apply.apply_to(obj, **kwargs)

        if isinstance(obj, FovVolumeList):
            unique_selector_values = set([vol["meta"].get(self.meta["selector_key"], None) for vol in obj.volumes])
            new_volumes = []

            for val in unique_selector_values:
                vols_with_val = [vol for vol in obj.volumes if vol["meta"].get(self.meta["selector_key"], None) == val]

                if val is None:
                    logger.warning(f"Volume(s) with missing selector key {self.meta['selector_key']} value found, "
                                   f"applying no effect to those volumes.")
                    new_volumes.extend(vols_with_val)
                    continue

                effect_to_apply = self.get_effect(val)

                if effect_to_apply is None:
                    new_volumes.extend(vols_with_val)
                    continue

                logger.debug(
                    f"Applying effect for {self.meta['selector_key']}: "
                    f"{val} -> {effect_to_apply}, volumes: {len(vols_with_val)}"
                )
                newvollist = FovVolumeList()
                newvollist.volumes = vols_with_val
                newvollist = effect_to_apply.apply_to(newvollist, **kwargs)
                new_volumes.extend(newvollist.volumes)

            obj.volumes = new_volumes

        if isinstance(obj, Detector):
            logger.debug("Since passed object is a Detector, selector_key by default is the ID of the Detector object.")
            selector_value = obj.meta[real_colname("id", obj.meta)] # Assuming detector ID is the selector

            effect_to_apply = self.get_effect(selector_value)
            if effect_to_apply is None:
                return obj

            obj = effect_to_apply.apply_to(obj, **kwargs)

        if isinstance(obj, ImagePlane):
            selector_value = self._selector_value_from_image_plane(obj)
            effect_to_apply = self.get_effect(selector_value)
            if effect_to_apply is None:
                return obj

            obj = effect_to_apply.apply_to(obj, **kwargs)

        return obj


    def get_effect(self, selector_value):
        eff = None
        if selector_value not in self.wheel_effects.keys():
            if self._is_missing_selector_value_allowed(selector_value):
                logger.debug(
                    "Entry for selector value %s intentionally absent from wheel effects.",
                    selector_value,
                )
            else:
                logger.warning(f"Entry for selector value {selector_value} not found in wheel effects. "
                               f"Assuming no effect to apply for this selector value.")
        else:
            eff = self.wheel_effects[selector_value]
        return eff

    def _is_missing_selector_value_allowed(self, selector_value):
        values = self.meta.get(
            "allowed_missing_selector_values",
            self.meta.get("allow_missing_selector_values", ()),
        )
        if values is None:
            return False
        if isinstance(values, str):
            if values.lower() in {"all", "any", "*"}:
                return True
            values = (values,)
        elif not isinstance(values, (list, tuple, set, frozenset)):
            values = (values,)
        return selector_value in values

    def _resolve_z_order(self):
        """Use an explicit wheel z_order if supplied, otherwise inherit one."""
        configured_z_order = self.meta.get("z_order")
        if configured_z_order is not None:
            if isinstance(configured_z_order, Integral):
                return (int(configured_z_order),)
            return tuple(configured_z_order)

        if not self.wheel_effects:
            return ()
        return tuple(next(iter(self.wheel_effects.values())).z_order)


    def _selector_value_from_image_plane(self, obj):
        selector_key = self.meta["selector_key"]
        if selector_key in obj.meta:
            return obj.meta[selector_key]
        if selector_key in obj.hdu.header:
            return obj.hdu.header[selector_key]
        if selector_key in {"id", "image_plane_id", "IMGPLANE"}:
            return obj.id
        raise ValueError(f"Selector key {selector_key} not found in ImagePlane.")
