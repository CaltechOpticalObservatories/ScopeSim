# -*- coding: utf-8 -*-
"""Image-plane illumination effects."""

from collections.abc import Callable, Mapping
from typing import ClassVar

import numpy as np
from astropy import units as u
from astropy.units import UnitConversionError
from astropy.modeling.functional_models import Gaussian2D
from synphot.units import PHOTLAM

from . import Effect
from .surface_list import SurfaceList
from .ter_curves import SpectralQuantumEfficiency
from ..optics.image_plane import ImagePlane
from ..utils import figure_factory, from_currsys, quantify, real_colname


_POST_DISPERSER_RATE_CACHE_MAXSIZE = 64
_POST_DISPERSER_RATE_CACHE: dict[tuple, float] = {}


__all__ = [
    "Illumination",
    "ImagePlaneBackground",
    "PostDisperserDiffuseBackground",
    "effective_diffuse_qe",
    "gaussian2d",
    "image_plane_pixel_area",
    "integrate_spectral_background",
    "post_disperser_diffuse_spectrum",
    "quadratic_vignetting",
    "wavelength_bin_widths",
]


def gaussian2d(
    shape: tuple[int, int],
    amp: float = 1.0,
    mu: tuple[float, float] = (0.0, 0.0),
    sigma: tuple[float, float] = (2000.0, 2000.0),
    theta: u.Quantity[u.deg] | float = 0.0 * u.deg,
) -> np.ndarray:
    """Return a 2D elliptical Gaussian illumination map."""
    nx, ny = reversed(shape)
    y, x = np.ogrid[:ny, :nx]
    x = x - nx / 2
    y = y - ny / 2

    model = Gaussian2D(
        amplitude=amp,
        x_mean=mu[0],
        y_mean=mu[1],
        x_stddev=sigma[0],
        y_stddev=sigma[1],
        theta=theta << u.deg,
    )
    return model(x, y)


def quadratic_vignetting(
    shape: tuple[int, int],
    falloff: float = 0.01,
    r_ref: float | None = None,
    mu: tuple[float, float] = (0.0, 0.0),
    stretch: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> np.ndarray:
    """Return a quadratic vignetting pattern."""
    nx, ny = reversed(shape)

    yy, xx = np.ogrid[:ny, :nx]
    dx = xx - (nx / 2 + mu[0])
    dy = yy - (ny / 2 + mu[1])

    sx = np.where(dx >= 0, stretch[0], stretch[1])
    sy = np.where(dy >= 0, stretch[2], stretch[3])

    r2 = (dx / sx) ** 2 + (dy / sy) ** 2
    r2_ref = r2.max() if r_ref is None else r_ref ** 2

    return np.clip(1.0 - falloff * r2 / r2_ref, 0.0, 1.0)


def _as_float_array(values) -> np.ndarray:
    if hasattr(values, "value"):
        values = values.value
    return np.asarray(values, dtype=float)


def wavelength_bin_widths(wave: u.Quantity) -> u.Quantity:
    """Return centre-bin widths for a sampled wavelength grid."""
    wave = wave.to(u.um)
    if wave.size < 2:
        raise ValueError("At least two wavelength samples are required.")

    wave_values = wave.to_value(wave.unit)
    if np.any(np.diff(wave_values) <= 0):
        raise ValueError("Wavelength samples must be strictly increasing.")

    widths = np.zeros(wave.size, dtype=float)
    diffs = np.diff(wave_values)
    widths[:-1] += 0.5 * diffs
    widths[1:] += 0.5 * diffs
    return widths * wave.unit


def _representative_positional_qe(positional_qe, wave: u.Quantity):
    if positional_qe is None:
        return 1.0
    values = positional_qe(wave) if callable(positional_qe) else positional_qe
    values = _as_float_array(values)
    if values.shape == wave.shape:
        return values
    return float(np.nanmean(values))


def _cache_object_key(filename, obj):
    if filename is not None:
        return ("file", str(filename))
    if obj is None:
        return None
    return ("object", id(obj))


def _image_plane_cache_key(image_plane):
    header = image_plane.header
    return (
        header.get("EXTNAME"),
        header.get("IMAGEID"),
        header.get("IMGPLANE"),
        header.get("CDELT1"),
        header.get("CDELT2"),
        header.get("CDELT1D"),
        header.get("CDELT2D"),
    )


def _store_post_disperser_rate_cache(key: tuple, value: float) -> None:
    if len(_POST_DISPERSER_RATE_CACHE) >= _POST_DISPERSER_RATE_CACHE_MAXSIZE:
        _POST_DISPERSER_RATE_CACHE.clear()
    _POST_DISPERSER_RATE_CACHE[key] = value


def effective_diffuse_qe(
    detector_qe,
    wave: u.Quantity,
    positional_qe=None,
) -> np.ndarray:
    """Return effective detector QE for non-dispersed diffuse backgrounds.

    Ordinary detector QE contributes its spectral throughput. Future tapered
    QE coatings can provide a positional map/callable; this function reduces
    that positional response to a representative footprint average instead of
    skipping QE for diffuse image-plane backgrounds.
    """
    if detector_qe is None:
        spectral_qe = np.ones(wave.size, dtype=float)
    else:
        spectral_qe = _as_float_array(detector_qe.throughput(wave))
    return spectral_qe * _representative_positional_qe(positional_qe, wave)


def _row_value(row, name):
    value = row[name]
    return value.item() if hasattr(value, "item") else value


def _clean_text(value) -> str | None:
    text = str(value).strip()
    if not text or text in {"--", "None", "nan"}:
        return None
    return text


def _row_phase(row) -> str | None:
    phase_col = real_colname("emission_phase", row.colnames)
    if phase_col is None:
        return None
    return _clean_text(_row_value(row, phase_col))


def _surface_emission_values(surface, wave: u.Quantity):
    emission = surface.emission
    if emission is None:
        return None
    values = emission(wave)
    if not isinstance(values, u.Quantity):
        values = values * PHOTLAM
    return values


def post_disperser_diffuse_spectrum(
    surface_list,
    wave: u.Quantity,
    qe_values: np.ndarray | None = None,
    emission_phase: str = "post_disperser",
):
    """Return summed post-disperser diffuse emission after downstream optics.

    This mirrors :meth:`SurfaceList.combine_emissions`, but selects emitting
    rows by explicit ``emission_phase`` metadata instead of by z-order. The
    returned spectrum is still a surface-brightness-like spectral density; use
    :func:`integrate_spectral_background` to collapse it to image-plane
    ``ph s-1 pixel-1``.
    """
    if surface_list.table is None or len(surface_list.table) == 0:
        return None

    name_col = real_colname("name", surface_list.table.colnames)
    action_col = real_colname("action", surface_list.table.colnames)
    if name_col is None or action_col is None:
        raise ValueError("SurfaceList table must contain name and action columns.")

    rows = []
    for row in surface_list.table:
        surface_name = str(_row_value(row, name_col))
        action_name = str(_row_value(row, action_col))
        surface = surface_list.surfaces[surface_name]
        rows.append({
            "phase": _row_phase(row),
            "surface": surface,
            "action_values": _as_float_array(getattr(surface, action_name)(wave)),
            "emission_values": _surface_emission_values(surface, wave),
        })

    downstream = [np.ones(wave.size, dtype=float) for _ in range(len(rows) + 1)]
    for idx in range(len(rows) - 1, -1, -1):
        downstream[idx] = downstream[idx + 1] * rows[idx]["action_values"]

    if qe_values is None:
        qe_values = np.ones(wave.size, dtype=float)

    total = None
    for idx, row in enumerate(rows):
        if row["phase"] != emission_phase or row["emission_values"] is None:
            continue
        contribution = row["emission_values"] * downstream[idx + 1] * qe_values
        total = contribution if total is None else total + contribution

    return total


def integrate_spectral_background(
    spectral_density,
    wave: u.Quantity,
    telescope_area: u.Quantity,
    image_pixel_area: u.Quantity,
) -> float:
    """Integrate a diffuse spectrum to ScopeSim image-plane units.

    The returned scalar is ``ph s-1 pixel-1``. If the spectrum already carries
    an inverse-solid-angle unit, that unit is converted explicitly. Otherwise
    the result follows ScopeSim's current ``BackgroundSourceField`` convention:
    PHOTLAM-like thermal spectra are interpreted as per square arcsecond, then
    multiplied by the image-plane pixel area.
    """
    if spectral_density is None:
        return 0.0
    if not isinstance(spectral_density, u.Quantity):
        spectral_density = spectral_density * PHOTLAM

    widths = wavelength_bin_widths(wave).to(u.AA)
    telescope_area = telescope_area << u.m**2
    image_pixel_area = image_pixel_area << u.arcsec**2

    rate = np.sum(spectral_density * widths * telescope_area)
    try:
        return (rate * image_pixel_area).to_value(u.ph / u.s)
    except UnitConversionError:
        return rate.to_value(u.ph / u.s) * image_pixel_area.to_value(u.arcsec**2)


def image_plane_pixel_area(header, cmds=None) -> u.Quantity:
    """Return the angular area represented by one image-plane pixel.

    Some spectroscopic image-plane headers only carry detector WCS keywords
    such as ``CDELT1D``/``CUNIT1D``. Those are detector lengths, not sky angles,
    so diffuse surface-brightness backgrounds need the instrument angular pixel
    scale instead of :func:`scopesim.utils.pixel_area`.
    """
    for suffix in ("", "S", "D"):
        area = _angular_pixel_area_from_header(header, suffix)
        if area is not None:
            return area

    detector_area = _detector_pixel_area_from_header(header, cmds)
    if detector_area is not None:
        return detector_area

    if cmds is not None and "!INST.pixel_scale" in cmds:
        scale = quantify(from_currsys("!INST.pixel_scale", cmds), u.arcsec)
        return (abs(scale) ** 2).to(u.arcsec**2)

    raise KeyError(
        "Image-plane header has no angular WCS pixel scale and "
        "!INST.pixel_scale is unavailable."
    )


def _angular_pixel_area_from_header(header, suffix: str) -> u.Quantity | None:
    keys = (f"CDELT1{suffix}", f"CUNIT1{suffix}",
            f"CDELT2{suffix}", f"CUNIT2{suffix}")
    if not all(key in header for key in keys):
        return None

    unit1 = u.Unit(header[keys[1]])
    unit2 = u.Unit(header[keys[3]])
    area = abs(header[keys[0]] * header[keys[2]]) * unit1 * unit2
    if area.unit.is_equivalent(u.arcsec**2):
        return area.to(u.arcsec**2)
    return None


def _detector_pixel_area_from_header(header, cmds) -> u.Quantity | None:
    keys = ("CDELT1D", "CUNIT1D", "CDELT2D", "CUNIT2D")
    if cmds is None or not all(key in header for key in keys):
        return None

    unit1 = u.Unit(header["CUNIT1D"])
    unit2 = u.Unit(header["CUNIT2D"])
    if not (
        unit1.is_equivalent(u.mm)
        and unit2.is_equivalent(u.mm)
        and "!INST.plate_scale" in cmds
    ):
        return None

    plate_scale = quantify(
        from_currsys("!INST.plate_scale", cmds), u.arcsec / u.mm,
    )
    dx = abs(header["CDELT1D"]) * unit1
    dy = abs(header["CDELT2D"]) * unit2
    return (dx * plate_scale * dy * plate_scale).to(u.arcsec**2)


class Illumination(Effect):
    """Large-scale multiplicative illumination variation on the image plane."""

    z_order: ClassVar[tuple[int, ...]] = (750,)

    def __init__(
        self,
        model: Callable = gaussian2d,
        modelargs: Mapping | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.meta.setdefault("include", "!DET.include_illumination")
        self._model = model
        self._modelargs = modelargs or {}
        self._map = None
        self._map_shape = None

    def apply_to(self, obj, **kwargs):
        if not isinstance(obj, ImagePlane):
            return obj

        shape = obj.hdu.data.shape
        if self._map is None or shape != self._map_shape:
            self._map = self._make_map(shape)
            self._map_shape = shape

        obj.hdu.data *= self._map
        return obj

    def _make_map(self, shape):
        illumination_map = self._model(shape, **self._modelargs)
        return illumination_map.astype(np.float32)

    def plot(self):
        """Plot the cached illumination map."""
        if self._map is None:
            raise RuntimeError("No illumination map cached - run a simulation first.")

        fig, ax = figure_factory()
        im = ax.imshow(
            self._map, origin="lower", vmin=0.98, vmax=1.0, cmap="gray_r",
        )
        fig.colorbar(im, ax=ax, label="Relative illumination")
        ax.set_title("Illumination")
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        return fig


class ImagePlaneBackground(Effect):
    """Add an already integrated diffuse background to the image plane.

    The value added by this effect is in ScopeSim image-plane units,
    ``ph s-1 pixel-1``. Use this for post-disperser diffuse backgrounds that
    should not be converted into a source cube and sent through the spectral
    trace list. The supplied value or model should already include downstream
    throughput and detector QE. For tapered QE maps, use an average or otherwise
    representative positional QE when deriving this non-dispersed background.
    """

    z_order: ClassVar[tuple[int, ...]] = (760,)

    def __init__(
        self,
        value: float = 0.0,
        model: Callable | None = None,
        modelargs: Mapping | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.meta.setdefault("include", "!DET.include_image_plane_background")
        self.meta.setdefault("value", value)
        self._model = model
        self._modelargs = modelargs or {}
        self._map = None
        self._map_shape = None

    def apply_to(self, obj, **kwargs):
        if not isinstance(obj, ImagePlane):
            return obj

        shape = obj.hdu.data.shape
        if self._map is None or shape != self._map_shape:
            self._map = self._make_map(shape)
            self._map_shape = shape

        obj.hdu.data = obj.hdu.data + self._map
        return obj

    def _make_map(self, shape):
        if self._model is None:
            value = from_currsys(self.meta["value"], self.cmds)
            background = np.full(shape, value, dtype=np.float32)
        else:
            background = self._model(shape, **self._modelargs)
        return np.asarray(background, dtype=np.float32)

    def plot(self):
        """Plot the cached background map."""
        if self._map is None:
            raise RuntimeError("No background map cached - run a simulation first.")

        fig, ax = figure_factory()
        im = ax.imshow(self._map, origin="lower")
        fig.colorbar(im, ax=ax, label="ph s-1 pixel-1")
        ax.set_title("Image-plane background")
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        return fig


class PostDisperserDiffuseBackground(ImagePlaneBackground):
    """Add thermal emission from post-disperser optics as image-plane light.

    The source surfaces are selected by explicit ``emission_phase`` metadata in
    a ``SurfaceList`` table. This avoids using z-order as a proxy for optical
    phase: pre-disperser emission remains ordinary source/FOV background, while
    post-disperser emission is integrated and added after trace mapping.
    Detector QE is applied as throughput only through
    :func:`effective_diffuse_qe`.
    """

    z_order: ClassVar[tuple[int, ...]] = (760,)

    def __init__(
        self,
        filename: str | None = None,
        detector_qe_filename: str | None = None,
        surface_list=None,
        detector_qe=None,
        positional_qe=None,
        emission_phase: str = "post_disperser",
        **kwargs,
    ) -> None:
        super().__init__(value=0.0, **kwargs)
        self.meta["include"] = kwargs.get("include", True)
        self.meta.update({
            "filename": filename,
            "detector_qe_filename": detector_qe_filename,
            "wave_min": kwargs.get("wave_min", "!SIM.spectral.wave_min"),
            "wave_max": kwargs.get("wave_max", "!SIM.spectral.wave_max"),
            "wave_bin": kwargs.get("wave_bin", "!SIM.spectral.spectral_bin_width"),
            "wave_unit": kwargs.get("wave_unit", "!SIM.spectral.wave_unit"),
            "area": kwargs.get("area", "!TEL.area"),
            "emission_phase": emission_phase,
        })
        self._surface_list = (
            surface_list if surface_list is not None
            else SurfaceList(filename=filename, cmds=self.cmds)
        )
        self._detector_qe = (
            detector_qe if detector_qe is not None
            else (
                SpectralQuantumEfficiency(
                    filename=detector_qe_filename, cmds=self.cmds)
                if detector_qe_filename is not None
                else None
            )
        )
        self._positional_qe = positional_qe
        self._last_value = None

    def apply_to(self, obj, **kwargs):
        if not isinstance(obj, ImagePlane):
            return obj

        shape = obj.hdu.data.shape
        value = self.background_value(obj)
        if (
            self._map is None or shape != self._map_shape
            or value != self._last_value
        ):
            self._map = np.full(shape, value, dtype=np.float32)
            self._map_shape = shape
            self._last_value = value

        obj.hdu.data = obj.hdu.data + self._map
        return obj

    def background_value(self, image_plane: ImagePlane) -> float:
        """Return the scalar background in ``ph s-1 pixel-1``."""
        rate_per_arcsec2 = self._background_rate_per_arcsec2(image_plane)
        pixel_area = image_plane_pixel_area(
            image_plane.header, self.cmds,
        ).to_value(u.arcsec**2)
        return rate_per_arcsec2 * pixel_area

    def _background_rate_per_arcsec2(self, image_plane: ImagePlane) -> float:
        """Return the image-plane background rate before pixel-area scaling."""
        key = self._background_rate_cache_key(image_plane)
        if key in _POST_DISPERSER_RATE_CACHE:
            return _POST_DISPERSER_RATE_CACHE[key]

        wave = self._waveset()
        qe_values = effective_diffuse_qe(
            self._detector_qe, wave, positional_qe=self._positional_qe,
        )
        spectrum = post_disperser_diffuse_spectrum(
            self._surface_list,
            wave,
            qe_values=qe_values,
            emission_phase=self.meta["emission_phase"],
        )
        area = quantify(from_currsys(self.meta["area"], self.cmds), u.m**2)
        rate = integrate_spectral_background(
            spectrum,
            wave,
            telescope_area=area,
            image_pixel_area=1.0 * u.arcsec**2,
        )
        _store_post_disperser_rate_cache(key, rate)
        return rate

    def _background_rate_cache_key(self, image_plane: ImagePlane) -> tuple:
        wave_unit = u.Unit(from_currsys(self.meta["wave_unit"], self.cmds))
        wave_min = quantify(
            from_currsys(self.meta["wave_min"], self.cmds), wave_unit,
        ).to_value(wave_unit)
        wave_max = quantify(
            from_currsys(self.meta["wave_max"], self.cmds), wave_unit,
        ).to_value(wave_unit)
        wave_bin = quantify(
            from_currsys(self.meta["wave_bin"], self.cmds), wave_unit,
        ).to_value(wave_unit)
        area = quantify(
            from_currsys(self.meta["area"], self.cmds), u.m**2,
        ).to_value(u.m**2)
        filename = self.meta["filename"]
        qe_filename = self.meta["detector_qe_filename"]
        positional_qe_key = None
        if self._positional_qe is not None:
            positional_qe_key = (
                id(self._positional_qe),
                _image_plane_cache_key(image_plane),
            )
        return (
            id(self.cmds),
            _cache_object_key(filename, self._surface_list),
            _cache_object_key(qe_filename, self._detector_qe),
            positional_qe_key,
            str(self.meta["emission_phase"]),
            str(wave_unit),
            float(wave_min),
            float(wave_max),
            float(wave_bin),
            float(area),
        )

    def _waveset(self) -> u.Quantity:
        wave_unit = u.Unit(from_currsys(self.meta["wave_unit"], self.cmds))
        wave_min = quantify(from_currsys(self.meta["wave_min"], self.cmds),
                            wave_unit).to(wave_unit)
        wave_max = quantify(from_currsys(self.meta["wave_max"], self.cmds),
                            wave_unit).to(wave_unit)
        wave_bin = quantify(from_currsys(self.meta["wave_bin"], self.cmds),
                            wave_unit).to(wave_unit)
        stop = wave_max.to_value(wave_unit) + 0.5 * wave_bin.to_value(wave_unit)
        wave = np.arange(
            wave_min.to_value(wave_unit),
            stop,
            wave_bin.to_value(wave_unit),
        ) * wave_unit
        if wave.size < 2:
            raise ValueError("Post-disperser background wavelength grid is empty.")
        return wave
