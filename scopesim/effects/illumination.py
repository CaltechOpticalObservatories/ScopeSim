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
from ..utils import figure_factory, from_currsys, pixel_area, quantify, real_colname


__all__ = [
    "Illumination",
    "ImagePlaneBackground",
    "PostDisperserDiffuseBackground",
    "effective_diffuse_qe",
    "gaussian2d",
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
        return integrate_spectral_background(
            spectrum,
            wave,
            telescope_area=area,
            image_pixel_area=pixel_area(image_plane.header),
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
