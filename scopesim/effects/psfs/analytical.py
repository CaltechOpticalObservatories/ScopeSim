# -*- coding: utf-8 -*-
"""Contains simple Vibration, NCPA, Seeing and Diffraction PSFs."""

from typing import ClassVar, Any
from functools import partial

import numpy as np
from astropy import units as u
from astropy.convolution import Gaussian2DKernel
from astropy.modeling.models import Moffat2D
from scipy.interpolate import make_interp_spline

from .. import DataContainer
from ...optics import ImagePlane
from ...optics.fov import FieldOfView
from ...utils import (from_currsys, quantify, quantity_from_table, figure_factory, check_keys, get_logger, find_file,
                      get_zenith_angle, get_observation_info_from_cmds)
from . import PSF, PoorMansFOV

logger = get_logger(__name__)

class AnalyticalPSF(PSF):
    """Base class for analytical PSFs."""

    z_order: ClassVar[tuple[int, ...]] = (41, 641)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.convolution_classes = FieldOfView


class Vibration(AnalyticalPSF):
    """Creates a wavelength independent kernel image."""

    required_keys = {"fwhm", "pixel_scale"}
    z_order: ClassVar[tuple[int, ...]] = (244, 744)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meta["width_n_fwhms"] = 4
        self.convolution_classes = ImagePlane

        check_keys(self.meta, self.required_keys, action="error")
        self.kernel = None

    def get_kernel(self, obj):
        if self.kernel is not None:
            return self.kernel

        from_currsys(self.meta, self.cmds)
        fwhm_pix = self.meta["fwhm"] / self.meta["pixel_scale"]
        sigma = fwhm_pix / 2.35
        width = max(1, int(fwhm_pix * self.meta["width_n_fwhms"]))
        self.kernel = Gaussian2DKernel(sigma, x_size=width, y_size=width,
                                       mode="center").array
        self.kernel /= np.sum(self.kernel)

        return self.kernel.astype(float)


class NonCommonPathAberration(AnalyticalPSF):
    """
    TBA.

    Needed: pixel_scale
    Accepted: kernel_width, strehl_drift
    """

    required_keys = {"pixel_scale"}
    z_order: ClassVar[tuple[int, ...]] = (241, 641)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.meta["kernel_width"] = None
        self.meta["strehl_drift"] = 0.02
        self.meta["wave_min"] = "!SIM.spectral.wave_min"
        self.meta["wave_max"] = "!SIM.spectral.wave_max"

        self._total_wfe = None

        self.valid_waverange = [0.1 * u.um, 0.2 * u.um]

        self.convolution_classes = FieldOfView
        check_keys(self.meta, self.required_keys, action="error")

    def get_kernel(self, obj):
        waves = obj.meta["wave_min"], obj.meta["wave_max"]

        old_waves = self.valid_waverange
        wave_mid_old = 0.5 * (old_waves[0] + old_waves[1])
        wave_mid_new = 0.5 * (waves[0] + waves[1])
        strehl_old = wfe2strehl(wfe=self.total_wfe, wave=wave_mid_old)
        strehl_new = wfe2strehl(wfe=self.total_wfe, wave=wave_mid_new)

        if np.abs(1 - strehl_old / strehl_new) > self.meta["strehl_drift"]:
            self.valid_waverange = waves
            self.kernel = wfe2gauss(wfe=self.total_wfe, wave=wave_mid_new,
                                    width=self.meta["kernel_width"])
            self.kernel /= np.sum(self.kernel)

        return self.kernel

    def _get_total_wfe_from_table(self):
        wfes = quantity_from_table("wfe_rms", self.table, "um")
        n_surfs = self.table["n_surfaces"]
        return np.sum(n_surfs * wfes**2)**0.5

    @property
    def total_wfe(self):
        if self._total_wfe is not None:
            return self._total_wfe

        if self.table is not None:
            self._total_wfe = self._get_total_wfe_from_table()
        else:
            self._total_wfe = 0

        return self._total_wfe

    def plot(self):
        fig, axes = figure_factory()

        wave_min, wave_max = from_currsys([self.meta["wave_min"],
                                           self.meta["wave_max"]], self.cmds)
        waves = np.linspace(wave_min, wave_max, 1001) * u.um
        wfe = self.total_wfe
        strehl = wfe2strehl(wfe=wfe, wave=waves)

        axes.plot(waves, strehl)
        axes.set_xlabel(f"Wavelength [{waves.unit}]")
        axes.set_ylabel(f"Strehl Ratio \n[Total WFE = {wfe}]")

        return fig


class SeeingPSF(AnalyticalPSF):
    """
    Currently only returns gaussian kernel with a ``fwhm`` [arcsec].

    Parameters
    ----------
    fwhm : flaot
        [arcsec]

    """

    z_order: ClassVar[tuple[int, ...]] = (242, 642)

    def __init__(self, fwhm=1.5, **kwargs):
        super().__init__(**kwargs)

        self.meta["fwhm"] = fwhm

    def get_kernel(self, fov):
        # called by .apply_to() from the base PSF class

        pixel_scale = fov.header["CDELT1"] * u.deg.to(u.arcsec)
        pixel_scale = quantify(pixel_scale, u.arcsec)

        # add in the conversion to fwhm from seeing and wavelength here
        fwhm = from_currsys(self.meta["fwhm"], self.cmds) * u.arcsec / pixel_scale

        sigma = fwhm.value / 2.35
        kernel = Gaussian2DKernel(sigma, mode="center").array
        kernel /= np.sum(kernel)

        return kernel

    def plot(self):
        pixel_scale = from_currsys("!INST.pixel_scale", self.cmds)
        spec_dict = from_currsys("!SIM.spectral", self.cmds)
        return super().plot(PoorMansFOV(pixel_scale, spec_dict))


class GaussianDiffractionPSF(AnalyticalPSF):
    z_order: ClassVar[tuple[int, ...]] = (242, 642)

    def __init__(self, diameter, **kwargs):
        super().__init__(**kwargs)
        self.meta["diameter"] = diameter

    def update(self, **kwargs):
        if "diameter" in kwargs:
            self.meta["diameter"] = kwargs["diameter"]

    def get_kernel(self, fov):
        # called by .apply_to() from the base PSF class

        pixel_scale = fov.header["CDELT1"] * u.deg.to(u.arcsec)
        pixel_scale = quantify(pixel_scale, u.arcsec)

        wave = 0.5 * (fov.meta["wave_max"] + fov.meta["wave_min"])

        wave = quantify(wave, u.um)
        diameter = quantify(self.meta["diameter"], u.m).to(u.um)
        fwhm = 1.22 * (wave / diameter) * u.rad.to(u.arcsec) / pixel_scale

        sigma = fwhm.value / 2.35
        kernel = Gaussian2DKernel(sigma, mode="center").array
        kernel /= np.sum(kernel)

        return kernel

    def plot(self):
        pixel_scale = from_currsys("!INST.pixel_scale", self.cmds)
        spec_dict = from_currsys("!SIM.spectral", self.cmds)
        return super().plot(PoorMansFOV(pixel_scale, spec_dict))


class MoffatPSF(AnalyticalPSF):
    """
    Moffat PSF with given FWHM (seeing), and alpha (also known as beta) parameters.

    Required kwargs:

    - alpha: Moffat alpha parameter
    - fwhm: Moffat FWHM (dict OR filename)

    Optional kwargs:

    - kernel_size: Minimum size of kernel in multiples of FWHM (int)
    - kernel_enclosed_energy: Target enclosed kernel flux. Defaults to
      ``1 - flux_accuracy``.
    - max_kernel_size: Maximum kernel width in pixels. Defaults to 501.
    - max_kernel_wavelength_samples: Maximum number of wavelengths used to
      choose the kernel size. Defaults to 16.
    - renormalize_clipped_kernel: If True, restore the historical behaviour of
      normalizing the clipped finite kernel to unity. Defaults to False.

    Examples
    --------

    FWHM as a function of wavelength through seeing law parameters. If "pivot_wave" is not supplied, FWHM is
    assumed to be wavelength independent and no scaling is applied.
    ::

        name: seeing_psf
        class: MoffatPSF
        kwargs:
            alpha: 4.765
            fwhm:
                seeing: 0.7
                seeing_unit: "arcsec"
                pivot_wave: 500
                pivot_wave_unit: "nm"


    * If "fwhm" is a float, it is assumed to be wavelength independent and in arcsec unit.
    * If "fwhm" is a string, it is assumed to be a .dat filename that contains a table with columns
      "wavelength" and "fwhm".

    """
    z_order: ClassVar[tuple[int, ...]] = (43, 643)
    required_keys = {"alpha", "fwhm"}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._fwhm_interp = None

    def fwhm(self, wavelengths):
        target, location, time, _ = get_observation_info_from_cmds(self.cmds)
        if isinstance(self.meta, dict):
            fwhm = self.meta["fwhm"]
            key = (fwhm["seeing"], fwhm["seeing_unit"], fwhm["pivot_wave"], fwhm["pivot_wave_unit"],
                   target, location, time)
        else:
            key = self.meta["fwhm"], target, location, time

        if self._fwhm_interp is None or self._fwhm_interp[1] != key:
            self._fwhm_interp = self.get_fwhm_interp(), key

        return self._fwhm_interp[0](wavelengths)

    @property
    def alpha(self):
        return self.meta["alpha"]

    def get_fwhm_interp(self):
        """
        Parses supplied FWHM input kwarg and returns a function that takes wavelength as input and returns FWHM.
        Overwrite this function for subclassing if FWHM input options change.
        """
        if isinstance(self.meta["fwhm"], dict):
            logger.info("dict supplied for FWHM")
            fwhm = self.meta["fwhm"]
            if check_keys(fwhm, {"seeing", "seeing_unit", "pivot_wave", "pivot_wave_unit"}, action="warn"):
                logger.info("seeing and pivot supplied, using natural scale seeing law")
                target, location, time, _ = get_observation_info_from_cmds(self.cmds)
                zenith_angle = get_zenith_angle(target, location, time)

                return partial(self.natural_scale, seeing=fwhm["seeing"]*u.Unit(fwhm["seeing_unit"]),
                                pivot=fwhm["pivot_wave"]*u.Unit(fwhm["pivot_wave_unit"]),
                                zenith_angle=zenith_angle*u.deg)

            elif check_keys(fwhm, {"seeing", "seeing_unit"}, action="error"):
                logger.info("only seeing supplied, FWHM is wavelength independent")
                return lambda wavelengths: np.full(wavelengths.shape, fwhm["seeing"], dtype=float) * u.Unit(fwhm["seeing_unit"])

        if isinstance(self.meta["fwhm"], (int, float)):
            logger.info("float value supplied, assuming arcsec")
            return lambda wavelengths: np.full(wavelengths.shape, self.meta["fwhm"], dtype=float) * u.arcsec

        if isinstance(self.meta["fwhm"], str):
            logger.info("filename supplied for FWHM")
            self.table = DataContainer(filename=find_file(from_currsys(self.meta["fwhm"], self.cmds))).table
            return lambda wavelengths: self.fwhm_from_table(self.table)(wavelengths) * u.arcsec

        raise TypeError("fwhm kwarg must be of type dict or float or str")

    def get_kernel(self, fov):
        pixel_scale = fov.header["CDELT1"] * u.deg.to(u.arcsec)
        pixel_scale = quantify(pixel_scale, u.arcsec)

        # Sample the wavelength-dependent seeing law with a bounded number of
        # representative points. Kernel sizing is driven by the broadest PSF
        # over the FOV, not by every spectral sample in the cube.
        wavelengths = self._sample_kernel_wavelengths(fov.waveset)
        fwhms = self.fwhm(wavelengths) / pixel_scale
        alpha = self.alpha
        gammas = np.asarray(self.fwhm2gamma(fwhms, alpha), dtype=float)

        target = self._target_enclosed_energy()
        max_ksize = self._max_kernel_size()
        ksize = self._minimum_kernel_size(np.max(fwhms))
        if max_ksize is not None:
            ksize = min(ksize, max_ksize)

        kernel, norm = self._make_moffat_kernel(gammas, alpha, ksize)
        while norm < target and (max_ksize is None or ksize < max_ksize):
            ksize = self._next_kernel_size(ksize, max_ksize)
            kernel, norm = self._make_moffat_kernel(gammas, alpha, ksize)

        if norm < target:
            logger.warning(
                "%s Moffat PSF kernel encloses %.6f of the analytic flux; "
                "target is %.6f. wave_range=(%.6g, %.6g) um, "
                "pixel_scale=%.6g arcsec, max_fwhm=%.6g pix, "
                "kernel_size=%d pix, max_kernel_size=%s.",
                self.display_name,
                norm,
                target,
                wavelengths[0].to_value(u.um),
                wavelengths[-1].to_value(u.um),
                pixel_scale.to_value(u.arcsec),
                np.max(fwhms),
                ksize,
                max_ksize,
            )

        if self._renormalize_clipped_kernel() and norm > 0:
            kernel /= norm
        return kernel

    def _sample_kernel_wavelengths(self, waveset: u.Quantity) -> u.Quantity:
        max_samples = max(2, int(from_currsys(
            self.meta.get("max_kernel_wavelength_samples", 16), self.cmds)))
        waveset = quantify(waveset, u.um)
        if waveset.size <= max_samples:
            return waveset

        indices = np.unique(np.linspace(0, waveset.size - 1, max_samples).round().astype(int))
        return waveset[indices]

    def _target_enclosed_energy(self) -> float:
        target = self.meta.get("kernel_enclosed_energy")
        if target is None:
            flux_accuracy = float(from_currsys(
                self.meta.get("flux_accuracy", 1e-3), self.cmds))
            target = 1.0 - flux_accuracy
        else:
            target = float(from_currsys(target, self.cmds))
        if not 0 < target <= 1:
            raise ValueError("kernel_enclosed_energy must be in the range (0, 1].")
        return target

    def _max_kernel_size(self) -> int | None:
        max_ksize = self.meta.get("max_kernel_size", 501)
        max_ksize = from_currsys(max_ksize, self.cmds)
        if max_ksize in (None, "None"):
            return None
        return self._ensure_odd_int(max_ksize)

    def _minimum_kernel_size(self, max_fwhm_pix: float) -> int:
        kx = float(from_currsys(self.meta.get("kernel_size", 4.0), self.cmds))
        return self._ensure_odd_int(kx * max_fwhm_pix)

    def _make_moffat_kernel(self, gammas: np.ndarray, alpha:float, ksize: int) -> tuple[np.ndarray, float]:
        amplitude = (alpha - 1) / (np.pi * gammas**2)
        x, y = np.meshgrid(
            np.arange(ksize) - ksize // 2,
            np.arange(ksize) - ksize // 2,
        )
        cube = Moffat2D.evaluate(
            x=x[None, ...],
            y=y[None, ...],
            amplitude=amplitude[:, None, None],
            x_0=0,
            y_0=0,
            gamma=gammas[:, None, None],
            alpha=alpha,
        )
        kernel = np.mean(cube, axis=0)
        if from_currsys(self.meta.get("rounded_edges", False), self.cmds):
            kernel = self._round_kernel_edges(kernel)
        return kernel, float(np.sum(kernel))

    @staticmethod
    def _ensure_odd_int(value) -> int:
        value = max(1, int(np.ceil(value)))
        return value + 1 if value % 2 == 0 else value

    def _next_kernel_size(self, ksize: int, max_ksize: int | None) -> int:
        next_size = self._ensure_odd_int(max(ksize + 2, int(np.ceil(1.25 * ksize))))
        if max_ksize is not None:
            next_size = min(next_size, max_ksize)
            next_size = next_size - 1 if next_size % 2 == 0 else next_size
        return max(next_size, ksize + 2)

    def _renormalize_clipped_kernel(self) -> bool:
        return bool(from_currsys(
            self.meta.get("renormalize_clipped_kernel", False), self.cmds))

    @staticmethod
    def natural_scale(wavelengths: u.Quantity,
                      seeing: u.Quantity = 0.7*u.arcsec, pivot: u.Quantity = 500*u.nm,
                      zenith_angle: u.Quantity = 0.0*u.deg) -> u.Quantity:
        """
        Seeing law scaled to wavelength

        https://opg.optica.org/josa/fulltext.cfm?uri=josa-68-7-877&id=57124
        https://www.mdpi.com/2072-4292/14/2/405
        """
        return seeing * (wavelengths.to_value(u.um) / pivot.to_value(u.um)) ** -0.2 * 1 / np.cos(zenith_angle.to(u.rad)) ** .6

    @staticmethod
    def fwhm2gamma(fwhm: u.Quantity, alpha) -> u.Quantity:
        """Convert FWHM to Moffat gamma parameter."""
        theta_factor = 0.5 / np.sqrt(2**(1./alpha) - 1.0)
        return fwhm * theta_factor

    @staticmethod
    def fwhm_from_table(table):
        if "wavelength" not in table.colnames or "fwhm" not in table.colnames:
            raise ValueError("Table must contain 'wavelength' and 'fwhm' columns.")
        wave_array = quantity_from_table("wavelength", table, "um").to_value(u.um)
        fwhm_array = table["fwhm"]
        return lambda wavelengths: make_interp_spline(wave_array, fwhm_array)(wavelengths.to(u.um).value)  # returns bspline instance


class AOEnhanceablePSF(MoffatPSF):
    """
    Wavelength dependent Moffat PSF but with added AO scaling.
    Required kwargs:

    - ao_table: A .dat file with AO scaling as a function of wavelength. Must have columns "wavelength" and "fwhm".
                Either supply 'alpha' parameter in table header or as kwarg. Table header value overrides kwarg.
    - is_absolute: bool, True if fwhm in ao_table are absolute values, False if they are scaling factors for "seeing".
    - enable_ao: bool, True to turn AO enhancement "on", False for regular seeing law Moffat

    Optional kwargs:

    - kernel_size: Size of kernel in multiples of FWHM (int)
    - alpha: Moffat alpha parameter
    - fwhm: Moffat FWHM (dict OR filename), not needed if is_absolute is True

    Examples
    --------

    FWHM as a function of wavelength through seeing law parameters. If "pivot_wave" is not supplied, FWHM is
    assumed to be wavelength independent and no scaling is applied.
    ::

        name: seeing_psf
        class: MoffatPSF
        kwargs:
            ao_table: "path/to/table.dat"
            is_absolute: True
            enable_ao: True
            alpha: 4.765
            fwhm:
                seeing: 0.7
                seeing_unit: "arcsec"
                pivot_wave: 500
                pivot_wave_unit: "nm"

    """
    z_order: ClassVar[tuple[int, ...]] = (43, 643)
    required_keys = {"ao_table", "is_absolute", "enable_ao"}

    def __init__(self, **kwargs):
        # setting alpha and fwhm before super().__init__ to avoid KeyError
        kwargs["alpha"] = None if "alpha" not in kwargs else kwargs["alpha"]
        kwargs["fwhm"] = 1.0 if "fwhm" not in kwargs else kwargs["fwhm"]
        super().__init__(**kwargs)
        self._ao_alpha = None
        self._ao_interp = None

    @property
    def natural_alpha(self):
        return super().alpha

    @property
    def ao_alpha(self):
        if self._ao_alpha is None or self._ao_alpha[1] != self.meta['ao_table']:
            _, alpha = self.ao_table_data()
            self._ao_alpha = alpha, self.meta['ao_table']
        return self._ao_alpha[0]

    @property
    def alpha(self):
        return self.ao_alpha if self.meta['enable_ao'] else self.natural_alpha

    def ao_table_data(self):
        aotab = DataContainer(filename=find_file(from_currsys(self.meta["ao_table"], self.cmds))).table

        if "alpha" in aotab.meta:
            alpha = aotab.meta["alpha"]
            logger.info(f"Alpha parameter found in AO table header: {alpha}")
        elif self.meta["alpha"] is not None:
            alpha = self.meta["alpha"]
            logger.info(f"Alpha parameter found in kwargs: {alpha}")
        else:
            raise ValueError("Alpha parameter missing: Not found in ao_table header or kwargs.")

        return aotab, alpha

    @property
    def ao_scale(self):
        if self._ao_interp is None or self._ao_interp[1] != self.meta['ao_table']:
            ao_table, _ = self.ao_table_data()
            self._ao_interp = self.fwhm_from_table(ao_table), self.meta['ao_table']
        return self._ao_interp[0]

    def ao_fwhm(self, wavelengths):
        ao = self.ao_scale(wavelengths)
        if not self.meta["is_absolute"]:
            ao *= self.natural_fwhm(wavelengths)
        return quantify(ao, u.arcsec)

    def natural_fwhm(self, wavelengths):
        return super().fwhm(wavelengths)

    def fwhm(self, wavelengths):
        return self.ao_fwhm(wavelengths) if self.meta["enable_ao"] else self.natural_fwhm(wavelengths)


def wfe2gauss(wfe, wave, width=None):
    strehl = wfe2strehl(wfe, wave)
    sigma = _strehl2sigma(strehl)
    if width is None:
        width = int(np.ceil(8 * sigma))
        width += (width + 1) % 2
    gauss = _sigma2gauss(sigma, x_size=width, y_size=width)

    return gauss


def wfe2strehl(wfe, wave):
    wave = quantify(wave, u.um)
    wfe = quantify(wfe, u.um)
    x = 2 * 3.1415926526 * wfe / wave
    strehl = np.exp(-x**2)
    return strehl


def _strehl2sigma(strehl):
    amplitudes = [0.00465, 0.00480, 0.00506, 0.00553, 0.00637, 0.00793,
                  0.01092, 0.01669, 0.02736, 0.04584, 0.07656, 0.12639,
                  0.20474, 0.32156, 0.48097, 0.66895, 0.84376, 0.95514,
                  0.99437, 0.99982, 0.99999]
    sigmas = [19.9526, 15.3108, 11.7489, 9.01571, 6.91830, 5.30884, 4.07380,
              3.12607, 2.39883, 1.84077, 1.41253, 1.08392, 0.83176, 0.63826,
              0.48977, 0.37583, 0.28840, 0.22130, 0.16982, 0.13031, 0.1]
    sigma = np.interp(strehl, amplitudes, sigmas)
    return sigma


def _sigma2gauss(sigma, x_size=15, y_size=15):
    kernel = Gaussian2DKernel(sigma, x_size=x_size, y_size=y_size,
                              mode="oversample").array
    kernel /= np.sum(kernel)
    return kernel
