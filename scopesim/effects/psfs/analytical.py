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

    - kernel_size: Size of kernel in multiples of FWHM (int)

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

        self.target, self.location, self.time = get_observation_info_from_cmds(self.cmds)
        self.alpha = self.meta["alpha"]
        self.fwhm = self.get_fwhm_interp()

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
                zenith_angle = get_zenith_angle(self.target, self.location, self.time)

                return partial(self.natural_scale, seeing=fwhm["seeing"]*u.Unit(fwhm["seeing_unit"]),
                                pivot=fwhm["pivot_wave"]*u.Unit(fwhm["pivot_wave_unit"]),
                                zenith_angle=zenith_angle*u.deg)

            elif check_keys(fwhm, {"seeing", "seeing_unit"}, action="error"):
                logger.info("only seeing supplied, FWHM is wavelength independent")
                return lambda wavelengths: fwhm["seeing"] * u.Unit(fwhm["seeing_unit"])

        if isinstance(self.meta["fwhm"], (int, float)):
            logger.info("float value supplied, assuming arcsec")
            return lambda wavelengths: self.meta["fwhm"] * u.arcsec

        if isinstance(self.meta["fwhm"], str):
            logger.info("filename supplied for FWHM")
            self.table = DataContainer(filename=find_file(from_currsys(self.meta["fwhm"], self.cmds))).table
            return self.fwhm_from_table(self.table)

        raise TypeError("fwhm kwarg must be of type dict or float or str")

    def get_kernel(self, fov):
        pixel_scale = fov.header["CDELT1"] * u.deg.to(u.arcsec)
        pixel_scale = quantify(pixel_scale, u.arcsec)

        # for each wavelength in waveset, get the corresponding FWHM, convert to gamma, and create a Moffat kernel
        npts = (fov.meta["wave_max"] - fov.meta["wave_min"]) / (from_currsys("!SIM.spectral.spectral_bin_width", self.cmds) * u.um)
        ##sample only npts from len(fov.waveset)
        wavelengths = fov.waveset[::max(1, int(len(fov.waveset) / npts))]
        ## get fwhm and gamma for the sampled wave pts
        fwhms = self.fwhm(wavelengths).to(u.arcsec) / pixel_scale
        gammas = self.fwhm2gamma(fwhms, self.alpha).value

        kx = self.meta.get("kernel_size", 4.0)
        ksize = int(kx * np.max(fwhms).value)
        ksize = ksize + 1 if ksize % 2 == 0 else ksize

        amplitude = (self.alpha - 1)/(np.pi * gammas**2)
        x, y = np.meshgrid(np.arange(ksize)-ksize//2, np.arange(ksize)-ksize//2)
        cube = Moffat2D.evaluate(x=x[None, ...], y=y[None, ...],
                                 amplitude=amplitude[:, None, None], x_0=0, y_0=0,
                                 gamma=gammas[:, None, None], alpha=self.alpha)
        kernel = np.mean(cube, axis=0) # average over wavelength axis
        norm = np.sum(kernel)
        if norm < 0.98:
            logger.warning(f"Kernel size too small, kernel sums to {norm}")
        kernel /= norm
        return kernel

    @staticmethod
    def natural_scale(wavelengths: u.Quantity,
                      seeing: u.Quantity = 0.7*u.arcsec, pivot: u.Quantity = 500*u.nm,
                      zenith_angle: u.Quantity = 0.0*u.deg) -> u.Quantity:
        """
        Seeing law scaled to wavelength

        https://opg.optica.org/josa/fulltext.cfm?uri=josa-68-7-877&id=57124
        https://www.mdpi.com/2072-4292/14/2/405
        """
        return seeing * (wavelengths / pivot) ** -0.2 * 1 / np.cos(zenith_angle.to(u.rad)) ** .6

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
        return make_interp_spline(wave_array, fwhm_array)  # returns bspline instance


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

        aotab = DataContainer(filename=find_file(from_currsys(self.meta["ao_table"], self.cmds))).table
        self.ao_scale = self.fwhm_from_table(aotab)

        if "alpha" in aotab.meta:
            self.alpha = aotab.meta["alpha"]
            logger.info(f"Alpha parameter found in AO table header: {self.alpha}")
        elif self.meta["alpha"] is not None:
            self.alpha = self.meta["alpha"]
            logger.info(f"Alpha parameter found in kwargs: {self.alpha}")
        else:
            raise ValueError("Alpha parameter missing: Not found in ao_table header or kwargs.")

        if self.meta["enable_ao"]:
            if self.meta["is_absolute"]:
                self.fwhm = self.ao_scale
            else:
                self.fwhm = lambda wavelengths: (self.fwhm(wavelengths) * self.ao_scale(wavelengths)).to(u.arcsec)


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
