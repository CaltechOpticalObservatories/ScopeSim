# -*- coding: utf-8 -*-
"""Contains simple Vibration, NCPA, Seeing and Diffraction PSFs."""

from typing import ClassVar
from functools import partial

import numpy as np
from astropy import units as u
from astropy.convolution import Gaussian2DKernel, Moffat2DKernel
from scipy.interpolate import make_interp_spline

from .. import DataContainer
from ...optics import ImagePlane
from ...optics.fov import FieldOfView
from ...utils import (from_currsys, quantify, quantity_from_table,
                      figure_factory, check_keys, get_logger, airmass2zendist,
                      get_target, get_location, get_zenith_angle)
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
    """ Moffat PSF with given FWHM (seeing), and alpha (also known as beta) parameters.

        FWHM (seeing) can be given
        - using the filename keyword to read it from a table. with columns "wavelength" and "fwhm"
        - a single value ("seeing")

        Filename overrides seeing, if both are given.
        If "seeing" and "pivot" are given, FWHM will be scaled to wavelengths using the seeing law (natural_scale()).
        If "enable_ao" is set to True, "ao_filename", "ao_alpha" will be used for MoffatPSF.

        :: Example config using filename for FWHM:
        name: seeing_psf
        class: MoffatPSF
        kwargs:
            filename: path/to/fwhm_table.dat
            alpha: 4.765
            enable_ao: False
            ao_filename: path/to/ao_fwhm_table.dat
            ao_alpha: 3.25

        :: Example config using seeing value:
        name: seeing_psf
        class: MoffatPSF
        kwargs:
            seeing: 0.7
            seeing_unit: "arcsec"
            pivot_wave: 500
            pivot_wave_unit: "nm"
            alpha: 4.765
            enable_ao: False
            ao_alpha = 3.25

    """
    z_order: ClassVar[tuple[int, ...]] = (43, 643)
    required_keys = ["alpha"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.meta.get("enable_ao", False):
            logger.info("AO enabled, using AO parameters for FWHM.")
            check_keys(self.meta, {"ao_alpha"}, action="error")
            if self.meta.get("ao_filename", None):
                logger.info("filename given for AO FWHM")
                _file = DataContainer(filename=self.meta["ao_filename"])
                self.fwhm = self.fwhm_from_table(_file.get_data())
            else:
                logger.info("using in-built AO scale for FWHM")
                self.fwhm = self.ao_scale
            self.alpha = self.meta["ao_alpha"]

        else:
            if self.meta.get("filename", None):
                logger.info("filename given for FWHM")
                self.fwhm = self.fwhm_from_table(self.table)
            elif check_keys(self.meta, {"seeing", "seeing_unit", "pivot_wave", "pivot_wave_unit"}, action="warn"):
                logger.info("using seeing and natural scale for FWHM")
                self.fwhm = partial(self.natural_scale, seeing=self.meta["seeing"]*u.Unit(self.meta["seeing_unit"]),
                                    pivot=self.meta["pivot_wave"]*u.Unit(self.meta["pivot_wave_unit"]),
                                    zenith_angle=self.zenith_angle*u.deg)
            elif check_keys(self.meta, {"seeing", "seeing_unit"}, action="error"):
                logger.info("using seeing only (wavelength independent) for FWHM")
                self.fwhm = lambda wavelengths: self.meta["seeing"] * u.Unit(self.meta["seeing_unit"])
            self.alpha = self.meta["alpha"]

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
        ksize = int(4.0*np.max(fwhms).value)
        ksize = ksize + 1 if ksize % 2 == 0 else ksize
        kernel = np.zeros((ksize, ksize))
        for gamma in gammas:
            kernel += Moffat2DKernel(gamma=gamma, alpha=self.alpha, x_size=ksize, y_size=ksize).array
        kernel /= len(gammas)
        kernel /= np.sum(kernel)
        return kernel

    @staticmethod
    def fwhm_from_table(table):
        if "wavelength" not in table.colnames or "fwhm" not in table.colnames:
            raise ValueError("Table must contain 'wavelength' and 'fwhm' columns.")
        wave_array = quantity_from_table("wavelength", table, u.um)
        fwhm_array = quantity_from_table("fwhm", table, u.arcsec)
        return make_interp_spline(wave_array, fwhm_array)  # returns bspline instance

    @property
    def zenith_angle(self):
        if check_keys(self.cmds, {"!OBS.alt"}):
            logger.info("Using !OBS.alt to determine zenith angle.")
            return 90 - from_currsys("!OBS.alt", self.cmds)
        elif check_keys(self.cmds, {"!OBS.ra", "!OBS.dec", "!ATMO.longitude", "!ATMO.latitude", "!ATMO.altitude", "!OBS.mjdobs"}):
            logger.info("Using !OBS.ra, !OBS.dec, and !ATMO location info to determine zenith angle.")
            target = get_target(ra=from_currsys("!OBS.ra", self.cmds),
                                dec=from_currsys("!OBS.dec", self.cmds))
            location = get_location(lon=from_currsys("!ATMO.longitude", self.cmds),
                                    lat=from_currsys("!ATMO.latitude", self.cmds),
                                    alt=from_currsys("!ATMO.altitude", self.cmds))
            obstime = from_currsys("!OBS.mjdobs", self.cmds)
            return get_zenith_angle(target, location, obstime)
        elif check_keys(self.cmds, {"!OBS.airmass"}):
            logger.info("Using !OBS.airmass to determine zenith angle.")
            return airmass2zendist(from_currsys("!OBS.airmass", self.cmds))
        else:
            logger.warning("No valid input found to determine zenith angle. Defaulting to z=0 (zenith).")
            return 0.0

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
    def ao_scale(wavelengths: u.Quantity) -> u.Quantity:
        SKYBANDS = (np.array([300, 400]) * u.nm,
                    np.array([395, 505]) * u.nm,
                    np.array([495, 635]) * u.nm,
                    np.array([625, 800]) * u.nm,
                    np.array([785, 1000]) * u.nm,
                    np.array([990, 1185]) * u.nm,
                    np.array([1165, 1400]) * u.nm,
                    np.array([1500, 1800]) * u.nm,
                    np.array([2000, 2600]) * u.nm)
        fwhm_ao = lambda x, *a, **kw: make_interp_spline(
            [300] + list(map(lambda x: np.mean(x).value, SKYBANDS))[1:][:-1] + [2550],
            [343, 357, 220, 190, 185, 183, 182, 175, 182], k=1)(x.to(u.nm).value) * u.mas
        return fwhm_ao(wavelengths)

    @staticmethod
    def fwhm2gamma(fwhm: u.Quantity, alpha) -> u.Quantity:
        """Convert FWHM to Moffat gamma parameter."""
        theta_factor = 0.5 / np.sqrt(2**(1./alpha) - 1.0)
        return fwhm * theta_factor


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
