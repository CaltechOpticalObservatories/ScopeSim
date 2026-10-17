import logging

import numpy as np
import pytest
from astropy import units as u

from scopesim.effects.psfs.analytical import MoffatPSF


def test_seeing_cache_uses_physical_zenith_angle(monkeypatch):
    from astropy.coordinates import SkyCoord, AltAz, EarthLocation
    from astropy.time import Time
    from scopesim.effects.psfs import analytical

    location = EarthLocation.from_geodetic(0*u.deg, 30*u.deg)
    altitude = 60*u.deg
    times = iter(Time(60310 + np.arange(10)/86400, format='mjd'))

    def observing_info(cmds):
        time = next(times)
        target = SkyCoord(alt=altitude, az=0*u.deg, frame=AltAz(obstime=time, location=location))
        return target, location, time, 'dark'

    monkeypatch.setattr(analytical, 'get_observation_info_from_cmds', observing_info)
    psf = MoffatPSF(alpha=4.765, fwhm={'seeing': .7, 'seeing_unit': 'arcsec', 'pivot_wave': 500, 'pivot_wave_unit': 'nm'})
    first = psf.fwhm(np.array([.5])*u.um)
    np.testing.assert_allclose(psf.fwhm(np.array([.5])*u.um), first)
    altitude = 30*u.deg
    expected = .7 / np.sin(altitude.to_value(u.rad))**.6
    assert psf.fwhm(np.array([.5])*u.um)[0].to_value(u.arcsec) == pytest.approx(expected)


class SimpleFov:
    header = {"CDELT1": 0.1 / 3600}
    meta = {"wave_min": 1.0 * u.um, "wave_max": 1.1 * u.um}
    waveset = np.linspace(1.0, 1.1, 5) * u.um


def test_constant_fwhm_does_not_require_observation_commands():
    psf = MoffatPSF(alpha=4.765, fwhm=0.7)

    assert isinstance(psf, MoffatPSF)


def test_kernel_grows_to_flux_accuracy_without_renormalizing():
    psf = MoffatPSF(
        alpha=4.765,
        fwhm=0.7,
        kernel_size=2,
        max_kernel_size=151,
        flux_accuracy=1e-3,
        rounded_edges=False,
    )

    kernel = psf.get_kernel(SimpleFov())

    assert kernel.shape[0] > 15
    assert np.sum(kernel) >= 0.999
    assert np.sum(kernel) < 1.0


def test_rounded_kernel_grows_to_flux_accuracy_after_rounding():
    psf = MoffatPSF(
        alpha=4.765,
        fwhm=0.7,
        kernel_size=2,
        max_kernel_size=151,
        flux_accuracy=1e-3,
        rounded_edges=True,
    )

    kernel = psf.get_kernel(SimpleFov())

    assert np.sum(kernel) >= 0.999
    assert np.sum(kernel) < 1.0


def test_capped_kernel_warns_and_keeps_missing_wing_flux(caplog):
    psf = MoffatPSF(
        alpha=4.765,
        fwhm=0.7,
        kernel_size=2,
        max_kernel_size=15,
        flux_accuracy=1e-6,
        rounded_edges=False,
    )

    with caplog.at_level(
        logging.WARNING,
        logger="astar.scopesim.effects.psfs.analytical",
    ):
        kernel = psf.get_kernel(SimpleFov())

    assert np.sum(kernel) < 1.0
    assert "Moffat PSF kernel encloses" in caplog.text
    assert "max_kernel_size=15" in caplog.text


def test_legacy_renormalization_can_be_requested():
    psf = MoffatPSF(
        alpha=4.765,
        fwhm=0.7,
        kernel_size=2,
        max_kernel_size=15,
        flux_accuracy=1e-6,
        rounded_edges=False,
        renormalize_clipped_kernel=True,
    )

    kernel = psf.get_kernel(SimpleFov())

    assert np.sum(kernel) == pytest.approx(1.0)
