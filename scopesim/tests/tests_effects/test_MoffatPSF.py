import logging

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.table import Table
from astropy.time import Time
from astropy.utils import iers

from scopesim.commands import UserCommands
from scopesim.effects.psfs.analytical import AOEnhanceablePSF, MoffatPSF


class SimpleFov:
    header = {"CDELT1": 0.1 / 3600}
    meta = {"wave_min": 1.0 * u.um, "wave_max": 1.1 * u.um}
    waveset = np.linspace(1.0, 1.1, 5) * u.um


def test_constant_fwhm_does_not_require_observation_commands():
    psf = MoffatPSF(alpha=4.765, fwhm=0.7)

    wave = np.array([500., 1000.])*u.nm
    np.testing.assert_allclose(psf.fwhm(wave).to_value(u.arcsec), [.7, .7])
    cached = psf._fwhm_interp[0]
    np.testing.assert_allclose(psf.fwhm(wave.to(u.um)).to_value(u.arcsec), [.7, .7])
    assert psf._fwhm_interp[0] is cached


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


@pytest.fixture
def seeing_cmds(monkeypatch):
    monkeypatch.setattr(iers.conf, 'auto_download', False)
    # The regression uses historical timestamps, independent of table freshness.
    monkeypatch.setattr(iers.conf, 'auto_max_age', None)
    monkeypatch.setattr(Time, 'now', lambda: Time('2026-10-18T10:50:25.709309'))
    return UserCommands(properties={
        '!ATMO.longitude': -155.47833, '!ATMO.latitude': 19.82833, '!ATMO.altitude': 4160.,
        '!OBS.alt': 65.38002267134289, '!OBS.az': 0., '!OBS.airmass': None,
        '!OBS.ra': None, '!OBS.dec': None, '!OBS.brightness': 'dark',
    })


@pytest.fixture
def natural_psf(seeing_cmds):
    return MoffatPSF(cmds=seeing_cmds, alpha=4.765,
                     fwhm={'seeing': .65, 'seeing_unit': 'arcsec', 'pivot_wave': 500., 'pivot_wave_unit': 'nm'})


def test_nighttime_timestamp_change_reuses_the_same_seeing(natural_psf, monkeypatch):
    wave = np.array([500., 1000.])*u.nm
    first = natural_psf.fwhm(wave)
    cached = natural_psf._fwhm_interp[0]
    # These are the two times in the performance notebook's original traceback.
    monkeypatch.setattr(Time, 'now', lambda: Time('2026-10-18T10:50:25.735796'))
    second = natural_psf.fwhm(wave.to(u.um))
    np.testing.assert_allclose(first.to_value(u.arcsec), [.68825435, .59916022], atol=1e-8, rtol=0)
    np.testing.assert_array_equal(second, first)
    assert natural_psf._fwhm_interp[0] is cached


def test_seeing_and_pivot_changes_refresh_the_cache(natural_psf):
    wave = np.array([500., 1000.])*u.nm
    first = natural_psf.fwhm(wave)
    natural_psf.meta['fwhm']['seeing'] = 1.3
    np.testing.assert_allclose(natural_psf.fwhm(wave).to_value(u.arcsec), 2*first.to_value(u.arcsec))
    natural_psf.meta['fwhm']['pivot_wave'] = 1000.
    np.testing.assert_allclose(natural_psf.fwhm(wave).to_value(u.arcsec), 2*2**.2*first.to_value(u.arcsec))


def test_altitude_change_refreshes_the_seeing(natural_psf):
    wave = np.array([500.])*u.nm
    natural_psf.cmds['!OBS.alt'] = 90.
    np.testing.assert_allclose(natural_psf.fwhm(wave).to_value(u.arcsec), [.65])
    natural_psf.cmds['!OBS.alt'] = 30.
    np.testing.assert_allclose(natural_psf.fwhm(wave).to_value(u.arcsec), [.65*2**.6])


def test_radec_observation_time_changes_the_seeing(natural_psf):
    natural_psf.cmds['!OBS.ra'], natural_psf.cmds['!OBS.dec'] = 30., 20.
    target = SkyCoord(ra=30*u.deg, dec=20*u.deg, distance=1*u.AU)
    location = EarthLocation.from_geodetic(-155.47833*u.deg, 19.82833*u.deg, 4160*u.m)
    widths = []
    for time in ['2026-10-18T10:50:25', '2026-10-18T12:50:25']:
        natural_psf.cmds['!OBS.mjdobs'] = time
        alt = target.transform_to(AltAz(obstime=Time(time), location=location)).alt
        assert alt > 0*u.deg
        width = natural_psf.fwhm(np.array([500.])*u.nm).to_value(u.arcsec)[0]
        np.testing.assert_allclose(width, .65/np.sin(alt)**.6)
        widths.append(width)
    assert not np.isclose(widths[0], widths[1])


def test_relative_ao_uses_the_same_natural_seeing_cache(seeing_cmds, tmp_path, monkeypatch):
    path = tmp_path / 'ao.dat'
    Table({'wavelength': [.4, .5, 1., 2.5], 'fwhm': [.5, .5, .5, .5]}, meta={'comments': ['wavelength_unit : um']}).write(path, format='ascii')
    psf = AOEnhanceablePSF(cmds=seeing_cmds, ao_table=str(path), is_absolute=False, enable_ao=True, alpha=4.765,
                          fwhm={'seeing': .65, 'seeing_unit': 'arcsec', 'pivot_wave': 500., 'pivot_wave_unit': 'nm'})
    wave = np.array([500., 1000.])*u.nm
    first = psf.fwhm(wave)
    cached = psf._fwhm_interp[0]
    monkeypatch.setattr(Time, 'now', lambda: Time('2026-10-18T10:50:25.735796'))
    np.testing.assert_array_equal(psf.fwhm(wave), first)
    assert psf._fwhm_interp[0] is cached
    psf.meta['enable_ao'] = False
    np.testing.assert_allclose(psf.fwhm(wave).to_value(u.arcsec), 2*first.to_value(u.arcsec))


def test_table_fwhm_reuses_interpolator_without_observation_commands(tmp_path):
    path = tmp_path / 'fwhm.dat'
    Table({'wavelength': [.4, .5, 1., 2.5], 'fwhm': [.4, .5, 1., 2.5]}, meta={'comments': ['wavelength_unit : um']}).write(path, format='ascii')
    psf = MoffatPSF(alpha=4.765, fwhm=str(path))
    wave = np.array([500., 1000.])*u.nm
    np.testing.assert_allclose(psf.fwhm(wave).to_value(u.arcsec), [.5, 1.])
    cached = psf._fwhm_interp[0]
    np.testing.assert_allclose(psf.fwhm(wave.to(u.um)).to_value(u.arcsec), [.5, 1.])
    assert psf._fwhm_interp[0] is cached
