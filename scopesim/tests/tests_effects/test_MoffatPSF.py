import logging
from unittest.mock import Mock

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.table import Table
from astropy.time import Time

from scopesim.effects.psfs.analytical import AOEnhanceablePSF, MoffatPSF


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


@pytest.fixture
def seeing_law():
    return {"seeing": 0.7, "seeing_unit": "arcsec", "pivot_wave": 500, "pivot_wave_unit": "nm"}


@pytest.fixture
def observing_conditions(monkeypatch):
    location = EarthLocation.from_geodetic(-155 * u.deg, 20 * u.deg, 4000 * u.m)
    conditions = {"time": Time("2026-01-01T00:00:00"), "alt": 60 * u.deg}

    def observation_info(cmds):
        time = conditions["time"]
        target = SkyCoord(alt=conditions["alt"], az=0 * u.deg, frame="altaz", location=location, obstime=time)
        return target, location, time, None

    monkeypatch.setattr("scopesim.effects.psfs.analytical.get_observation_info_from_cmds", observation_info)
    return conditions


@pytest.mark.parametrize("psf_class", [MoffatPSF, AOEnhanceablePSF])
def test_observation_configuration_is_only_required_when_evaluating(psf_class, seeing_law, monkeypatch):
    observation_info = Mock(side_effect=KeyError("!OBS.airmass"))
    monkeypatch.setattr("scopesim.effects.psfs.analytical.get_observation_info_from_cmds", observation_info)
    kwargs = {"alpha": 4.765, "fwhm": seeing_law}
    if psf_class is AOEnhanceablePSF:
        kwargs.update(ao_table="unused.dat", enable_ao=False, is_absolute=False)
    psf = psf_class(**kwargs)
    observation_info.assert_not_called()
    with pytest.raises(KeyError, match="airmass"):
        psf.fwhm(np.array([0.5]) * u.um)


@pytest.mark.parametrize("fwhm", [0.7, {"seeing": 700, "seeing_unit": "mas"}])
def test_constant_seeing_is_evaluated_without_observation_commands(fwhm, monkeypatch):
    observation_info = Mock(side_effect=KeyError("!OBS.airmass"))
    monkeypatch.setattr("scopesim.effects.psfs.analytical.get_observation_info_from_cmds", observation_info)
    psf = MoffatPSF(alpha=4.765, fwhm=fwhm)
    np.testing.assert_allclose(psf.fwhm(np.array([0.5, 1.0]) * u.um).to_value(u.arcsec), [0.7, 0.7])
    observation_info.assert_not_called()


def test_changing_observation_times_reuses_same_seeing_law(seeing_law, observing_conditions):
    psf = MoffatPSF(alpha=4.765, fwhm=seeing_law)
    wavelengths = np.array([0.5, 1.0]) * u.um
    first = psf.fwhm(wavelengths)
    cached_interp = psf._fwhm_interp[0]
    observing_conditions["time"] = observing_conditions["time"] + 1 * u.hour
    second = psf.fwhm(wavelengths)
    np.testing.assert_allclose(second.to_value(u.arcsec), first.to_value(u.arcsec))
    assert psf._fwhm_interp[0] is cached_interp


def test_zenith_angle_change_updates_seeing_law(seeing_law, observing_conditions):
    psf = MoffatPSF(alpha=4.765, fwhm=seeing_law)
    wavelengths = np.array([0.5]) * u.um
    first = psf.fwhm(wavelengths)
    cached_interp = psf._fwhm_interp[0]
    observing_conditions["alt"] = 30 * u.deg
    second = psf.fwhm(wavelengths)
    expected_ratio = (np.cos(30 * u.deg) / np.cos(60 * u.deg))**0.6
    np.testing.assert_allclose(second.to_value(u.arcsec) / first.to_value(u.arcsec), expected_ratio)
    assert psf._fwhm_interp[0] is not cached_interp


@pytest.mark.parametrize("parameter, value, expected_ratio", [("seeing", 0.9, 0.9 / 0.7), ("pivot_wave", 1000, 2**0.2)])
def test_seeing_parameters_update_cached_law(parameter, value, expected_ratio, seeing_law, observing_conditions):
    psf = MoffatPSF(alpha=4.765, fwhm=seeing_law)
    wavelengths = np.array([1.0]) * u.um
    first = psf.fwhm(wavelengths)
    psf.meta["fwhm"][parameter] = value
    second = psf.fwhm(wavelengths)
    np.testing.assert_allclose(second.to_value(u.arcsec) / first.to_value(u.arcsec), expected_ratio)


def test_tabulated_fwhm_without_observation_commands(tmp_path, monkeypatch):
    observation_info = Mock(side_effect=KeyError("!OBS.airmass"))
    monkeypatch.setattr("scopesim.effects.psfs.analytical.get_observation_info_from_cmds", observation_info)
    filename = tmp_path / "fwhm.ecsv"
    table = Table({"wavelength": np.array([0.5, 1.0, 1.5, 2.0]) * u.um, "fwhm": [0.5, 0.6, 0.7, 0.8]})
    table.write(filename, format="ascii.ecsv")
    psf = MoffatPSF(alpha=4.765, fwhm=str(filename))
    np.testing.assert_allclose(psf.fwhm(np.array([0.5, 1.0]) * u.um).to_value(u.arcsec), [0.5, 0.6])
    observation_info.assert_not_called()


def test_ao_toggle_preserves_natural_and_absolute_widths(tmp_path):
    filename = tmp_path / "ao.ecsv"
    table = Table({"wavelength": np.array([0.5, 1.0, 1.5, 2.0]) * u.um, "fwhm": [0.5, 0.5, 0.5, 0.5]})
    table.write(filename, format="ascii.ecsv")
    psf = AOEnhanceablePSF(alpha=4.765, fwhm=0.7, ao_table=str(filename), enable_ao=False, is_absolute=False)
    wavelengths = np.array([1.0]) * u.um
    np.testing.assert_allclose(psf.fwhm(wavelengths).to_value(u.arcsec), [0.7])
    psf.meta["enable_ao"] = True
    np.testing.assert_allclose(psf.fwhm(wavelengths).to_value(u.arcsec), [0.35])
    psf.meta["is_absolute"] = True
    np.testing.assert_allclose(psf.fwhm(wavelengths).to_value(u.arcsec), [0.5])
    psf.meta["fwhm"] = 0.9
    psf.meta["enable_ao"] = False
    np.testing.assert_allclose(psf.fwhm(wavelengths).to_value(u.arcsec), [0.9])
