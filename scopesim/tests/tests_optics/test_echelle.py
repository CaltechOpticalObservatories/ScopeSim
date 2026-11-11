"""Planar grating geometry checked through the camera's forward projection."""

import numpy as np
import pytest
from astropy import units as u

from scopesim.optics.echelle import GratingSetup, spectrograph_factory


# Wavelength endpoints [nm], cross-disperser density [mm^-1], detector height [pix].
CHANNELS = [(310, 430, 2000, 4096), (430, 650, 1100, 4096), (650, 980, 625, 4096),
            (960, 1340, 350, 2048), (1470, 1830, 300, 2048), (1990, 2450, 270, 2048)]


def test_asymmetric_pupil_slit_dispersion_and_anamorphism():
    instrument = spectrograph_factory(960, 1340, 225, 18000, 65.6, 58, 80, 1/24, 4.46,
                                     2048, 2048, .015, xdisp_groove_length=1/350, xdisp_beta_center=10.9,
                                     collimator1_focal_length=1500*u.mm, collimator2_focal_length=1250*u.mm, entrance_slit_scale=.5306*u.mm/u.arcsec)
    assert instrument.dispersion_focal_length == 270*u.mm
    center = instrument.central_wave(65)
    derivative = instrument.slit_dispersion(center, 65)
    assert derivative < 0*u.um/u.arcsec
    for slit, expected in ((.7, 17800), (.33, 37800)):
        resolution = (center / abs(derivative * slit*u.arcsec)).to_value(u.dimensionless_unscaled)
        assert resolution == pytest.approx(expected, rel=.01)
    width = instrument.projected_slit_width(center, 65, .7*u.arcsec)
    assert width.to_value(u.pixel) == pytest.approx(4.45704, rel=1e-5)
    wavelengths = center * np.array([.99, 1., 1.01])
    widths = instrument.projected_slit_width(wavelengths, 65, .7*u.arcsec)
    assert np.all(np.diff(widths) > 0*u.pixel)
    half_angle = (.35*u.arcsec * instrument.entrance_slit_scale / instrument.collimator1_focal_length).value * u.rad
    beta_low = np.arcsin(65*wavelengths/instrument.grating.d - np.sin(instrument.grating.alpha + half_angle))
    beta_high = np.arcsin(65*wavelengths/instrument.grating.d - np.sin(instrument.grating.alpha - half_angle))
    edge_width = abs(instrument.x_pixel_for_beta(beta_high) - instrument.x_pixel_for_beta(beta_low))
    # Independent finite slit edges retain the same anamorphism as dispersion.
    np.testing.assert_allclose(widths.to_value(u.pixel), edge_width.value, rtol=2e-7)
    step = center*1e-7
    dispersion = (instrument.wavelength_to_x_pixel(wavelengths+step, 65)-instrument.wavelength_to_x_pixel(wavelengths-step, 65))/(2*step)
    measured_delta = widths.to_value(u.pixel)/dispersion
    np.testing.assert_allclose(measured_delta.to_value(u.nm), abs(derivative*.7*u.arcsec).to_value(u.nm), rtol=2e-7)
    eta = .0825*u.arcsec
    dalpha = (eta * instrument.entrance_slit_scale / instrument.collimator1_focal_length).to_value(u.dimensionless_unscaled) * u.rad
    exact = instrument.grating.d / 65 * (np.sin(instrument.grating.alpha) - np.sin(instrument.grating.alpha + dalpha))
    assert (derivative * eta).to_value(u.nm) == pytest.approx(exact.to_value(u.nm), rel=1e-4)


def test_incomplete_physical_prescription_fails():
    with pytest.raises(AttributeError):
        spectrograph_factory(960, 1340, 225, 18000, 65.6, 58, 80, 1/24, 4.46, 2048, 2048, .015,
                             xdisp_groove_length=1/350, collimator1_focal_length=1500*u.mm)


@pytest.mark.parametrize("low,high,density,height", CHANNELS)
def test_fixed_groove_endpoints_fill_camera(low, high, density, height):
    instrument = spectrograph_factory(low, high, 225, 18000, 65.6, 40, 50, 1/24, 4.46,
                                     2048, height, .015, xdisp_groove_length=1/density, dispersion_focal_len=270)
    wavelength = np.array([low, high]) * u.nm
    pixels = instrument.wavelength_to_y_pixel(wavelength)
    # Algebraic inverse/forward evaluation; 1e-7 pixel allows floating-point cancellation.
    np.testing.assert_allclose(pixels.value, [0, height], atol=1e-7, rtol=0)
    recovered = instrument.y_pixel_to_wavelength(pixels)
    np.testing.assert_allclose(recovered.to_value(u.nm), wavelength.value, atol=1e-9, rtol=0)


def test_units_and_signed_incidence():
    numeric = GratingSetup.estimate_xdisp_angle_with_groove_len(1470, 1830, 30.72, 225, 1/300)
    physical = GratingSetup.estimate_xdisp_angle_with_groove_len(1.47*u.um, 1.83*u.um, 3.072*u.cm, .225*u.m, (1/300)*u.mm)
    assert physical[0] < 0 * u.rad
    for plain, quantity in zip(numeric, physical):
        assert plain.unit == u.rad
        assert quantity.unit == u.rad
        assert plain.to_value(u.rad) == pytest.approx(quantity.to_value(u.rad), abs=1e-12)


@pytest.mark.parametrize("inputs", [(310, 430, 1, 225, .0005), (3000, 3010, 61.44, 225, .0001),
                                  (430, 310, 61.44, 225, .0005), (310, 430, 0, 225, .0005)])
def test_impossible_geometry_fails(inputs):
    with pytest.raises(ValueError):
        GratingSetup.estimate_xdisp_angle_with_groove_len(*inputs)


def test_explicit_angle_keeps_configured_geometry():
    instrument = spectrograph_factory(960, 1340, 225, 18000, 65.6, 57, 80, 1/24, 4.46,
                                     2048, 2048, .015, xdisp_groove_length=1/350, xdisp_beta_center=12)
    assert instrument.cross_disperser.alpha.to_value(u.deg) == pytest.approx(12)
    assert instrument.cross_disperser.beta_center.to_value(u.deg) == pytest.approx(12)


def test_free_frequency_endpoints_fill_camera():
    instrument = spectrograph_factory(960, 1340, 225, 18000, 65.6, 57, 80, 1/24, 4.46,
                                     2048, 2048, .015, xdisp_groove_length=np.nan)
    pixels = instrument.wavelength_to_y_pixel(np.array([960, 1340]) * u.nm)
    np.testing.assert_allclose(pixels.value, [0, 2048], atol=1e-7, rtol=0)


def test_fsr_retains_the_central_pixel_wavelength_convention():
    instrument = spectrograph_factory(1990, 2450, 225, 18000, 65.6, 31, 38, 1/24, 4.46,
                                     2048, 2048, .015, xdisp_groove_length=1/270, dispersion_focal_len=270)
    edges = instrument.edge_wave(fsr=True)
    center = instrument.x_pixel_to_wavelength(instrument.detector.n_pix_x//2 + .5, instrument.orders)
    np.testing.assert_allclose(edges.mean(axis=1).to_value(u.nm), center.to_value(u.nm), rtol=1e-12)
    np.testing.assert_allclose(np.diff(edges, axis=1).ravel().to_value(u.nm), (center/instrument.orders).to_value(u.nm), rtol=1e-12)


class TestEchelleHygiene:
    """Defects that were unreachable from production but trivially wrong."""

    @staticmethod
    def _echelle():
        return GratingSetup(alpha=65.6*u.deg, beta_center=65.6*u.deg,
                            delta=65.6*u.deg, groove_length=u.mm/50.15)

    def test_blaze_accepts_a_scalar_angle(self):
        # k[k > 1] = 1 needed a writable array, so a scalar beta used to raise.
        grating = self._echelle()
        scalar = grating.blaze(np.deg2rad(65.)*u.rad, 100)
        array = grating.blaze(np.deg2rad(np.array([64., 65., 66.]))*u.rad, 100)
        assert np.ndim(scalar) == 0
        assert scalar == pytest.approx(array[1])

    def test_blaze_is_capped_at_unity(self):
        grating = self._echelle()
        values = grating.blaze(np.deg2rad(np.linspace(55., 75., 41))*u.rad, 100)
        assert np.all(values <= 1.)

    def test_str_works_for_a_vph_grating_without_a_blaze_facet(self):
        # delta is None for VPH, so np.rad2deg(delta) used to raise.
        vph = GratingSetup(grating_type="vph", alpha=.1*u.rad, beta_center=.1*u.rad,
                           groove_length=u.mm/2000, vph_center_wave=500*u.nm)
        assert "delta=n/a" in str(vph)

    def test_order_cache_refreshes_without_relying_on_assert(self):
        # The old guard used `assert`, which python -O strips, leaving the
        # cached order array permanently stale.
        model = spectrograph_factory(310, 430, 225, 18000, 65.6, 85, 118, 1/50.15,
                                     4.46, 2048, 4096, .015,
                                     xdisp_groove_length=1/2000, xdisp_beta_center=21.2)
        before = len(model.orders)
        model.m_max = model.m_max + 2
        assert len(model.orders) == before + 2

    def test_detector_class_does_not_shadow_the_core_detector(self):
        from scopesim.optics import echelle
        from scopesim.detector import Detector as CoreDetector
        assert echelle.EchelleDetector is not CoreDetector
        assert not hasattr(echelle, "Detector")

    def test_dead_diagnostics_are_gone(self):
        # average_res raised AttributeError on every call (detector.n_pixels
        # does not exist); neither it nor implied_design_res had any caller.
        from scopesim.optics.echelle import SpectrographSetup
        assert not hasattr(SpectrographSetup, "average_res")
        assert not hasattr(SpectrographSetup, "implied_design_res")
        assert not hasattr(SpectrographSetup, "plot_echellogram")
