"""Tests for image-plane illumination effects."""

import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table
from synphot.units import PHOTLAM

from scopesim.effects.illumination import (
    ImagePlaneBackground,
    Illumination,
    PostDisperserDiffuseBackground,
    effective_diffuse_qe,
    gaussian2d,
    integrate_spectral_background,
    post_disperser_diffuse_spectrum,
    quadratic_vignetting,
)
from scopesim.optics.image_plane import ImagePlane
from scopesim.tests.mocks.py_objects.imagehdu_objects import _image_hdu_square
from scopesim.utils import pixel_area


@pytest.fixture
def imageplane():
    ip = ImagePlane(_image_hdu_square().header)
    ip.hdu.data = np.ones((100, 100), dtype=np.float64)
    return ip


def test_gaussian2d_peak_at_centre():
    result = np.asarray(gaussian2d((100, 100)))
    assert result.max() == pytest.approx(1.0)


def test_gaussian2d_values_leq_amp():
    result = gaussian2d((101, 101))
    assert result.max() <= 1.0 + 1e-12


def test_quadratic_vignetting_centre_is_one():
    result = quadratic_vignetting((101, 101))
    assert result[50, 50] == pytest.approx(1.0)


def test_quadratic_vignetting_values_in_range():
    result = quadratic_vignetting((101, 101))
    assert np.all(result >= 0.0) and np.all(result <= 1.0)


def test_illumination_instantiates():
    assert isinstance(Illumination(), Illumination)


def test_illumination_apply_to_returns_imageplane(imageplane):
    eff = Illumination()
    assert eff.apply_to(imageplane) is imageplane


def test_illumination_apply_to_skips_non_imageplane():
    eff = Illumination()
    obj = object()
    assert eff.apply_to(obj) is obj


def test_illumination_modifies_data(imageplane):
    original = imageplane.hdu.data.copy()
    Illumination().apply_to(imageplane)
    assert not np.array_equal(imageplane.hdu.data, original)


def test_illumination_caches_map(imageplane):
    eff = Illumination()
    eff.apply_to(imageplane)
    assert eff._map is not None and eff._map_shape == (100, 100)


def test_illumination_make_map_shape_and_dtype():
    eff = Illumination()
    illumination_map = eff._make_map((80, 60))
    assert illumination_map.shape == (80, 60)
    assert illumination_map.dtype == np.float32


def test_illumination_plot_raises_before_apply():
    with pytest.raises(RuntimeError):
        Illumination().plot()


def test_image_plane_background_adds_constant(imageplane):
    ImagePlaneBackground(value=2.5).apply_to(imageplane)
    assert np.all(imageplane.hdu.data == 3.5)


def test_image_plane_background_uses_model(imageplane):
    def model(shape, value):
        return np.full(shape, value)

    ImagePlaneBackground(model=model, modelargs={"value": 4}).apply_to(imageplane)
    assert np.all(imageplane.hdu.data == 5)


def test_image_plane_background_skips_non_imageplane():
    eff = ImagePlaneBackground(value=1)
    obj = object()
    assert eff.apply_to(obj) is obj


def test_image_plane_background_plot_raises_before_apply():
    with pytest.raises(RuntimeError):
        ImagePlaneBackground(value=1).plot()


class ConstantCurve:
    def __init__(self, value):
        self.value = value

    def __call__(self, wave):
        return np.full(wave.size, self.value)


class ConstantEmission:
    def __init__(self, value):
        self.value = value

    def __call__(self, wave):
        return np.full(wave.size, self.value) * PHOTLAM


class FakeSurface:
    def __init__(self, transmission, emission):
        self.transmission = ConstantCurve(transmission)
        self.emission = ConstantEmission(emission)


class FakeSurfaceList:
    table = Table({
        "name": ["pre", "camera"],
        "action": ["transmission", "transmission"],
        "emission_phase": ["pre_disperser", "post_disperser"],
    })

    def __init__(self):
        self.surfaces = {
            "pre": FakeSurface(transmission=0.5, emission=100.0),
            "camera": FakeSurface(transmission=0.8, emission=2.0),
        }


class FakeQE:
    throughput = ConstantCurve(0.5)


def test_effective_diffuse_qe_uses_average_positional_response():
    wave = np.linspace(1.0, 2.0, 3) * u.um
    positional_qe = np.array([[0.8, 1.0], [0.6, 1.0]])

    qe = effective_diffuse_qe(FakeQE(), wave, positional_qe=positional_qe)

    np.testing.assert_allclose(qe, np.full(wave.size, 0.425))


def test_post_disperser_diffuse_spectrum_uses_phase_metadata_and_qe():
    wave = np.linspace(1.0, 2.0, 3) * u.um
    qe = np.full(wave.size, 0.5)

    spectrum = post_disperser_diffuse_spectrum(
        FakeSurfaceList(), wave, qe_values=qe,
    )

    np.testing.assert_allclose(spectrum.value, np.full(wave.size, 1.0))


def test_integrate_spectral_background_returns_image_plane_rate():
    wave = np.array([1.0, 2.0]) * u.um
    spectrum = np.full(wave.size, 1.0) * PHOTLAM

    rate = integrate_spectral_background(
        spectrum,
        wave,
        telescope_area=1.0 * u.m**2,
        image_pixel_area=0.01 * u.arcsec**2,
    )

    assert rate == pytest.approx(1.0e6)


def test_post_disperser_diffuse_background_adds_integrated_rate(imageplane):
    eff = PostDisperserDiffuseBackground(
        surface_list=FakeSurfaceList(),
        detector_qe=FakeQE(),
        wave_min=1.0,
        wave_max=2.0,
        wave_bin=1.0,
        wave_unit="um",
        area=1.0 * u.m**2,
    )

    eff.apply_to(imageplane)

    expected = 1.0 + 1.0e8 * pixel_area(imageplane.header).to_value(u.arcsec**2)
    assert imageplane.hdu.data[0, 0] == pytest.approx(expected)
