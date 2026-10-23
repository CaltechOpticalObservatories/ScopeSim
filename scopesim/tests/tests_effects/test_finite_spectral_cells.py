"""Photon integrals and line moments across finite slit and wavelength cells."""
import numpy as np
import pytest
from scopesim.effects.spectral_trace_list_utils import _slit_cell_operator


@pytest.mark.parametrize('phase', [0., .25, .5, .75])
@pytest.mark.parametrize('slope', [0., .02])
def test_slit_remap_preserves_photons_and_illumination_centroid(phase, slope):
    wave = np.linspace(.99, 1.01, 2001)
    derivative = -.001 + slope * (wave - 1.)
    step = wave[1] - wave[0]
    eta, deta = .0825 + phase*step/abs(derivative[1000]), .03
    photons = np.zeros(len(wave))
    photons[1000] = 17.
    output = _slit_cell_operator(wave, derivative, eta, deta) @ photons
    assert output.sum() == pytest.approx(17., rel=1e-12)
    center = np.sum(output * wave) / output.sum()
    assert abs(center - (1. + eta*derivative[1000])) < .01*step


def test_zero_slit_dispersion_is_identity():
    wave = np.linspace(1., 1.1, 101)
    photons = np.arange(len(wave)) + 1.
    actual = _slit_cell_operator(wave, np.zeros_like(wave), .1, .1) @ photons
    np.testing.assert_allclose(actual, photons, rtol=1e-12)


def test_finite_slit_cell_width_has_uniform_variance():
    wave = np.linspace(.99, 1.01, 10001)
    derivative = np.full_like(wave, -.001)
    photons = np.zeros(len(wave))
    photons[5000] = 1.
    output = _slit_cell_operator(wave, derivative, 0., .1) @ photons
    variance = np.sum(output * (wave - 1.)**2)
    assert variance == pytest.approx((.001*.1)**2 / 12, rel=.002)
