import numpy as np
import pytest
from astropy import units as u
from synphot import SourceSpectrum, Empirical1D
from synphot.units import PHOTLAM
from scopesim.optics.fov import average_spectrum, spectral_bin_edges, rebin_spectral_density
from scopesim.optics.optical_train import OpticalTrain
from scopesim.source.source import Source
from scopesim.source.source_templates import ab_spectrum


@pytest.mark.parametrize('resolving_power', [20000, 200000])
@pytest.mark.parametrize('phase', [0, .2, .5, .8])
def test_narrow_line_photons_survive_coarse_grid_phase(resolving_power, phase):
    center = 5000 + phase*.1
    sigma = center / resolving_power / np.sqrt(8*np.log(2))
    native = center + np.linspace(-8, 8, 193)*sigma
    flux = np.exp(-.5*((native - center)/sigma)**2) / (sigma*np.sqrt(2*np.pi))
    spectrum = SourceSpectrum(Empirical1D, points=native*u.AA, lookup_table=flux*PHOTLAM)
    wave = np.arange(4998, 5002, .1)*u.AA
    density = average_spectrum(spectrum, wave)
    photons = (density * np.diff(spectral_bin_edges(wave.to_value(u.AA)))*u.AA).sum()
    assert photons.to_value(u.ph/u.s/u.cm**2) == pytest.approx(1, rel=.001)


def test_flat_ab_continuum_bin_integral():
    spectrum = ab_spectrum(mag=20)
    wave = np.linspace(5000, 5100, 101)*u.AA
    edges = spectral_bin_edges(wave.to_value(u.AA))
    # Constant f_nu gives photon density proportional to 1/lambda.
    scale = spectrum(5000*u.AA).to_value(PHOTLAM)*5000
    expected = scale*np.log(edges[1:]/edges[:-1])/np.diff(edges)
    np.testing.assert_allclose(average_spectrum(spectrum, wave).to_value(PHOTLAM), expected, rtol=1e-6)


def test_mutating_spectrum_redshift_updates_native_sampling():
    points = np.array([4999., 4999.99, 5000., 5000.01, 5001.])*u.AA
    spectrum = SourceSpectrum(Empirical1D, points=points, lookup_table=[0., 0., 1., 0., 0.])
    average_spectrum(spectrum, np.arange(4998., 5002., .1)*u.AA)
    spectrum.z = .01
    fresh = SourceSpectrum(Empirical1D, points=points, lookup_table=[0., 0., 1., 0., 0.], z=.01)
    wave = np.arange(5048., 5052., .1)*u.AA
    actual = average_spectrum(spectrum, wave)
    np.testing.assert_allclose(actual, average_spectrum(fresh, wave), rtol=1e-12)
    assert actual.sum() > 0


@pytest.mark.parametrize('descending', [False, True])
def test_cube_rebinning_preserves_flux_spatial_weights_and_support(descending):
    wave = np.array([1., 2., 3.])
    weights = np.array([[1., 2.], [3., 4.]])
    density = wave[:, None, None]*weights
    if descending:
        wave, density = wave[::-1], density[::-1]
    rebinned = rebin_spectral_density(wave, density, np.array([.5, 1.5, 2.5, 3.5]))
    np.testing.assert_allclose(rebinned, np.array([.625, 2., 1.375])[:, None, None]*weights)
    np.testing.assert_allclose(rebinned.sum(axis=0), 4*weights)


def test_prepare_source_keeps_native_line_sampling():
    wave = np.linspace(4999.9, 5000.1, 101)*u.AA
    spectrum = SourceSpectrum(Empirical1D, points=wave, lookup_table=np.exp(-((wave.value - 5000)/.01)**2)*PHOTLAM)
    source = Source(x=[0.], y=[0.], ref=[0], weight=[1.], spectra=[spectrum])
    train = OpticalTrain()
    prepared = train.prepare_source(source)
    np.testing.assert_array_equal(prepared.spectra[0].waveset, spectrum.waveset)


def test_prepare_cube_retains_faint_contrast_in_double_precision():
    from scopesim.tests.mocks.py_objects.source_objects import _cube_source
    source = _cube_source()
    field = source.cube_fields[0]
    field.data.fill(1e8)
    field.data[:, 0, 0] += 1.
    prepared = OpticalTrain().prepare_source(source)
    data = prepared.cube_fields[0].data
    contrast = (data[:, 0, 0] - data[:, 0, 1])/data[:, 0, 1]
    np.testing.assert_allclose(contrast, 1e-8, rtol=1e-6)
