from types import SimpleNamespace
import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from scopesim.effects.spectral_trace_list_utils import XiLamImage, SpectralTrace
from astropy.table import Table
from scopesim.optics.image_plane_utils import header_from_list_of_xy


def test_single_precision_cube_is_integrated_in_double_precision():
    fov = line_cube(.05)
    fov.cube.data = np.zeros(fov.cube.data.shape, dtype=np.float32)
    fov.cube.data[1000, :, :] = 1.
    fov.cube.data[1000, 0, :] = 1e8
    image = XiLamImage(fov, lambda wave: .00001, slit_dispersion=lambda wave: .002)
    # Integration across eta must retain the 19 faint cells beside the bright
    # cell. The remapped line lies well inside the spectral boundaries.
    expected = (1e8 + 19)*.05
    np.testing.assert_allclose(image.image.sum(axis=1), expected, rtol=1e-12)


def line_cube(step, offset=0):
    wave = np.linspace(.995, 1.005, 2001)
    eta = np.arange(-.5+step/2, .5, step)
    spatial = np.exp(-.5*((eta-offset)/.12)**2)
    profile = np.exp(-.5*((wave-1)/.00002)**2)
    cube = profile[:, None, None]*spatial[None, :, None]*np.ones((1, 1, 3))
    header = fits.Header({'CTYPE1': 'LINEAR', 'CTYPE2': 'LINEAR', 'CTYPE3': 'WAVE', 'CUNIT1': 'arcsec', 'CUNIT2': 'arcsec', 'CUNIT3': 'um', 'CDELT1': .1, 'CDELT2': step, 'CDELT3': wave[1]-wave[0], 'CRPIX1': 1., 'CRPIX2': 1., 'CRPIX3': 1., 'CRVAL1': 0., 'CRVAL2': eta[0], 'CRVAL3': wave[0]})
    return SimpleNamespace(cube=fits.ImageHDU(cube, header), meta={'xi_min': 0*u.arcsec, 'eta_min': -.5*u.arcsec, 'eta_max': .5*u.arcsec})



def test_slit_dispersion_is_independent_of_numerical_sky_grid():
    spectra = [XiLamImage(line_cube(step), lambda w: .00001, slit_dispersion=lambda w: .0002) for step in [.1, .05, .025]]
    # Finite spatial quadrature changes, while the physical slit dispersion stays fixed.
    for image in spectra[:-1]:
        np.testing.assert_allclose(image.image, spectra[-1].image, atol=.001, rtol=.02)



def test_calibrated_slit_shift_has_forward_model_sign():
    image = XiLamImage(line_cube(.025, offset=.1), lambda w: .00001, slit_dispersion=lambda w: .002)
    centroid = np.sum(image.image[1]*image.lam)/image.image[1].sum()
    assert centroid == pytest.approx(1.0002, abs=2e-6)



def test_optical_lsf_preserves_photons_and_adds_variance():
    image = XiLamImage(line_cube(.05), lambda w: 0., slit_dispersion=lambda w: 0.)
    before = np.trapezoid(image.image[1], image.lam)
    image.convolve_lsf(10000)
    after = np.trapezoid(image.image[1], image.lam)
    assert after == pytest.approx(before, rel=.001)
    center = np.sum(image.lam*image.image[1])/image.image[1].sum()
    variance = np.sum((image.lam-center)**2*image.image[1])/image.image[1].sum()
    assert variance == pytest.approx(.00002**2 + (1/10000/np.sqrt(8*np.log(2)))**2, rel=.02)


def test_slit_coordinates_follow_cube_wcs():
    fov = line_cube(.025)
    fov.cube.header['CRVAL1'] = 2.
    fov.cube.header['CRPIX1'] = 2.
    fov.cube.header['CRVAL2'] += .1
    image = XiLamImage(fov, lambda wave: .00001, slit_dispersion=lambda wave: .002)
    np.testing.assert_allclose(image.xi, [1.9, 2., 2.1])
    centroid = np.sum(image.image[1] * image.lam) / image.image[1].sum()
    assert centroid == pytest.approx(1.0002, abs=2e-6)


def test_translating_an_ifu_slice_preserves_its_local_spectrum():
    fov = line_cube(.025)
    centered = XiLamImage(fov, lambda wave: .00001, slit_dispersion=lambda wave: .002)
    fov.cube.header['CRVAL2'] += 2.
    fov.meta['eta_min'] += 2*u.arcsec
    fov.meta['eta_max'] += 2*u.arcsec
    translated = XiLamImage(fov, lambda wave: .00001, slit_dispersion=lambda wave: .002)
    np.testing.assert_allclose(translated.image, centered.image, atol=1e-7)


def test_missing_aperture_geometry_fails():
    fov = line_cube(.025)
    del fov.meta['eta_min']
    with pytest.raises(KeyError, match='eta_min'):
        XiLamImage(fov, lambda wave: .00001, slit_dispersion=lambda wave: .002)


def test_detector_jacobian_uses_physical_pixel_area():
    header = fits.Header({'CTYPE1D': 'LINEAR', 'CTYPE2D': 'LINEAR', 'CUNIT1D': 'm', 'CUNIT2D': 'm', 'CDELT1D': .000015, 'CDELT2D': .000030, 'CRPIX1D': 1., 'CRPIX2D': 1., 'CRVAL1D': 0., 'CRVAL2D': 0.})
    assert SpectralTrace._detector_pixel_area(header) == pytest.approx(.015*.030)


def test_rectified_interpolation_reproduces_a_linear_density_field():
    fov = line_cube(.1)
    wave = .995 + np.arange(2001)*fov.cube.header['CDELT3']
    xi = np.arange(3)*.1
    fov.cube.data[:] = 2 + 3*xi[None, None, :] + 5*wave[:, None, None]
    image = XiLamImage(fov, lambda w: 0., slit_dispersion=lambda w: 0.)
    positions, wavelengths = np.array([.03, .17]), np.array([.998, 1.003])
    expected = 2 + 3*positions + 5*wavelengths
    np.testing.assert_allclose(image.interp(positions, wavelengths, grid=False), expected, rtol=2e-6)


@pytest.mark.parametrize('change', ['wavelength_origin', 'wavelength_step', 'wavelength_pc', 'slit_origin', 'slit_step', 'slit_pc', 'slice_center', 'dispersion', 'source', 'units'])
def test_reused_slit_operator_matches_fresh_after_direct_changes(change):
    fov = line_cube(.1)
    cache = {}
    slope = [.002]
    dispersion = lambda wave: slope[0]*wave
    XiLamImage(fov, lambda wave: .00001, slit_dispersion=dispersion, _cache=cache)
    if change == 'wavelength_origin': fov.cube.header['CRVAL3'] += .001
    if change == 'wavelength_step': fov.cube.header['CDELT3'] *= 1.1
    if change == 'wavelength_pc': fov.cube.header['PC3_3'] = 1.1
    if change == 'slit_origin': fov.cube.header['CRVAL2'] += .04
    if change == 'slit_step': fov.cube.header['CDELT2'] *= 1.1
    if change == 'slit_pc': fov.cube.header['PC2_2'] = 1.1
    if change == 'slice_center': fov.meta['eta_min'] += .04*u.arcsec
    if change == 'dispersion': slope[0] *= 1.3
    if change == 'source': fov.cube.data *= 2.
    if change == 'units':
        fov.cube.header['CUNIT2'], fov.cube.header['CUNIT3'] = 'deg', 'nm'
        for keyword in ('CDELT2', 'CRVAL2'): fov.cube.header[keyword] /= 3600
        for keyword in ('CDELT3', 'CRVAL3'): fov.cube.header[keyword] *= 1000
    reused = XiLamImage(fov, lambda wave: .00001, slit_dispersion=dispersion, _cache=cache)
    fresh = XiLamImage(fov, lambda wave: .00001, slit_dispersion=dispersion)
    np.testing.assert_array_equal(reused.image, fresh.image)


@pytest.mark.parametrize('coordinate', ['xy2lam', 'xy2xi'])
def test_direct_optical_transform_edits_reach_next_projection(coordinate):
    xi, wave = np.meshgrid(np.linspace(-1., 1., 5), np.linspace(.99, 1.01, 9))
    table = Table({'wavelength': wave.ravel()*u.um, 's': xi.ravel()*u.arcsec, 'x': (wave.ravel()-1)*1000*u.mm, 'y': xi.ravel()*.15*u.mm})
    reused, fresh = [SpectralTrace(table.copy(), trace_flux_jacobian=True) for _ in range(2)]
    fov = line_cube(.1)
    fov.cube.data *= np.array([1., 2., 4.])[None, None, :]
    fov.header = fov.cube.header
    fov.detector_header = header_from_list_of_xy([-10., 10.], [-2., 2.], .015, 'D')
    fov.trace_id = 'linear'
    fov.meta.update(wave_min=.995*u.um, wave_max=1.005*u.um, xi_min=-.05*u.arcsec, xi_max=.25*u.arcsec)
    before = reused.map_spectra_to_focal_plane(fov).data.copy()
    for trace in (reused, fresh):
        if coordinate == 'xy2lam': trace.xy2lam.matrix[0, 0] += .0002
        if coordinate == 'xy2xi': trace.xy2xi.matrix[0, 0] += .04
    actual = reused.map_spectra_to_focal_plane(fov).data
    expected = fresh.map_spectra_to_focal_plane(fov).data
    np.testing.assert_array_equal(actual, expected)
    assert np.max(abs(actual-before)) > .001*before.max()
