from types import SimpleNamespace
import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from scopesim.effects.spectral_trace_list_utils import XiLamImage


def line_cube(step, offset=0):
    wave = np.linspace(.995, 1.005, 2001)
    eta = np.arange(-.5+step/2, .5, step)
    spatial = np.exp(-.5*((eta-offset)/.12)**2)
    profile = np.exp(-.5*((wave-1)/.00002)**2)
    cube = profile[:, None, None]*spatial[None, :, None]*np.ones((1, 1, 3))
    header = fits.Header({'CTYPE1': 'LINEAR', 'CTYPE2': 'LINEAR', 'CTYPE3': 'WAVE', 'CUNIT1': 'arcsec', 'CUNIT2': 'arcsec', 'CUNIT3': 'um', 'CDELT1': .1, 'CDELT2': step, 'CDELT3': wave[1]-wave[0], 'CRPIX1': 1., 'CRPIX2': 1., 'CRPIX3': 1., 'CRVAL1': 0., 'CRVAL2': eta[0], 'CRVAL3': wave[0]})
    return SimpleNamespace(cube=fits.ImageHDU(cube, header), meta={'xi_min': 0*u.arcsec})



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

