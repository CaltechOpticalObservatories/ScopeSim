"""Direct tests for the Moffat kernel mathematics.

These were previously exercised only through MoffatPSF.get_kernel, so an error
in the closed-form integral could only show up as a slightly wrong PSF.
"""

import numpy as np
import pytest
from astropy import units as u
from scipy.integrate import dblquad

from scopesim.effects.psfs.analytical import moffat_cell_integrals, MoffatPSF

ALPHA = 4.765
FWHM = .7


def _gamma(fwhm=FWHM, alpha=ALPHA):
    return MoffatPSF.fwhm2gamma(fwhm*u.arcsec, alpha).to_value(u.arcsec)


def _moffat(x, y, gamma, alpha):
    """Normalised 2D Moffat surface brightness."""
    return (alpha - 1)/(np.pi*gamma**2) * (1 + (x**2 + y**2)/gamma**2)**(-alpha)


class TestFwhm2Gamma:
    @pytest.mark.parametrize("alpha", [1.5, 2.5, 4.765, 10.])
    @pytest.mark.parametrize("fwhm", [.2, .7, 1.3])
    def test_profile_is_half_its_peak_at_half_the_fwhm(self, fwhm, alpha):
        gamma = _gamma(fwhm, alpha)
        assert (1 + ((fwhm/2)/gamma)**2)**(-alpha) == pytest.approx(.5)

    def test_gamma_scales_linearly_with_fwhm(self):
        assert _gamma(1.4)/_gamma(.7) == pytest.approx(2.)

    def test_returns_the_input_angular_unit(self):
        assert MoffatPSF.fwhm2gamma(700*u.marcsec, ALPHA).unit == u.marcsec


class TestMoffatCellIntegrals:
    @pytest.mark.parametrize("centre", [(0., 0.), (.35, -.2), (1.1, .9)])
    def test_matches_numerical_quadrature_over_one_cell(self, centre):
        gamma, (cx, cy), step = _gamma(), centre, .1
        analytic = moffat_cell_integrals(np.array([cx]), np.array([cy]), step, step,
                                         np.array([gamma]), ALPHA)[0]
        numeric, _ = dblquad(lambda y, x: _moffat(x, y, gamma, ALPHA),
                             cx - step/2, cx + step/2, cy - step/2, cy + step/2,
                             epsabs=1e-13)
        assert analytic == pytest.approx(numeric, rel=1e-9)

    def test_integrates_to_unity_over_a_wide_grid(self):
        step, half = .1, 100
        centres = (np.arange(2*half + 1) - half)*step
        xx, yy = np.meshgrid(centres, centres, indexing="ij")
        total = moffat_cell_integrals(xx, yy, step, step, np.array([_gamma()]), ALPHA).sum()
        assert total == pytest.approx(1., rel=1e-5)

    def test_wavelength_averaging_is_the_mean_of_the_single_widths(self):
        # The kernel averages over sampled wavelengths inside this function.
        gammas = np.array([.3, .5])
        centres = (np.arange(81) - 40)*.1
        xx, yy = np.meshgrid(centres, centres, indexing="ij")
        averaged = moffat_cell_integrals(xx, yy, .1, .1, gammas, ALPHA)
        singles = [moffat_cell_integrals(xx, yy, .1, .1, gammas[[i]], ALPHA) for i in (0, 1)]
        np.testing.assert_allclose(averaged, sum(singles)/2, rtol=1e-12)

    def test_narrow_profile_is_resolved_without_oversampling(self):
        # A profile far narrower than the cell must still integrate to unity;
        # this is the case point sampling of the kernel would get wrong.
        step, half = 1., 60
        centres = (np.arange(2*half + 1) - half)*step
        xx, yy = np.meshgrid(centres, centres, indexing="ij")
        total = moffat_cell_integrals(xx, yy, step, step, np.array([.05]), ALPHA).sum()
        assert total == pytest.approx(1., rel=1e-4)

    def test_is_symmetric_about_the_origin(self):
        centres = (np.arange(21) - 10)*.1
        xx, yy = np.meshgrid(centres, centres, indexing="ij")
        cells = moffat_cell_integrals(xx, yy, .1, .1, np.array([_gamma()]), ALPHA)
        np.testing.assert_allclose(cells, cells[::-1, :], rtol=1e-12)
        np.testing.assert_allclose(cells, cells[:, ::-1], rtol=1e-12)
