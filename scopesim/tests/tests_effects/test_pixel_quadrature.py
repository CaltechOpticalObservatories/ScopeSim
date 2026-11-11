"""The per-pixel quadrature order used by SpectralTrace.map_spectra_to_focal_plane.

Order 1 reproduces upstream's single sample at the pixel centre; higher orders
integrate the spectral density across the pixel.
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table

from scopesim.commands import UserCommands
from scopesim.effects.spectral_trace_list_utils import SpectralTrace


def _linear_trace_table():
    xi_grid, wave_grid = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(1, 3, 5), indexing="ij")
    xi, wave = xi_grid.ravel(), wave_grid.ravel()
    return Table(data=[wave*u.um, xi*u.arcsec, (wave + .1*xi)*u.mm, (xi/10)*u.mm],
                 names=["wavelength", "s", "x", "y"])


class TestQuadratureOrderIsConfigured:
    def test_defaults_to_upstream_single_centre_sample(self):
        trace = SpectralTrace(_linear_trace_table(), cmds=UserCommands())
        assert trace.quadrature_order() == 1

    @pytest.mark.parametrize("order, n_nodes", [(1, 1), (2, 2), (4, 4)])
    def test_legendre_rule_matches_the_requested_order(self, order, n_nodes):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        assert len(nodes) == n_nodes
        # The mapping loop uses nodes/2 and weights/2, so the weights must sum
        # to 1 over the unit pixel for flux to be conserved at any order.
        assert (weights/2).sum() == pytest.approx(1.)

    def test_order_one_is_a_single_centre_sample_of_unit_weight(self):
        # This is exactly upstream's scheme: one sample at the pixel centre.
        nodes, weights = np.polynomial.legendre.leggauss(1)
        assert (nodes/2) == pytest.approx([0.])
        assert (weights/2) == pytest.approx([1.])

    def test_order_four_samples_away_from_the_centre(self):
        nodes, _ = np.polynomial.legendre.leggauss(4)
        assert np.all(np.abs(nodes/2) > 0.)
        assert np.max(np.abs(nodes/2)) < .5     # stays inside the pixel

    @pytest.mark.parametrize("order", [0, -1])
    def test_rejects_orders_below_one(self, order):
        trace = SpectralTrace(_linear_trace_table(), cmds=UserCommands(),
                              pixel_quadrature_order=order)
        with pytest.raises(ValueError, match="at least 1"):
            trace.quadrature_order()

    def test_opting_in_is_honoured_per_trace(self):
        trace = SpectralTrace(_linear_trace_table(), cmds=UserCommands(),
                              pixel_quadrature_order=4)
        assert trace.quadrature_order() == 4
