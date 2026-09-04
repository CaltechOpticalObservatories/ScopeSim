import logging

import numpy as np
import pytest
from astropy import units as u

from scopesim.effects.psfs.analytical import MoffatPSF


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
