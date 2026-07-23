"""Tests for class SpectralEfficiency"""

import numpy as np
import pytest

from astropy import units as u
from astropy.io import fits

from scopesim.effects import EchelleSpectralEfficiency, SpectralEfficiency, \
    TERCurve


@pytest.fixture(name="speceff", scope="class")
def fixture_speceff(mock_path):
    """Instantiate SpectralEfficiency object"""
    return SpectralEfficiency(filename=str(mock_path / "TER_grating.fits"))


class TestSpectralEfficiency:
    def test_initialises_from_file(self, speceff):
        assert isinstance(speceff, SpectralEfficiency)

    def test_initialises_from_hdulist(self, mock_path):
        # fitsfile = find_file("TER_grating.fits")
        fitsfile = mock_path / "TER_grating.fits"
        with fits.open(fitsfile) as hdul:
            speceff = SpectralEfficiency(hdulist=hdul)
        assert isinstance(speceff, SpectralEfficiency)

    def test_has_efficiencies(self, speceff):
        efficiencies = speceff.efficiencies
        assert all(isinstance(effic, TERCurve)
                   for effic in efficiencies.values())


def _write_echelle_trace_params(path, dispersion_focal_length):
    path.write_text(
        "# min_wave_unit : nm\n"
        "# max_wave_unit : nm\n"
        "# echelle_blaze_unit : deg\n"
        "# focal_length_unit : mm\n"
        "# dispersion_focal_length_unit : mm\n"
        "# pixel_size_unit : mm\n"
        "# disp_freq_unit : mm\n"
        "# xdisp_freq_unit : mm\n"
        "prefix m0 n min_wave max_wave design_res echelle_blaze "
        "focal_length dispersion_focal_length fwhm detector_pad "
        "pixel_size n_disp n_xdisp disp_freq xdisp_freq xbeta_center\n"
        f"b 91 0 310 420 17799 65.6 225 {dispersion_focal_length} "
        "4.5 10 0.015 128 128 65.0 1.0 0\n",
        encoding="utf-8",
    )


def test_echelle_efficiency_uses_dispersion_focal_length(tmp_path):
    params_225 = tmp_path / "echelle_225.dat"
    params_270 = tmp_path / "echelle_270.dat"
    _write_echelle_trace_params(params_225, 225)
    _write_echelle_trace_params(params_270, 270)

    efficiency_225 = EchelleSpectralEfficiency(filename=str(params_225))
    efficiency_270 = EchelleSpectralEfficiency(filename=str(params_270))
    spectrograph = efficiency_270._spectrographs["b"]

    assert spectrograph.focal_length == 225 * u.mm
    assert spectrograph.dispersion_focal_length == 270 * u.mm

    wave = (
        efficiency_225._spectrographs["b"].central_wave(91)
        * np.linspace(0.99, 1.01, 20)
    )
    np.testing.assert_allclose(
        efficiency_225.efficiency_generator("b_91", wave),
        efficiency_270.efficiency_generator("b_91", wave),
    )
