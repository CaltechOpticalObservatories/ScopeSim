"""Tests for module spectral_trace_list.py"""

import pytest
from unittest.mock import patch

import numpy as np
from astropy import units as u
from astropy.io import fits


from scopesim.effects.spectral_trace_list import SpectralTraceList, \
    SpectralTraceListWheel, EchelleSpectralTraceList
from scopesim.effects.spectral_trace_list_utils import SpectralTrace
from scopesim.commands import UserCommands
from scopesim.tests.mocks.py_objects import trace_list_objects as tlo
from scopesim.tests.mocks.py_objects import header_objects as ho


PLOTS = False

# pylint: disable=missing-class-docstring,
# pylint: disable=missing-function-docstring


@pytest.fixture(name="slit_header", scope="class")
def fixture_slit_header():
    return ho._short_micado_slit_header()


@pytest.fixture(name="long_slit_header", scope="class")
def fixture_long_slit_header():
    return ho._long_micado_slit_header()


@pytest.fixture(name="full_trace_list", scope="class")
def fixture_full_trace_list():
    """Instantiate a trace definition hdu list"""
    return tlo.make_trace_hdulist()


class TestInit:
    def test_initialises_with_nothing(self):
        assert isinstance(SpectralTraceList(), SpectralTraceList)

    def test_initialises_with_a_hdulist(self, full_trace_list):
        spt = SpectralTraceList(hdulist=full_trace_list)
        assert isinstance(spt, SpectralTraceList)
        assert isinstance(spt.data_container._file[2], fits.BinTableHDU)
        # next assert that dispersion axis determined correctly
        assert list(spt.spectral_traces.values())[2].dispersion_axis == 'y'

    def test_initialises_with_filename(self, mock_dir):
        micado_spec_dir = mock_dir / "MICADO_SPEC"
        with patch("scopesim.rc.__search_path__", [micado_spec_dir]):
            spt = SpectralTraceList(filename="TRACE_MICADO.fits",
                                    wave_colname="wavelength", s_colname="xi")
        assert isinstance(spt, SpectralTraceList)
        # assert that dispersion axis taken correctly from header keyword
        assert list(spt.spectral_traces.values())[2].dispersion_axis == 'y'

    def test_getitem_returns_spectral_trace(self, full_trace_list):
        slist = SpectralTraceList(hdulist=full_trace_list)
        assert isinstance(slist['Sheared'], SpectralTrace)

    def test_setitem_appends_correctly(self, full_trace_list):
        slist = SpectralTraceList(hdulist=full_trace_list)
        n_trace = len(slist.spectral_traces)
        spt = tlo.trace_1()
        slist["New trace"] = spt
        assert len(slist.spectral_traces) == n_trace + 1


@pytest.fixture(name="spectral_trace_list", scope="class")
def fixture_spectral_trace_list():
    """Instantiate a SpectralTraceList"""
    return SpectralTraceList(hdulist=tlo.make_trace_hdulist())

class TestRectification:
    def test_rectify_cube_not_implemented(self, spectral_trace_list):
        hdulist = fits.HDUList()
        with pytest.raises(NotImplementedError):
            spectral_trace_list.rectify_cube(hdulist)

    # def test_rectify_traces_needs_ximin_and_ximax(self, spectral_trace_list):
    #    hdulist = fits.HDUList([fits.PrimaryHDU()])
    #    with pytest.raises(KeyError):
    #        spectral_trace_list.rectify_traces(hdulist)


class TestSpectralTraceListWheel:
    @pytest.mark.usefixtures("no_file_error")
    def test_basic_init(self):
        """
        This is a super basic test just to see the thing basically works and
        parameters are passed correctly. Please feel free to improve this!!
        """
        kwargs = {"current_trace_list": "bogus",
                  "filename_format": "bogus_{}",
                  "trace_list_names": ["foo"]}
        stw = SpectralTraceListWheel(**kwargs)
        assert isinstance(stw, SpectralTraceListWheel)
        assert stw.meta["current_trace_list"] == "bogus"
        assert stw.meta["filename_format"] == "bogus_{}"
        assert stw.meta["trace_list_names"] == ["foo"]
        assert isinstance(stw.trace_lists["foo"], SpectralTraceList)
        assert stw.trace_lists["foo"].meta["filename"] == "bogus_foo"


def _write_echelle_trace_params(path, detector_angle=None, detector_pad=10):
    angle_unit = "# detector_angle_unit : deg\n" if detector_angle is not None else ""
    angle_col = " detector_angle" if detector_angle is not None else ""
    angle_value = f" {detector_angle}" if detector_angle is not None else ""
    path.write_text(
        "# min_wave_unit : nm\n"
        "# max_wave_unit : nm\n"
        "# echelle_blaze_unit : deg\n"
        "# focal_length_unit : mm\n"
        "# fwhm_unit : pixel\n"
        "# detector_pad_unit : pixel\n"
        "# pixel_size_unit : mm\n"
        "# n_disp_unit : pixel\n"
        "# n_xdisp_unit : pixel\n"
        "# disp_freq_unit : mm\n"
        "# xdisp_freq_unit : mm\n"
        "# slitlength_unit : arcsec\n"
        f"{angle_unit}"
        "prefix aperture_id image_plane_id m0 n min_wave max_wave "
        "design_res echelle_blaze focal_length fwhm detector_pad "
        "pixel_size n_disp n_xdisp disp_freq xdisp_freq slitlength "
        f"dispdir xbeta_center{angle_col}\n"
        f"b 0 0 91 0 310 420 17799 65.6 225 4.5 {detector_pad} 0.015 "
        f"128 128 65.0 1.0 10 x 0{angle_value}\n",
        encoding="utf-8",
    )


def _echelle_cmds():
    cmds = UserCommands()
    cmds["!INST.pixel_scale"] = 0.2
    return cmds


class TestEchelleSpectralTraceList:
    def test_does_not_write_generated_hdulist_by_default(self, tmp_path):
        params_file = tmp_path / "echelle_trace_parameters.dat"
        _write_echelle_trace_params(params_file)

        spt = EchelleSpectralTraceList(
            cmds=_echelle_cmds(),
            filename=str(params_file),
            wave_colname="wavelength",
            s_colname="s",
        )

        assert isinstance(spt, EchelleSpectralTraceList)
        assert not (tmp_path / "analytical_echelle_traces.fits").exists()

    def test_writes_generated_hdulist_to_working_dir_when_requested(
            self, tmp_path, monkeypatch):
        params_file = tmp_path / "echelle_trace_parameters.dat"
        _write_echelle_trace_params(params_file)
        monkeypatch.chdir(tmp_path)

        EchelleSpectralTraceList(
            cmds=_echelle_cmds(),
            filename=str(params_file),
            wave_colname="wavelength",
            s_colname="s",
            save_generated_hdulist=True,
        )

        assert (tmp_path / "analytical_echelle_traces.fits").exists()

    def test_detector_angle_clips_to_detector_not_padding(self, tmp_path):
        params_file = tmp_path / "echelle_trace_parameters.dat"
        _write_echelle_trace_params(params_file, detector_angle=25)

        spt = EchelleSpectralTraceList(
            cmds=_echelle_cmds(),
            filename=str(params_file),
            wave_colname="wavelength",
            s_colname="s",
        )

        trace = next(iter(spt.spectral_traces.values()))
        n_wave = len(trace.table) // 3
        xpix = (
            u.Quantity(trace.table["x"]).to_value(u.mm) / 0.015 + 64
        ).reshape(3, n_wave)
        ypix = (
            u.Quantity(trace.table["y"]).to_value(u.mm) / 0.015 + 64
        ).reshape(3, n_wave)
        inside = (
            (xpix >= 0)
            & (xpix <= 128)
            & (ypix >= 0)
            & (ypix <= 128)
        )

        assert np.all(np.any(inside, axis=0))
        assert trace.meta["detector_angle"] == 25
        assert trace.meta["detector_pad"] == 10

        padded_file = tmp_path / "echelle_trace_parameters_more_padding.dat"
        _write_echelle_trace_params(
            padded_file,
            detector_angle=25,
            detector_pad=30,
        )
        padded = EchelleSpectralTraceList(
            cmds=_echelle_cmds(),
            filename=str(padded_file),
            wave_colname="wavelength",
            s_colname="s",
        )
        padded_trace = next(iter(padded.spectral_traces.values()))

        assert trace.wave_min == pytest.approx(padded_trace.wave_min)
        assert trace.wave_max == pytest.approx(padded_trace.wave_max)

        unrotated_file = tmp_path / "echelle_trace_parameters_unrotated.dat"
        _write_echelle_trace_params(unrotated_file, detector_angle=0)
        unrotated = EchelleSpectralTraceList(
            cmds=_echelle_cmds(),
            filename=str(unrotated_file),
            wave_colname="wavelength",
            s_colname="s",
        )
        unrotated_trace = next(iter(unrotated.spectral_traces.values()))

        assert trace.wave_min == pytest.approx(unrotated_trace.wave_min)
        assert trace.wave_max == pytest.approx(unrotated_trace.wave_max)
        assert not np.allclose(trace.table["x"], unrotated_trace.table["x"])
        assert not np.allclose(trace.table["y"], unrotated_trace.table["y"])

    def test_keeps_rotated_detector_pixel_coordinates(self, tmp_path):
        params_file = tmp_path / "echelle_trace_parameters.dat"
        _write_echelle_trace_params(params_file, detector_angle=25)

        spt = EchelleSpectralTraceList(
            cmds=_echelle_cmds(),
            filename=str(params_file),
            wave_colname="wavelength",
            s_colname="s",
        )

        trace = next(iter(spt.spectral_traces.values()))
        xpix = trace.table["x_pix"].quantity.to_value(u.pixel)
        ypix = trace.table["y_pix"].quantity.to_value(u.pixel)

        np.testing.assert_allclose(
            xpix,
            trace.table["x"].quantity.to_value(u.mm) / 0.015 + 64,
        )
        np.testing.assert_allclose(
            ypix,
            trace.table["y"].quantity.to_value(u.mm) / 0.015 + 64,
        )
