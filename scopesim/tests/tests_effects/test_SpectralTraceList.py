"""Tests for module spectral_trace_list.py"""

import pytest
from unittest.mock import patch

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table


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

#class TestRectification:

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
    @pytest.mark.parametrize('angle', [-25, 0, 25])
    @pytest.mark.parametrize('offset', [(0., 0.), (1.5, -1.2)])
    def test_fixed_placement_and_detector_intersections(self, tmp_path, angle, offset):
        params_file = tmp_path / 'echelle_trace_parameters.dat'
        _write_echelle_trace_params(params_file, detector_angle=angle)
        params = Table.read(params_file, format='ascii')
        for key, value in {'m0': 75, 'n': 6, 'min_wave': 960, 'max_wave': 1340, 'n_disp': 2048, 'n_xdisp': 1024, 'disp_freq': 24., 'xdisp_freq': 350.}.items():
            params[key][0] = value
        params['xbeta_center'] = [10.9]
        params['detector_offset_x'], params['detector_offset_y'] = [offset[0]], [offset[1]]
        params.meta['comments'].extend(['detector_offset_x_unit : mm', 'detector_offset_y_unit : mm'])
        params.write(params_file, format='ascii', overwrite=True)
        effect = EchelleSpectralTraceList(cmds=_echelle_cmds(), filename=str(params_file), wave_colname='wavelength', s_colname='s')
        model = effect._spectrographs['b']
        nx, ny = model.detector.n_pix_x, model.detector.n_pix_y
        raw_grid = np.linspace(-2*nx, 3*nx, 50001)
        for order, edges in zip(model.orders, model.edge_wave(fsr=True)):
            wave = np.linspace(edges[0], edges[1], 4*nx + 1)
            x, y = model._detector_pixels(wave, order)
            # Rotate about the camera origin, then translate in detector axes.
            xraw = model.wavelength_to_x_pixel(wave, order).value - nx/2
            yraw = model.wavelength_to_y_pixel(wave).value[None, :] - ny/2 + model._slit_offset_pix[:, None]
            c, s = np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))
            expected_x = c*xraw - s*yraw + nx/2 + offset[0]/.015
            expected_y = s*xraw + c*yraw + ny/2 + offset[1]/.015
            np.testing.assert_allclose(x, expected_x, atol=1e-9, rtol=0)
            np.testing.assert_allclose(y, expected_y, atol=1e-9, rtol=0)
            # Translation changes placement, not dispersion or slit length.
            xrot, yrot = model._rotated_detector_pixels(wave, order)
            np.testing.assert_allclose(np.diff(x), np.diff(xrot), atol=1e-9, rtol=0)
            np.testing.assert_allclose(np.diff(y), np.diff(yrot), atol=1e-9, rtol=0)
            slit_length = np.hypot(x[2] - x[0], y[2] - y[0])
            np.testing.assert_allclose(slit_length, model._slit_offset_pix[2] - model._slit_offset_pix[0], atol=1e-9, rtol=0)

            # A deliberately oversized wavelength search independently checks
            # that the translated clipping search has not lost either edge.
            wave = model.x_pixel_to_wavelength(raw_grid, order)
            x, y = model._detector_pixels(wave, order)
            inside = np.any((x >= 0) & (x <= nx) & (y >= 0) & (y <= ny), axis=0)
            trace_id = f'b_{order}'
            if not inside.any():
                assert trace_id not in effect.spectral_traces
                continue
            trace = effect.spectral_traces[trace_id]
            actual = model.wavelength_to_x_pixel(u.Quantity(trace.table['wavelength']), order).value
            expected = raw_grid[inside][[0, -1]]
            # The generator locates each edge on a roughly one-pixel grid.
            np.testing.assert_allclose([actual.min(), actual.max()], expected, atol=3, rtol=0)

    @pytest.mark.parametrize('missing', ['detector_offset_x', 'detector_offset_y', 'detector_offset_x_unit', 'detector_offset_y_unit'])
    def test_incomplete_placement_fails(self, tmp_path, missing):
        path = tmp_path / 'echelle_trace_parameters.dat'
        _write_echelle_trace_params(path)
        params = Table.read(path, format='ascii')
        for axis in ['x', 'y']:
            key = f'detector_offset_{axis}'
            if missing != key:
                params[key] = [0.]
            if missing != f'{key}_unit':
                params.meta['comments'].append(f'{key}_unit : mm')
        params.write(path, format='ascii', overwrite=True)
        with pytest.raises(KeyError, match=missing):
            EchelleSpectralTraceList(cmds=_echelle_cmds(), filename=str(path))

    @pytest.mark.parametrize('missing_value', ['detector_offset_x', 'detector_offset_y'])
    def test_blank_placement_value_does_not_become_zero(self, tmp_path, missing_value):
        path = tmp_path / 'echelle_trace_parameters.dat'
        _write_echelle_trace_params(path)
        params = Table.read(path, format='ascii')
        for axis in ['x', 'y']:
            key = f'detector_offset_{axis}'
            params[key] = np.ma.array([0.], mask=[key == missing_value])
            params.meta['comments'].append(f'{key}_unit : mm')
        params.write(path, format='ascii', overwrite=True)
        with pytest.raises(ValueError, match='Both detector offsets must have values'):
            EchelleSpectralTraceList(cmds=_echelle_cmds(), filename=str(path))

    def test_placement_requires_length_units(self, tmp_path):
        path = tmp_path / 'echelle_trace_parameters.dat'
        _write_echelle_trace_params(path)
        params = Table.read(path, format='ascii')
        params['detector_offset_x'], params['detector_offset_y'] = [0.], [0.]
        params.meta['comments'].extend(['detector_offset_x_unit : arcsec', 'detector_offset_y_unit : mm'])
        params.write(path, format='ascii', overwrite=True)
        with pytest.raises(u.UnitConversionError):
            EchelleSpectralTraceList(cmds=_echelle_cmds(), filename=str(path))

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

        model = spt._spectrographs['b']
        order = int(trace.meta['trace_id'].split('_')[1])
        wave = u.Quantity(trace.table['wavelength'][:n_wave])
        model_x, model_y = model._detector_pixels(wave, order)
        np.testing.assert_allclose(model_x, xpix, atol=1e-8, rtol=0)
        np.testing.assert_allclose(model_y, ypix, atol=1e-8, rtol=0)

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

        # Rotation changes which wavelengths reach the rectangular detector;
        # clipping must occur after rotation, not at the old unrotated edges.
        assert trace.wave_min < trace.wave_max
        assert not np.allclose([trace.wave_min, trace.wave_max], [unrotated_trace.wave_min, unrotated_trace.wave_max])
        assert not np.allclose(trace.table["x"], unrotated_trace.table["x"])
        assert not np.allclose(trace.table["y"], unrotated_trace.table["y"])
