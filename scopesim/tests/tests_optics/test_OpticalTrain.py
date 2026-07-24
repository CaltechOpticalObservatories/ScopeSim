
from copy import deepcopy
from types import SimpleNamespace
import pytest
from pytest import approx
from unittest.mock import patch

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.table import Table

import scopesim as sim
from scopesim.detector import DetectorManager
from scopesim.optics.fov_manager import FOVManager
from scopesim.optics.image_plane import ImagePlane
from scopesim.optics.optical_train import OpticalTrain
from scopesim.optics.optics_manager import OpticsManager
from scopesim.optics.optical_element import OpticalElement
from scopesim.commands.user_commands import UserCommands
from scopesim.effects import (
    DetectorList,
    Effect,
    SpectralTraceList,
    UnequalBinnedImage,
)
from scopesim.utils import find_file

from scopesim.tests.mocks.py_objects import source_objects as src_objs

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


PLOTS = False


@pytest.fixture(scope="module", autouse=True)
def patch_globals():
    """Prevent modification of globals from within this module."""
    with patch("scopesim.rc.__currsys__"):
        yield


# TODO: check if class scope breaks anything (used to be function scope)
@pytest.fixture(scope="class")
def cmds(mock_path, mock_path_yamls):
    with patch("scopesim.rc.__search_path__", [mock_path, mock_path_yamls]):
        return UserCommands(yamls=[find_file("CMD_mvs_cmds.yaml")])

@pytest.fixture(scope="class")
def cmds_with_ignore(mock_path, mock_path_yamls):
    with patch("scopesim.rc.__search_path__", [mock_path, mock_path_yamls]):
        cmds = UserCommands(yamls=[find_file("CMD_mvs_cmds.yaml")])
        cmds.ignore_effects += ["detector QE curve"]
        return cmds


# TODO: check if class scope breaks anything (used to be function scope)
@pytest.fixture(scope="class")
def unity_cmds(mock_path, mock_path_yamls):
    with patch("scopesim.rc.__search_path__", [mock_path, mock_path_yamls]):
        return UserCommands(yamls=[find_file("CMD_unity_cmds.yaml")])


@pytest.fixture(scope="function")
def tbl_src():
    return src_objs._table_source()


@pytest.fixture(scope="function")
def im_src():
    return src_objs._image_source()


@pytest.fixture(scope="function")
def unity_src():
    return src_objs._unity_source(n=10001)


@pytest.fixture(scope="class")
def simplecado_opt(mock_path_yamls):
    simplecado_yaml = str(mock_path_yamls / "SimpleCADO.yaml")
    cmd = sim.UserCommands(yamls=[simplecado_yaml])
    return sim.OpticalTrain(cmd)


def spectral_geometry_train(detector_count=1):
    """Return a small configured train that needs no observation."""
    xi_grid, wave_grid = np.meshgrid(
        np.linspace(-1, 1, 5),
        np.linspace(1, 2, 6),
        indexing="ij",
    )
    trace_table = Table({
        "wavelength": wave_grid.ravel() * u.um,
        "s": xi_grid.ravel() * u.arcsec,
        "x": (wave_grid.ravel() - 1.5 + 0.1 * xi_grid.ravel()) * u.mm,
        "y": (0.2 * xi_grid.ravel()) * u.mm,
    })
    trace_hdu = fits.BinTableHDU(trace_table, name="linear")
    catalog = fits.BinTableHDU(Table({
        "description": ["linear"],
        "extension_id": [2],
        "aperture_id": [0],
        "image_plane_id": [0],
    }))
    primary = fits.PrimaryHDU()
    primary.header["ECAT"] = 1
    primary.header["EDATA"] = 2
    trace_list = SpectralTraceList(
        hdulist=fits.HDUList([primary, catalog, trace_hdu]),
        wave_colname="wavelength",
        s_colname="s",
    )

    detector_list = DetectorList(
        image_plane_id=0,
        array_dict={
            "id": list(range(detector_count)),
            "x_cen": np.linspace(
                0, 12 * (detector_count - 1), detector_count),
            "y_cen": np.zeros(detector_count),
            "x_size": np.full(detector_count, 100),
            "y_size": np.full(detector_count, 80),
            "pixel_size": np.full(detector_count, 0.1),
            "angle": np.zeros(detector_count),
            "gain": np.ones(detector_count),
        },
        x_cen_unit="mm",
        y_cen_unit="mm",
        x_size_unit="pixel",
        y_size_unit="pixel",
        pixel_size_unit="mm",
        angle_unit="deg",
        gain_unit="electron/adu",
    )

    train = OpticalTrain()
    train.optics_manager = SimpleNamespace(
        get_all=lambda effect_class: [trace_list],
        detector_effects=[],
    )
    train.fov_manager = SimpleNamespace(fovs=[
        SimpleNamespace(
            trace_id="linear",
            meta={
                "image_plane_id": 0,
                "wave_min": 1 * u.um,
                "wave_max": 2 * u.um,
            },
        ),
    ])
    train.image_planes = [ImagePlane(detector_list.image_plane_header)]
    train.detector_managers = [DetectorManager(detector_list)]
    return train, trace_list


@pytest.mark.usefixtures("patch_mock_path")
class TestInit:
    def test_initialises_with_nothing(self):
        assert isinstance(OpticalTrain(), OpticalTrain)

    def test_initialises_with_basic_commands(self, cmds):
        opt = OpticalTrain(cmds=cmds)
        assert isinstance(opt, OpticalTrain)

    def test_has_user_commands_object_after_initialising(self, cmds):
        opt = OpticalTrain(cmds=cmds)
        assert isinstance(opt.cmds, UserCommands)

    def test_has_optics_manager_object_after_initialising(self, cmds):
        opt = OpticalTrain(cmds=cmds)
        assert isinstance(opt.optics_manager, OpticsManager)

    def test_has_fov_manager_object_after_initialising(self, cmds):
        opt = OpticalTrain(cmds=cmds)
        print(opt.fov_manager)
        assert isinstance(opt.fov_manager, FOVManager)

    def test_has_image_plane_object_after_initialising(self, cmds):
        opt = OpticalTrain(cmds=cmds)
        assert isinstance(opt.image_planes[0], ImagePlane)

    def test_has_yaml_dict_object_after_initialising(self, cmds):
        opt = OpticalTrain(cmds=cmds)
        assert isinstance(opt.yaml_dicts, list) and len(opt.yaml_dicts) > 0

    def test_ignore_effects_works(self, cmds_with_ignore):
        opt = OpticalTrain(cmds=cmds_with_ignore)
        assert opt["detector QE curve"].include is False


class TestTraceDetectorCoordinates:
    def test_maps_live_trace_to_zero_origin_detector_pixels(self):
        train, _ = spectral_geometry_train()

        coordinates = train.trace_detector_coordinates(
            wavelengths={"linear": [1.5] * u.um})

        assert coordinates["trace_id"][0] == "linear"
        assert coordinates["readout_index"][0] == 0
        assert coordinates["detector_x"][0].to_value(u.pixel) == approx(49.5)
        assert coordinates["detector_y"][0].to_value(u.pixel) == approx(39.5)

    def test_returns_all_trace_wavelengths_and_requested_slit_positions(self):
        train, _ = spectral_geometry_train()

        coordinates = train.trace_detector_coordinates(
            xi=[-0.5, 0.5] * u.arcsec)

        assert len(coordinates) == 12
        assert set(coordinates["xi"].to_value(u.arcsec)) == {-0.5, 0.5}
        assert coordinates["wavelength"].unit == u.um
        assert coordinates["detector_x"].unit == u.pixel

    def test_accepts_unit_roundoff_at_configured_wavelength_boundary(self):
        train, _ = spectral_geometry_train()
        boundary = np.nextafter(1.0, 0.0) * u.um

        coordinates = train.trace_detector_coordinates(
            wavelengths={"linear": [boundary]})

        assert coordinates["wavelength"][0] == boundary

    def test_uses_changed_live_trace_transform(self):
        train, trace_list = spectral_geometry_train()
        before = train.trace_detector_coordinates(
            wavelengths={"linear": [1.5] * u.um})

        trace = trace_list.spectral_traces["linear"]
        trace.meta["offset_y"] = 0.1
        trace.compute_interpolation_functions()
        after = train.trace_detector_coordinates(
            wavelengths={"linear": [1.5] * u.um})

        shift = after["detector_y"][0] - before["detector_y"][0]
        assert shift.to_value(u.pixel) == approx(1)

    def test_rejects_multiple_detectors_on_one_image_plane(self):
        train, _ = spectral_geometry_train(detector_count=2)

        with pytest.raises(NotImplementedError, match="multiple detectors"):
            train.trace_detector_coordinates()

    def test_rejects_post_extraction_binning(self):
        train, _ = spectral_geometry_train()
        train.optics_manager.detector_effects = [
            UnequalBinnedImage(binx=2, biny=1),
        ]

        with pytest.raises(NotImplementedError, match="binning"):
            train.trace_detector_coordinates()


@pytest.mark.slow
@pytest.mark.usefixtures("patch_mock_path")
class TestObserve:
    """
    Almost all tests here are for visual inspection.
    No asserts, this just to test that the puzzle gets put back together
    after it is chopped up by the FOVs.
    """
    def test_observe_works_for_none(self, cmds):
        opt = OpticalTrain(cmds)
        opt.observe()
        empty = sim.source.source_templates.empty_sky()
        assert(opt._last_source.fields[0].field == empty.fields[0].field)

    def test_observe_works_for_table(self, cmds, tbl_src):
        opt = OpticalTrain(cmds)
        opt.observe(tbl_src)

        if PLOTS:
            plt.imshow(opt.image_planes[0].image.T, origin="lower",
                       norm=LogNorm())
            plt.show()

    def test_observe_works_for_image(self, cmds, im_src):
        opt = OpticalTrain(cmds)
        opt.observe(im_src)

        if PLOTS:
            plt.imshow(opt.image_planes[0].image.T, origin="lower",
                       norm=LogNorm())
            plt.show()

    def test_observe_works_for_source_distributed_over_several_fovs(self, cmds,
                                                                    im_src):

        orig_sum = np.sum(im_src.fields[0].data)

        cmds["SIM_PIXEL_SCALE"] = 0.02
        opt = OpticalTrain(cmds)
        opt.observe(im_src)

        wave = np.arange(0.5, 2.51, 0.1)*u.um
        unit = u.Unit("ph s-1 m-2 um-1")
        implane = opt.image_planes[0]
        final_sum = np.sum(implane.image)
        print(orig_sum, final_sum)

        if PLOTS:
            for fov in opt.fov_manager.fovs:
                cnrs = fov.corners[1]
                plt.plot(cnrs[0], cnrs[1])

            plt.imshow(implane.image.T, origin="lower", norm=LogNorm(),
                       extent=(-implane.hdu.header["NAXIS1"] / 2,
                               implane.hdu.header["NAXIS1"] / 2,
                               -implane.hdu.header["NAXIS2"] / 2,
                               implane.hdu.header["NAXIS2"] / 2,))
            plt.colorbar()
            plt.show()

    def test_observe_works_for_many_sources_distributed(self, cmds, im_src):
        orig_sum = np.sum(im_src.fields[0].data)
        im_src.fields[0].data += 1
        im_src1 = deepcopy(im_src)
        im_src2 = deepcopy(im_src)
        im_src2.shift(7, 7)
        im_src3 = deepcopy(im_src)
        im_src3.shift(-10, 14)
        im_src4 = deepcopy(im_src)
        im_src4.shift(-4, -6)
        im_src5 = deepcopy(im_src)
        im_src5.shift(15, -15)
        multi_img = im_src1 + im_src2 + im_src3 + im_src4 + im_src5

        cmds["SIM_PIXEL_SCALE"] = 0.02
        opt = OpticalTrain(cmds)
        opt.observe(multi_img)

        implane = opt.image_planes[0]
        final_sum = np.sum(implane.image)
        print(orig_sum, final_sum)

        if PLOTS:
            for fov in opt.fov_manager.fovs:
                cnrs = fov.corners[1]
                plt.plot(cnrs[0], cnrs[1])

            plt.imshow(implane.image.T, origin="lower", norm=LogNorm(),
                       extent=(-implane.hdu.header["NAXIS1"] / 2,
                               implane.hdu.header["NAXIS1"] / 2,
                               -implane.hdu.header["NAXIS2"] / 2,
                               implane.hdu.header["NAXIS2"] / 2,))
            plt.colorbar()
            plt.show()

    def test_works_with_a_pointer_to_fits_imagehdu(self, cmds):
        # Basically just checking to make sure observe doesn't throw an error
        # when passed a Source object with a file pointer ImageHDU
        fits_src = src_objs._fits_image_source()
        array_src = src_objs._image_source()

        src = fits_src + array_src
        opt = OpticalTrain(cmds)
        opt.observe(src)

        assert np.sum(opt.image_planes[0].data) > 0


@pytest.mark.usefixtures("patch_mock_path")
class TestReadout:
    def test_readout_works_when_source_observed(self, unity_cmds, unity_src):

        opt = OpticalTrain(unity_cmds)
        opt.observe(unity_src)
        hdus = opt.readout()
        hdu = hdus[0]

        if PLOTS:
            plt.subplot(221)
            plt.imshow(unity_src.fields[0].data)
            plt.colorbar()

            plt.subplot(222)
            plt.imshow(opt.image_planes[0].image)
            plt.colorbar()

            plt.subplot(223)
            plt.imshow(hdu[1].data)
            plt.colorbar()
            plt.show()

        _ = np.average(unity_src.fields[0].data)
        assert np.median(hdu[1].data) == approx(np.pi / 4., rel=1e-2)


class TestGetItems:
    def test_optical_element_returned_for_unique_name(self, simplecado_opt):
        print(type(simplecado_opt["test_detector"]))
        assert isinstance(simplecado_opt["test_detector"], OpticalElement)

    def test_effect_returned_for_unique_name(self, simplecado_opt):
        assert isinstance(simplecado_opt["test_detector_list"], Effect)

    def test_raises_error_for_bogus_string(self, simplecado_opt):
        with pytest.raises(ValueError):
            simplecado_opt["bogus"]

    def test_list_of_effects_returned_for_effect_class(self, simplecado_opt):
        effects = simplecado_opt[DetectorList]
        assert isinstance(effects, list)
        assert len(effects) == 1


class TestSetItems:
    def test_effect_kwarg_can_be_changed_using(self, simplecado_opt):
        simplecado_opt["dark_current"] = {"value": 0.2}
        assert simplecado_opt["dark_current"].meta["value"] == 0.2

    def test_effect_include_can_be_toggled_with_setitem(self, simplecado_opt):
        assert simplecado_opt["dark_current"].include is True
        simplecado_opt["dark_current"].include = False
        assert simplecado_opt["dark_current"].include is False
        assert simplecado_opt["dark_current"].meta["include"] is False


class TestListEffects:
    def test_effects_listed_in_table(self, simplecado_opt):
        assert isinstance(simplecado_opt.effects, Table)
        simplecado_opt["dark_current"].include = False
        assert bool(simplecado_opt.effects["included"][1]) is False
        simplecado_opt["alt_dark_current"].include = True
        assert bool(simplecado_opt.effects["included"][2]) is True

        print("\n", simplecado_opt.effects)


class TestShutdown:
    """Test that fits files are closed on shutdown of OpticalTrain"""

    def test_files_closed_on_shutdown(self, simplecado_opt, mock_path):
        """Test for closed files in two ways:
        - `closed` flag is set to True
        - data access fails
        """
        # Add an effect with a psf
        with patch("scopesim.rc.__search_path__", [mock_path]):
            psf = sim.effects.FieldConstantPSF(filename="test_ConstPSF.fits",
                                               name="testpsf",
                                               cmds=simplecado_opt.cmds)
        simplecado_opt.optics_manager.add_effect(psf)
        # This is just to make sure that we have an open file
        assert not simplecado_opt['testpsf']._file._file.closed

        simplecado_opt.shutdown()

        # 1. Check the `closed` flags where available
        flags = []
        for effect_name in simplecado_opt.effects['name']:
            try:
                flags.append(simplecado_opt[effect_name]._file._file.closed)
            except AttributeError:
                pass

        assert all(flags)

        # 2. Check that data access fails
        with pytest.raises(ValueError):
            print(simplecado_opt['testpsf']._file[2].data)
