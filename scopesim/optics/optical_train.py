# -*- coding: utf-8 -*-

import copy
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.table import QTable
from astropy.wcs import WCS
from astropy.io.fits import HDUList

from tqdm.auto import tqdm

from synphot.units import PHOTLAM

from astar_utils.nested_mapping import RecursiveNestedMapping, is_bangkey, recursive_update

from .optics_manager import OpticsManager
from .fov_manager import FOVManager
from .image_plane import ImagePlane
from .fov import FieldOfView3D
from ..commands.user_commands import UserCommands
from ..detector import DetectorManager
from ..effects import ExtraFitsKeywords, SelectorWheel, MoffatPSF, TERCurve, SpectralEfficiency, EchelleSpectralEfficiency, SpectralTraceList
from ..effects.ter_curves import DichroicTree, FilterWheelBase, TaperedQuantumEfficiency
from ..effects.atmo_dispersion import ShiftFoV3D
from ..utils import from_currsys, top_level_catch, get_logger
from ..source.source_templates import empty_sky
from .. import __version__


logger = get_logger(__name__)

import multiprocessing as mp

N_PROCESSES = mp.cpu_count() - 1
USE_MULTIPROCESSING = False

def extract_source(fov, source):
    fov.extract_from(source)
    return fov

def view_fov(fov, hdu_type):
    fov.view(hdu_type)
    return fov

def apply_fov_effects(fov, fov_effects):
    for effect in fov_effects:
        fov = effect.apply_to(fov)
    return fov


class OpticalTrain:
    """
    The main class for controlling a simulation.

    Parameters
    ----------
    cmds : UserCommands, str
        If the name of an instrument is passed, OpticalTrain tries to find the
        instrument package, and internally creates the UserCommands object

    Examples
    --------
    Create an optical train::

        >>> import scopesim as im
        >>> cmd = sim.UserCommands("MICADO")
        >>> opt = sim.OpticalTrain(cmd)

    Observe a Source object::

        >>> src = sim.source.source_templates.empty_sky()
        >>> opt.observe(src)
        >>> hdus = opt.readout()

    If no Source is specified, an empty field is observe, so that the
    following is equivalent to the commands above::
        >>> opt.observe()
        >>> hdus = opt.readout()

    List the effects modelled in an OpticalTrain::

        >>> print(opt.effects)

    Effects can be accessed by using the name of the effect::

        >>> print(opt["dark_current"])

    To include or exclude an effect during a simulation run, use the
    ``.include`` attribute of the effect::

         >>> opt["dark_current"].include = False

    Data used by an Effect object is contained in the ``.data`` attribute,
    while other information is contained in the ``.meta`` attribute::

        >>> opt["dark_current"].data
        >>> opt["dark_current"].meta

    Meta data values can be set by either using the ``.meta`` attribute
    directly::

        >>> opt["dark_current"].meta["value"] = 0.5

    or by passing a dictionary (with one or multiple entries) to the
    OpticalTrain object::

        >>> opt["dark_current"] = {"value": 0.75, "dit": 30}

    """

    @top_level_catch
    def __init__(self, cmds=None):
        self.cmds = cmds
        self._description = self.__repr__()
        self.optics_manager = None
        self.fov_manager = None
        self.image_planes = []
        self.detector_managers = []
        self.yaml_dicts = None
        self._last_source = None
        self._readout_id = 0

        if cmds is not None:
            self.load(cmds)

    def load(self, user_commands):
        """
        (Re)Load an OpticalTrain with a new set of UserCommands.

        Parameters
        ----------
        user_commands : UserCommands or str

        """
        if isinstance(user_commands, str):
            user_commands = UserCommands(use_instrument=user_commands)
        elif isinstance(user_commands, UserCommands):
            user_commands = copy.deepcopy(user_commands)
        else:
            raise ValueError("user_commands must be a UserCommands or str object "
                             f"but is {type(user_commands)}")

        # HACK: The clean solution of new_child doesn't work because all the
        #       extras of UserCommands like yaml_dicts and so on are not
        #       passed to the new instance. Ideally fix that at some point...
        # self.cmds = user_commands.new_child(RecursiveNestedMapping(title="CurrObs"))
        self.cmds = user_commands
        self.cmds.maps = [RecursiveNestedMapping(title="CurrObs"), *self.cmds.maps]
        # FIXME: Setting rc.__currsys__ to user_commands causes many problems:
        #        UserCommands used NestedMapping internally, but is itself not
        #        an instance or subclas thereof. So rc.__currsys__ actually
        #        changes type as a result of this line. On one hand, some other
        #        code relies on this change, i.e. uses attributes from
        #        UserCommands via rc.__currsys__, but on the other hand some
        #        tests (now with proper patching) fail because of this type
        #        change. THIS IS A PROBLEM!
        # NOTE: All tests pass without setting rc.__currsys__ to user_commands.
        #       Nevertheless, I'm a bit reluctant to removing this code just
        #       yet. So it is commented out.
        # rc.__currsys__ = user_commands

        # Guard against all-empty cmds to avoid cryptical error downstream.
        if all(len(m) == 0 for m in self.cmds.maps):
            raise ValueError("Empty cmds, cannot construct OpticalTrain.")

        # Guard against empty yamls to avoid cryptical error downstream.
        if not self.cmds.yaml_dicts:
            raise ValueError("No YAMLS found, cannot construct OpticalTrain.")

        self.yaml_dicts = self.cmds.yaml_dicts
        self.optics_manager = OpticsManager(self.yaml_dicts, self.cmds)
        self.update(migrate_cmds=True)

    def update(self, migrate_cmds=False, **kwargs):
        """
        Update the user-defined parameters and remake main internal classes.

        Parameters
        ----------
        kwargs : expanded dict
            Any keyword-value pairs from a config file

        """
        self.optics_manager.update(**kwargs)
        opt_man = self.optics_manager

        self.fov_manager = FOVManager(
            opt_man.fov_setup_effects,
            cmds=self.cmds,
            **kwargs
        )
        self.image_planes = [
            ImagePlane(hdr, self.cmds, **kwargs)
            for hdr in opt_man.image_plane_headers
        ]
        self.detector_managers = [
            DetectorManager(det_list, cmds=self.cmds, **kwargs)
            for det_list in opt_man.detector_setup_effects
        ]

        # Move everything from CurrObs to CurrSys, so CurrObs is clean for
        # .observe and .readout. This is necessary because the setup and
        # potentially update need to be able to modify CurrSys, but CurrObs
        # needs to be already in place when creating e.g. effects, but then
        # writing to cmds will always use CurrObs. The alternative is manually
        # writing to cmds.maps[1] everywhere during the setup, but these two
        # lines make sure everything does indeed go into CurrSys.
        # self.cmds.maps[1].update(self.cmds.maps[0])
        # self.cmds.maps[0].clear()
        # HACK: recursive_update is needed to avoid overwriting emtpy !ABCs
        # TODO: or is it??
        if migrate_cmds:
            self.cmds.maps[1].dic = recursive_update(self.cmds.maps[1].dic, self.cmds.maps[0].dic)
        self.cmds.maps[0].dic.clear()

    def trace_detector_coordinates(self, *, xi=0 * u.arcsec,
                                   wavelengths: Mapping | None = None) -> QTable:
        """
        Return slit coordinates mapped to the configured detector pixels.

        With no wavelength argument, every configured spectral trace is
        sampled at the wavelengths carried by its trace table. A mapping of
        ``trace_id: wavelength`` arrays can instead request exact wavelengths.
        The returned detector coordinates are continuous, zero-origin pixel
        coordinates after the trace subimage rasterization, image-plane
        placement, and detector extraction conventions. No observation or
        image data are used.

        This deliberately supports one spectral-trace list and one detector
        per image plane. More complicated detector arrangements need an
        explicit definition of which output image owns each trace coordinate.
        """
        from ..effects import (
            DetectorModePropertiesSetter,
            ImagePlaneBackground,
            Rotate90CCD,
            SelectorWheel,
            SpectralTraceList,
        )
        from ..effects.electronic.pixels import BinnedImageBase

        trace_lists = self.optics_manager.get_all(SpectralTraceList)
        if len(trace_lists) != 1:
            raise NotImplementedError(
                "Trace detector coordinates require exactly one active "
                "SpectralTraceList.")
        trace_list = trace_lists[0]

        for effect in self.optics_manager.image_plane_effects:
            selected = effect.wheel_effects.values() if isinstance(effect, SelectorWheel) else [effect]
            if any(not isinstance(item, ImagePlaneBackground) for item in selected):
                raise NotImplementedError('Trace detector coordinates only support additive image-plane backgrounds.')
        for effect in self.optics_manager.detector_array_effects:
            if not isinstance(effect, DetectorModePropertiesSetter):
                raise NotImplementedError('Trace detector coordinates do not yet support this detector-array effect.')
            # Mode setters update commands, not image coordinates. Resolve an
            # explicit mode before reading any binning command it can set.
            if from_currsys(effect.mode_key, self.cmds) == 'auto':
                raise NotImplementedError('Select an explicit detector readout mode before inspecting coordinates.')
            effect.apply_to(None)

        fovs_by_trace = {}
        for fov in self.fov_manager.fovs:
            fovs_by_trace.setdefault(str(fov.trace_id), []).append(fov)

        if wavelengths is None:
            trace_ids = list(trace_list.spectral_traces)
        else:
            trace_ids = list(wavelengths)

        xi_arcsec = np.atleast_1d(
            u.Quantity(xi, u.arcsec).to_value(u.arcsec))

        def image_origin_on_canvas(shape_xy, image_wcs, canvas_wcs):
            """Mirror add_imagehdu_to_imagehdu and overlay_image placement."""
            image_center = (shape_xy - 1) / 2
            world_center = image_wcs.all_pix2world([image_center], 0)
            canvas_center = canvas_wcs.all_world2pix(world_center, 0)[0]
            return (
                np.ceil(np.round(canvas_center, 10)).astype(int)
                - shape_xy // 2
            )

        columns = {
            "readout_index": [],
            "image_plane_id": [],
            "detector_id": [],
            "trace_id": [],
            "wavelength": [],
            "xi": [],
            "detector_x": [],
            "detector_y": [],
        }

        for trace_id in trace_ids:
            trace = trace_list.spectral_traces[trace_id]
            trace_name = str(trace_id)
            trace_fovs = fovs_by_trace[trace_name]
            image_plane_id = int(trace.meta["image_plane_id"])

            if any(int(fov.meta["image_plane_id"]) != image_plane_id
                   for fov in trace_fovs):
                raise NotImplementedError(
                    f"Trace {trace_name!r} has conflicting image-plane "
                    "assignments.")

            image_planes = [
                image_plane for image_plane in self.image_planes
                if image_plane.id == image_plane_id
            ]
            detector_managers = [
                (index, manager)
                for index, manager in enumerate(self.detector_managers)
                if int(from_currsys(
                    manager._detector_list.image_plane_id, self.cmds
                )) == image_plane_id
            ]
            if len(image_planes) != 1 or len(detector_managers) != 1:
                raise NotImplementedError(
                    f"Trace {trace_name!r} does not resolve to exactly one "
                    "image plane and detector manager.")

            readout_index, detector_manager = detector_managers[0]
            image_plane = image_planes[0]
            if len(detector_manager) != 1:
                raise NotImplementedError(
                    "Trace detector coordinates do not yet support multiple "
                    "detectors on one image plane.")
            detector = detector_manager[0]
            detector_id = detector.header["ID"]
            readout_header = detector._native_header.copy()

            for effect in self.optics_manager.detector_effects:
                selected_effect = effect.get_effect(detector_id) if isinstance(effect, SelectorWheel) else effect
                if selected_effect is None:
                    continue
                if isinstance(selected_effect, BinnedImageBase):
                    readout_header = selected_effect.bin_header(readout_header)
                if isinstance(selected_effect, Rotate90CCD):
                    rotations = from_currsys(
                        selected_effect.meta["rotations"], self.cmds)
                    if rotations % 4:
                        raise NotImplementedError(
                            "Trace detector coordinates do not yet support "
                            "post-extraction detector rotation.")

            detector_wcs = WCS(detector._native_header, key="D")
            image_plane_wcs = image_plane._det_wcs
            if (
                not np.allclose(image_plane_wcs.wcs.pc, np.eye(2))
                or not np.allclose(detector_wcs.wcs.pc, np.eye(2))
            ):
                raise NotImplementedError(
                    "Trace detector coordinates do not yet support rotated "
                    "or sheared image-plane or detector D-WCSs.")
            if not np.allclose(
                np.abs(image_plane_wcs.wcs.cdelt[:2]),
                np.abs(detector_wcs.wcs.cdelt[:2]),
            ):
                raise NotImplementedError(
                    "Trace detector coordinates do not yet support rescaling "
                    "between image-plane and detector pixels.")

            image_plane_shape = np.array([
                image_plane.header["NAXIS1"],
                image_plane.header["NAXIS2"],
            ])
            detector_origin = image_origin_on_canvas(
                image_plane_shape, image_plane_wcs, detector_wcs)

            if wavelengths is None:
                wave_um = np.unique(
                    u.Quantity(
                        trace.table[trace.meta["wave_colname"]]
                    ).to_value(u.um)
                )
                configured = np.zeros(wave_um.shape, dtype=bool)
                for fov in trace_fovs:
                    wave_min = fov.meta["wave_min"].to_value(u.um)
                    wave_max = fov.meta["wave_max"].to_value(u.um)
                    configured |= (
                        (wave_um >= wave_min) & (wave_um <= wave_max))
                wave_um = wave_um[configured]
            else:
                wave_um = np.atleast_1d(
                    u.Quantity(wavelengths[trace_id]).to_value(u.um))

            wave_grid, xi_grid = np.meshgrid(wave_um, xi_arcsec)
            wave_flat = wave_grid.ravel()
            xi_flat = xi_grid.ravel()
            x_pix = np.empty(wave_flat.shape)
            y_pix = np.empty(wave_flat.shape)
            mapped_points = np.zeros(wave_flat.shape, dtype=bool)

            for fov in trace_fovs:
                wave_min = fov.meta["wave_min"].to_value(u.um)
                wave_max = fov.meta["wave_max"].to_value(u.um)
                in_fov = (
                    (
                        (wave_flat >= wave_min)
                        | np.isclose(
                            wave_flat, wave_min, rtol=1e-12, atol=0)
                    )
                    & (
                        (wave_flat <= wave_max)
                        | np.isclose(
                            wave_flat, wave_max, rtol=1e-12, atol=0)
                    )
                )
                if not np.any(in_fov):
                    continue
                if np.any(mapped_points & in_fov):
                    raise NotImplementedError(
                        f"Trace {trace_name!r} has overlapping FOV wavelength "
                        "ranges.")

                _, _, x_grid_mm, y_grid_mm, subimage_wcs = trace._focal_plane_grid(fov)
                if (
                    not np.allclose(subimage_wcs.wcs.pc, np.eye(2))
                    or not np.allclose(
                        np.abs(subimage_wcs.wcs.cdelt[:2]),
                        np.abs(image_plane_wcs.wcs.cdelt[:2]),
                    )
                ):
                    raise NotImplementedError(
                        "Trace detector coordinates do not yet support "
                        "rescaling or reorientation during image-plane "
                        "placement.")

                x_mm = trace.xilam2x(xi_flat[in_fov], wave_flat[in_fov])
                y_mm = trace.xilam2y(xi_flat[in_fov], wave_flat[in_fov])
                subimage_x, subimage_y = subimage_wcs.all_world2pix(x_mm, y_mm, 0)

                subimage_shape = np.array(
                    [len(x_grid_mm), len(y_grid_mm)])
                image_plane_origin = image_origin_on_canvas(
                    subimage_shape, subimage_wcs, image_plane_wcs)
                image_plane_x = image_plane_origin[0] + subimage_x
                image_plane_y = image_plane_origin[1] + subimage_y

                x_pix[in_fov] = detector_origin[0] + image_plane_x
                y_pix[in_fov] = detector_origin[1] + image_plane_y
                mapped_points |= in_fov

            if not np.all(mapped_points):
                raise ValueError(
                    f"Requested wavelengths for trace {trace_name!r} "
                    "do not map through a configured FOV.")

            focal_x, focal_y = detector_wcs.all_pix2world(x_pix, y_pix, 0)
            x_pix, y_pix = WCS(readout_header, key='D').all_world2pix(focal_x, focal_y, 0)
            point_count = wave_grid.size
            columns["readout_index"].extend(
                np.full(point_count, readout_index))
            columns["image_plane_id"].extend(
                np.full(point_count, image_plane_id))
            columns["detector_id"].extend(
                np.full(point_count, detector_id))
            columns["trace_id"].extend(
                np.full(point_count, trace_name))
            columns["wavelength"].extend(wave_grid.ravel())
            columns["xi"].extend(xi_grid.ravel())
            columns["detector_x"].extend(x_pix)
            columns["detector_y"].extend(y_pix)

        table = QTable()
        table["readout_index"] = columns["readout_index"]
        table["image_plane_id"] = columns["image_plane_id"]
        table["detector_id"] = columns["detector_id"]
        table["trace_id"] = columns["trace_id"]
        table["wavelength"] = columns["wavelength"] * u.um
        table["xi"] = columns["xi"] * u.arcsec
        table["detector_x"] = columns["detector_x"] * u.pixel
        table["detector_y"] = columns["detector_y"] * u.pixel
        return table

    @top_level_catch
    def observe(self, orig_source=None, update=True, **kwargs):
        """
        Main controlling method for observing ``Source`` objects.

        Parameters
        ----------
        orig_source : Source
        update : bool
            Reload optical system
        kwargs : expanded dict
            Any keyword-value pairs from a config file

        Notes
        -----
        When orig_source is None (e.g. when writing opt.observe()), an
        empty field is observed (internally created with empty_sky()).

        How the list of Effects is split between the 5 main tasks:

        - Make a FOV list - z_order = 0..99
        - Make a image plane - z_order = 100..199
        - Apply Source altering effects - z_order = 200..299
        - Apply FOV specific (3D) effects - z_order = 300..399
        - Apply FOV-independent (2D) effects - z_order = 400..499
        - [Apply detector plane (0D, 2D) effects - z_order = 500..599]

        .. todo:: List is out of date - update

        """
        if update:
            self.update(**kwargs)
        self._readout_id = 0

        if self.cmds["!SIM.random.seed"] is None:
            # Concretize seed from system entropy to keep constant during the
            # observe run. For multiple observations, this is cleared above.
            self.cmds["!SIM.random.seed"] = np.random.SeedSequence().entropy
        elif self.cmds["!SIM.random.seed"] in {"None", "none", "NONE"}:
            logger.warning(
                "Got random seed '%s' (a string, not None!), use NULL in yaml",
                self.cmds["!SIM.random.seed"]
            )
            self.cmds["!SIM.random.seed"] = np.random.SeedSequence().entropy

        # self.set_focus(**kwargs)    # put focus back on current instrument package

        # Make a copy of the Source and prepare for observation (convert to
        # internally used units, sample to internal wavelength grid)
        if orig_source is None:
            source = empty_sky()
            logger.info("Observing empty field")
        else:
            source = orig_source.make_copy()
        source = self.prepare_source(source)

        # [1D - transmission curves]
        for effect in self.optics_manager.source_effects:
            source = effect.apply_to(source)

        # This source is a private copy, after all source effects. FOV extraction
        # copies native-sampled spectra before FOV effects can modify them. Evaluate
        # native sampling once for this observation; retain no cache on the
        # source, spectrum, FOV or train, and never reuse it on the next call.
        spectral_wavesets = {}
        for spectrum in source.spectra.values():
            waveset = spectrum.waveset
            spectral_wavesets[spectrum] = None if waveset is None else waveset.to_value(u.AA)

        # [3D - Atmospheric shifts, PSF, NCPAs, Grating shift/distortion]

        # START OF MULTIPROCESSING
        if USE_MULTIPROCESSING:

            fovs = self.fov_manager.fovs
            fov_effects = self.optics_manager.fov_effects
            hdu_type = "cube" if self.fov_manager.is_spectroscope else "image"

            with mp.Pool(processes=N_PROCESSES) as pool:
                fovs = pool.starmap(extract_source,
                                    zip(fovs, [source] * len(fovs)))

            with mp.Pool(processes=N_PROCESSES) as pool:
                fovs = pool.starmap(view_fov,
                                    zip(fovs, [hdu_type] * len(fovs)))

            with mp.Pool(processes=N_PROCESSES) as pool:
                fovs = pool.starmap(apply_fov_effects,
                                    zip(fovs, [fov_effects] * len(fovs)))

        # OLD SINGLE CORE CODE
        else:

            fovs = self.fov_manager.fovs
            nobar = len(fovs) <= 1
            for fov in tqdm(fovs, desc=" FOVs", position=0, disable=nobar):
                # FOVManager can reuse a FOV whose previous observation is
                # already mapped to detector coordinates. Prepare new sky
                # illumination from its physical header, not that old HDU.
                fov.hdu = None
                foveffs = []
                for effect in self.optics_manager.fov_effects:
                    if isinstance(effect, SelectorWheel):
                        effect = effect.get_effect(fov.meta[effect.meta['selector_key']])
                    if isinstance(effect, FilterWheelBase):
                        effect = effect.current_filter
                    if effect is not None:
                        foveffs.append(effect)

                for effect in foveffs:
                    # Slit shear needs spectral support before extraction.
                    # Fibre bundles have no single FOV trace and map separately.
                    if isinstance(effect, SpectralTraceList) and isinstance(fov, FieldOfView3D) and fov.trace_id is not None:
                        fov.meta['spectral_padding'] = effect.spectral_traces[fov.trace_id].spectral_padding(fov)

                # Render at the PSF stage when preceding effects only multiply
                # spectra. Point positions then stay continuous, and resolved
                # images are convolved before clipping to the aperture.
                psf_indices = [i for i, effect in enumerate(foveffs) if isinstance(effect, MoffatPSF)]
                defer_render = isinstance(fov, FieldOfView3D) and len(psf_indices) == 1
                if defer_render:
                    psf_index = psf_indices[0]
                    defer_render = all(isinstance(effect, (TERCurve, SpectralEfficiency, EchelleSpectralEfficiency, TaperedQuantumEfficiency)) for effect in foveffs[:psf_index])
                spatial_shifts = []
                if defer_render:
                    for effect in foveffs[psf_index + 1:]:
                        if isinstance(effect, ShiftFoV3D):
                            spatial_shifts.append(effect)
                        elif not isinstance(effect, (TERCurve, DichroicTree, SpectralEfficiency, EchelleSpectralEfficiency, TaperedQuantumEfficiency)):
                            break
                    # Wavelength-only multipliers commute with a displacement
                    # within each spectral plane; stop before other spatial effects.
                    point_sources = not (source.image_fields or source.cube_fields)
                    spatial_header = foveffs[psf_index].padded_header(fov, point_sources=point_sources)
                    for effect in spatial_shifts:
                        spatial_header = effect._padded_header(spatial_header, effect.get_shifts(fov))
                    fov.extract_from(source, spatial_header=spatial_header, _spectral_wavesets=spectral_wavesets)
                else:
                    fov.extract_from(source, _spectral_wavesets=spectral_wavesets)
                    fov.view()
                nobar = len(foveffs) <= 1
                for effect in tqdm(foveffs, disable=nobar,
                                   desc=" FOV effects", position=1):#, leave=False):
                    if effect in spatial_shifts:
                        continue  # Applied to incoming illumination at the PSF stage.
                    if defer_render and effect is foveffs[psf_index] and spatial_shifts:
                        effect._apply_convolution(fov, spatial_shifts=spatial_shifts)
                    else:
                        fov = effect.apply_to(fov)

                if fov.hdu is None:
                    logger.info("  Skipping empty FOV")
                    continue

                if self.cmds.get("!INST.flatten", True):
                    fov.flatten()
                    self.image_planes[fov.image_plane_id].add(fov.hdu, wcs_suffix="D")
                else:  # cube output
                    self.image_planes[fov.image_plane_id].add(fov.hdu, wcs_suffix="D")
                    # HACK: to get sky WCS from FOV into image plane...
                    self.image_planes[fov.image_plane_id].header.update(
                        WCS(fov.hdu).to_header())

                # Mapping is complete. Keep the rendered FOV and trace's
                # rectified spectrum, without retaining every input cube.
                fov.cube = None

                # ..todo: finish off the multiple image plane stuff

        # END OF MULTIPROCESSING

        # [2D - Vibration, flat fielding, chopping+nodding]
        impeffs = self.optics_manager.image_plane_effects
        nobar = len(impeffs) <= 1
        for effect in tqdm(impeffs, disable=nobar,
                           desc=" Image Plane effects"):
            for ii, image_plane in enumerate(self.image_planes):
                self.image_planes[ii] = effect.apply_to(image_plane)

        self._last_fovs = fovs
        self._last_source = source

    def prepare_source(self, source):
        """
        Prepare source for observation.

        The method is currently applied to cube fields only.
        The source data are converted to internally used units (PHOTLAM).
        The source data are interpolated to the waveset used by the FieldOfView
        This is necessary when the source data are sampled on a coarser grid
        than used internally, or if the source data are sampled on irregular
        wavelengths.
        For cube fields, the method assumes that the wavelengths at which the
        cube is sampled is provided explicitely as attribute `wave` of the cube
        ImageHDU.
        """
        # Convert to PHOTLAM per arcsec2
        # TODO: this is not sufficiently general
        # TODO: Maybe move this to source_fields??

        for cube in source.cube_fields:
            header, data, wave = cube.header, cube.data, cube.waveset

            # Need to check whether BUNIT is per arcsec2 or per pixel
            inunit = u.Unit(header["BUNIT"])
            data = data.astype(np.float64, copy=False) * inunit
            factor = 1
            for base, power in zip(inunit.bases, inunit.powers):
                if (base**power).is_equivalent(u.arcsec**(-2)):
                    conversion = (base**power).to(u.arcsec**(-2)) / base**power
                    data *= conversion
                    factor = u.arcsec**(-2)

            data = data.to(PHOTLAM,
                           equivalencies=u.spectral_density(wave[:, None, None]))

            if factor == 1:    # Normalise to 1 arcsec2 if not a spatial density
                # ..todo: lower needed because "DEG" is not understood, this is ugly
                pixarea = (header["CDELT1"] * u.Unit(header["CUNIT1"].lower()) *
                           header["CDELT2"] * u.Unit(header["CUNIT2"].lower())).to(u.arcsec**2)
                pixarea = np.abs(pixarea)     # CDELTi can be negative, pixarea cannot
                data = data / pixarea.value    # cube is per arcsec2

            data = (data * factor).value

            cube.header["BUNIT"] = "photlam/arcsec2"    # ..todo: make this more explicit?

            # The imageplane_utils like to have the spatial WCS in units of "deg". Ensure
            # that the cube is passed on accordingly
            cube.header["CDELT1"] = header["CDELT1"] * u.Unit(header["CUNIT1"].lower()).to(u.deg)
            cube.header["CDELT2"] = header["CDELT2"] * u.Unit(header["CUNIT2"].lower()).to(u.deg)
            cube.header["CUNIT1"] = "deg"
            cube.header["CUNIT2"] = "deg"

            # Keep the native wavelength samples until the spatial FOV is cut.
            # Rebinning the entire source here loses narrow lines and expands
            # resolved cubes over the full instrument band unnecessarily.
            cube.data = data
            cube.wcs = WCS(cube.field)

        return source

    @top_level_catch
    def readout(
        self,
        filename: Path | str | None = None,
        reset: bool = True,
        roid: int | None = None,
        **kwargs
    ) -> list[HDUList]:
        """
        Produce detector readouts for the observed image.

        Parameters
        ----------
        filename : Path | str | None, optional
            Full path or local name for to store the resulting FITS file.
            If None (the default), the function still returns the result, but
            will not automatically save it to disk.
        reset : bool, optional
            If true (the default), do not keep any parameters set during any
            previous readout. This is usually what you want.
        roid : int or None, optional
            Readout ID for reproducible noise. By default, successive calls
            use 0, 1, 2, ... within each observation. Supply an explicit ID to
            replay that noise realization with the same root seed.
        **kwargs :
            Any other kwargs passed to the detector effects.

        Returns
        -------
        list[HDUList]
            List of HDUList objects.

        """
        if reset:
            self.cmds.clear()
        if roid is None:
            roid = self._readout_id
            self._readout_id += 1

        # Hack to make sure AutoExposure and Quantization work properly.
        # Should probably be removed once #428 is fixed properly.
        # if kwargs.get('exptime', None) is not None:
        #     # self.cmds.pop("!OBS.dit", None)
        #     # self.cmds.pop("!OBS.ndit", None)
        #     self.cmds["!OBS.dit"] = None
        #     self.cmds["!OBS.ndit"] = None
        # TODO: This is still hacky but seems to work for now...
        params = {"exptime": None, "roid": roid}
        params.update({key.removeprefix('!OBS.'): value for key, value in kwargs.items()})
        if params["exptime"] is not None and params.get("dit") is None:
            params.update(
                dit=None,
                ndit=None,
            )
        else:
            params.update(
                exptime=params.get("exptime") or self.cmds.get("!OBS.exptime"),
                dit=params.get("dit") or self.cmds.get("!OBS.dit"),
                ndit=params.get("ndit") or self.cmds.get("!OBS.ndit"),
            )

        for key, value in params.items():
            if is_bangkey(key):
                logger.debug("%s: %s given in kwargs", key, value)
                self.cmds[key] = value
                continue
            # otherwise fallback to !OBS
            logger.debug("%s: %s given in kwargs, put in !OBS", key, value)
            self.cmds[f"!OBS.{key}"] = value

        hduls = []
        for i, detector_manager in enumerate(self.detector_managers):
            hdul = detector_manager.readout(
                self.image_planes,
                self.optics_manager.detector_array_effects,
                self.optics_manager.detector_effects,
            )

            fits_effects = self.optics_manager.get_all(ExtraFitsKeywords)
            if len(fits_effects) > 0:
                for effect in fits_effects:
                    hdul = effect.apply_to(hdul, optical_train=self)
            else:
                try:
                    hdul = self.write_header(hdul, detector_manager)
                except Exception:
                    logger.exception(
                        "Header update failed, data will be saved with "
                        "incomplete header. See stack trace for details."
                    )

            if filename is not None:
                if len(self.detector_managers) == 1:
                    fname = filename
                else:
                    if isinstance(filename, str):
                        fname = f"{i}_{filename}"
                    if isinstance(filename, Path):
                        fname = filename.parent / f"{i}_{filename.name}"

                hdul.writeto(fname, overwrite=True)

            hduls.append(hdul)

        # Create a copy of the readouts, because they are still referenced here.
        # That is, self.detector_managers[0][0].hdu is hdul[1].
        # Subsequent readouts would therefor overwrite earlier readouts
        # without the copy.
        # It is necessary to keep the detector readouts in memory, because
        # this makes it possible to add support for up-the-ramp sampling,
        # that is, subsequent readouts without reseting the detector inbetween.
        return copy.deepcopy(hduls)

    def write_header(self, hdulist, detector_array):
        """Write minimal header to output HDUL.

        This method exists as a fallback of last resort, if no FITS header
        effect is present in the optical train. It should not be expanded,
        instead any other keywords required in the output should be included
        via those effects. This should only be called for very minimal
        instrument setups or during testing and debugging, where most effects
        are switched off.
        """
        # Primary HDU
        pheader = hdulist[0].header
        pheader["DATE"] = datetime.now().isoformat(timespec="seconds")
        pheader["ORIGIN"] = f"Scopesim {__version__}"

        # These are always present in cmds, so should be save
        pheader["INSTRUME"] = from_currsys("!OBS.instrument", self.cmds)
        pheader["INSTMODE"] = ", ".join(from_currsys("!OBS.modes", self.cmds))

        # Image HDUs
        for hdu in hdulist[1:]:
            # Arm-specific integration effects already recorded their actual
            # exposure. Global command values are only for minimal setups.
            for key in ('EXPTIME', 'DIT', 'NDIT'):
                if key not in hdu.header:
                    hdu.header[key] = from_currsys(f'!OBS.{key.lower()}', self.cmds)
            hdu.header['DETECTOR'] = detector_array._detector_list.display_name
        return hdulist

    def shutdown(self):
        """
        Shut down the instrument.

        This method closes all open file handles and should be called when the
        optical train is no longer needed.
        """
        for effect_name in self.effects["name"]:
            try:
                self[effect_name]._file.close()
            except AttributeError:
                pass

        self._description = "The instrument has been shut down."


    @property
    def effects(self):
        return self.optics_manager.list_effects()

    @property
    def effect_order(self):
        return self.optics_manager.effect_order_table()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.cmds!r})"

    def __str__(self):
        return self._description

    def _repr_pretty_(self, p, cycle):
        """For ipython."""
        if cycle:
            p.text(f"{self.__class__.__name__}(...)")
        else:
            p.text(f"{self.__class__.__name__} ")
            p.text(f"for {self.cmds['!OBS.instrument']} ")
            p.text(f"@ {self.cmds['!TEL.telescope']}:")
            p.breakable()
            p.text("UserCommands:")
            p.breakable()
            p.pretty(self.cmds)
            p.breakable()
            p.text("OpticalElements:")
            with p.indent(2):
                for item in self:
                    p.breakable()
                    p.pretty(item)
            p.breakable()
            p.text("DetectorManagers:")
            with p.indent(2):
                for item in self.detector_managers:
                    p.breakable()
                    p.pretty(item)
            p.breakable()
            p.text("Effects:")
            p.breakable()
            with p.indent(2):
                p.pretty(self.effects)

    def _repr_html_(self) -> str:
        """For notebooks."""
        html = (
            "<div>"
            f"<h4>{self.__class__.__name__} "
            f"for {self.cmds['!OBS.instrument']} "
            f"@ {self.cmds['!TEL.telescope']}</h4>"
            f"<div><h5>Settings</h5>{self.cmds._repr_html_()}</div>"
            f"<div><h5>Effects</h5>{self.effects._repr_html_()}</div>"
            "</div>"
        )
        return html

    def __getitem__(self, item):
        return self.optics_manager[item]

    def __setitem__(self, key, value):
        self.optics_manager[key] = value

    def __contains__(self, key):
        return key in self.optics_manager

    def report(self):
        pass
        # user commands report
        #   package dependencies
        #   modes names
        #   default modes
        #   yaml hierarchy
        #
        # optics_manager
        #   derived properties
        #   system transmission curve
        #   list of effects
        #
        # etc
        #   limiting magnitudes
        #
