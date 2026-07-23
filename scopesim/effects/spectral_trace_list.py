# -*- coding: utf-8 -*-
"""
Effect for mapping spectral cubes to the detector plane.

The Effect is called `SpectralTraceList`, it applies a list of
`spectral_trace_list_utils.SpectralTrace` objects to a `FieldOfView`.
"""
from itertools import cycle
from pathlib import Path
from typing import ClassVar

from tqdm.auto import tqdm

import numpy as np
from astropy.io import fits
from astropy.table import Table
import astropy.units as u

from .effects import Effect
from .ter_curves import FilterCurve
from .spectral_trace_list_utils import SpectralTrace, make_image_interpolations
from ..optics.image_plane_utils import header_from_list_of_xy
from ..optics.fov import FieldOfView
from ..optics.fov_volume_list import FovVolumeList
from ..utils import from_currsys, check_keys, figure_factory, get_logger
from .data_container import DataContainer
from ..optics import echelle

logger = get_logger(__name__)


class SpectralTraceList(Effect):
    """
    List of spectral trace geometries for the detector plane.

    Should work in concert with an ApertureList (or ApertureMask) object and a
    DetectorList object

    Spectral trace patterns are to be kept in a ``fits.HDUList`` with one or
    more ``fits.BinTableHDU`` extensions, each one describing the geometry of a
    single trace. The first extension should be a ``BinTableHDU`` connecting
    the traces to the correct ``Aperture`` and ``ImagePlane`` objects.

    The ``fits.HDUList`` objects can be loaded using one of these two keywords:

    - ``filename``: for on disk FITS files, or
    - ``hdulist``: for in-memory ``fits.HDUList`` objects

    The format and contents of the extensions in the HDUList (FITS file) object
    is listed below

    Input Data Format
    -----------------
    A trace list FITS file needs the following extensions:

    - 0 : PrimaryHDU [header]
    - 1 : BinTableHDU [header, data] : Overview table of all traces
    - 2..N : BinTableHDU [header, data] : Trace tables. One per spectral trace

    EXT 0 : PrimaryHDU
    ++++++++++++++++++
    Required Header Keywords:

    - ECAT : int : Extension number of overview table. Normally 1
    - EDATA : int : Extension number of first Trace table. Normally 2

    No data is required in this extension

    EXT 1 : BinTableHDU : Overview of traces
    ++++++++++++++++++++++++++++++++++++++++
    No special header keywords are required in this extension

    Required Table columns:

    - description : str : identifier of each trace
    - extension_id : int : which extension is each trace in
    - aperture_id : int : which aperture matches this trace (e.g. MOS / IFU)
    - image_plane_id : int : on which image plane is this trace projected

    EXT 2 : BinTableHDU : Individual traces
    +++++++++++++++++++++++++++++++++++++++
    Required header keywords:
    - EXTNAME : must be identical to the `description` in EXT 1

    Recommended header keywords:
    - DISPDIR : "x" or "y" : dispersion axis. If not present, Scopesim tries
      to determine this automatically; this may be unreliable in some cases.

    Required Table columns:
    - wavelength : float : [um] : wavelength of monochromatic aperture image
    - s : float : [arcsec] : position along aperture perpendicular to trace
    - x : float : [mm] : x position of aperture image on focal plane
    - y : float : [mm] : y position of aperture image on focal plane

    .. versionchanged:: 0.11.3

       Added support for slit offset.

    """

    _class_params = {
        "x_colname": "x",
        "y_colname": "y",
        "s_colname": "s",
        "wave_colname": "wavelength",
        "col_number_start": 0,
        "center_on_wave_mid": False,
        "dwave": 0.002,  # [um] for finding best fit dispersion
        "invalid_value": None,  # for dodgy trace file values
    }
    z_order: ClassVar[tuple[int, ...]] = (70, 270, 670)
    report_plot_include: ClassVar[bool] = True
    report_table_include: ClassVar[bool] = False

    def __init__(self, cmds=None, **kwargs):
        super().__init__(cmds=cmds, **kwargs)

        if "hdulist" in kwargs and isinstance(kwargs["hdulist"], fits.HDUList):
            self._file = kwargs["hdulist"]

        params = {
            "pixel_scale": "!INST.pixel_scale",  # [arcsec / pix]}
            "plate_scale": "!INST.plate_scale",  # [arcsec / mm]
            "spectral_bin_width": "!SIM.spectral.spectral_bin_width", # [um]
            "wave_min": "!SIM.spectral.wave_min",  # [um]
            "wave_mid": "!SIM.spectral.wave_mid",  # [um]
            "wave_max": "!SIM.spectral.wave_max",  # [um]
            "x_colname": "x",
            "y_colname": "y",
            "s_colname": "s",
            "offset_x": 0,     # [mm] in detector plane
            "offset_y": 0,     # [mm] in detector plane
            "wave_colname": "wavelength",
            "center_on_wave_mid": False,
            "dwave": 0.002,  # [um] for finding the best fit dispersion
            "invalid_value": None,  # for dodgy trace file values
        }
        self.meta.update(params)
        # Parameters that are specific to the subclass
        self.meta.update(self._class_params)
        self.meta.update(kwargs)

        if self._file is not None:
            self.make_spectral_traces()
            self.update_meta()

    def make_spectral_traces(self):
        """Return a dictionary of spectral traces read in from a file."""
        self.ext_data = self._file[0].header["EDATA"]
        self.ext_cat = self._file[0].header["ECAT"]
        self.catalog = Table(self._file[self.ext_cat].data)
        spec_traces = {}
        for row in self.catalog:
            params = {col: row[col] for col in row.colnames}
            params.update(self.meta)
            hdu = self._file[row["extension_id"]]
            spec_traces[row["description"]] = SpectralTrace(hdu, **params)

        self.spectral_traces = spec_traces

    def update_meta(self):
        """
        Update fov related meta values.

        The values describe the full extent of the spectral trace
        volume in wavelength and space
        """
        wlim, xlim, ylim = [], [], []
        for thetrace in self.spectral_traces.values():
            fov = thetrace.fov_grid()
            if "wave_min" in fov:
                wlim.extend([fov["wave_min"], fov["wave_max"]])
            if "x_min" in fov:
                xlim.extend([fov["x_min"], fov["x_max"]])
            if "y_min" in fov:
                ylim.extend([fov["y_min"], fov["y_max"]])

        if wlim:
            self.meta["wave_min"] = min(wlim)
            self.meta["wave_max"] = max(wlim)
        if xlim:
            self.meta["x_min"] = min(xlim)
            self.meta["x_max"] = max(xlim)
        if ylim:
            self.meta["y_min"] = min(ylim)
            self.meta["y_max"] = max(ylim)

    def apply_to(self, obj, **kwargs):
        """
        Interface between ``FieldOfView`` and ``SpectralTraceList``.

        This is called twice:
        1. During setup of the required FieldOfView objects, the
        SpectralTraceList is asked for the source space volumes that
        it requires (spatial limits and wavelength limits).
        2. During "observation" the method is passed a single FieldOfView
        object and applies the mapping to the image plane to it.
        The FieldOfView object is associated to one SpectralTrace from the
        list, identified by meta["trace_id"].
        """
        if isinstance(obj, FovVolumeList):
            logger.debug("%s applied to %s", self.display_name,
                         obj.__class__.__name__)
            # Setup of FieldOfView object
            # volumes = [spectral_trace.fov_grid()
            #            for spectral_trace in self.spectral_traces.values()]

            new_vols_list = []

            # for vol in volumes:
            for spt in self.spectral_traces.values():
                vol = spt.fov_grid()
                wave_edges = [vol["wave_min"], vol["wave_max"]]
                if "x_min" in vol:
                    x_edges = [vol["x_min"], vol["x_max"]]
                    y_edges = [vol["y_min"], vol["y_max"]]
                    extracted_vols = obj.extract(
                        axes=["wave", "x", "y"],
                        edges=(wave_edges, x_edges, y_edges),
                        aperture_id=vol["aperture_id"])
                else:
                    extracted_vols = obj.extract(
                        axes=["wave"],
                        edges=(wave_edges, ),
                        aperture_id=vol["aperture_id"])

                for ex_vol in extracted_vols:
                    ex_vol["meta"].update(vol)
                    ex_vol["meta"].pop("wave_min")
                    ex_vol["meta"].pop("wave_max")
                new_vols_list.extend(extracted_vols)

            obj.volumes = new_vols_list

        if isinstance(obj, FieldOfView):
            logger.debug("%s applied to %s", self.display_name,
                         obj.__class__.__name__)
            # Application to field of view
            if obj.hdu is not None and obj.hdu.header["NAXIS"] == 3:
                obj.cube = obj.hdu
            elif obj.hdu is not None and obj.hdu.header["NAXIS"] == 2:
                # todo: catch the case of obj.hdu.header["NAXIS"] == 2
                # for MAAT
                pass
            elif obj.hdu is None and obj.cube is None:
                logger.info("Making cube")
                obj.cube = obj.make_hdu()

            # Check whether an offset slit is used. If so, recompute spectral traces.
            offset_x = obj.cube.header["CRVAL1D"]
            offset_y = obj.cube.header["CRVAL2D"]
            if (offset_x != self.meta["offset_x"] or
                offset_y != self.meta["offset_y"]):
                logger.debug("Recomputing spectral traces for offset (%.1g, %.1g)",
                             offset_x, offset_y)
                self.meta["offset_x"] = offset_x
                self.meta["offset_y"] = offset_y
                self.make_spectral_traces()
                self.update_meta()

            spt = self.spectral_traces[obj.trace_id]
            obj.hdu = spt.map_spectra_to_focal_plane(obj)
            obj.image_plane_id = spt.meta["image_plane_id"]

        logger.debug("%s done", self.display_name)
        return obj

    @property
    def footprint(self):
        """Return the footprint of the entire SpectralTraceList."""
        xfoot, yfoot = [], []
        for spt in self.spectral_traces.values():
            xtrace, ytrace = spt.footprint()
            xfoot.extend(xtrace)
            yfoot.extend(ytrace)

        xfoot = [min(xfoot), max(xfoot), max(xfoot), min(xfoot)]
        yfoot = [min(yfoot), min(yfoot), max(yfoot), max(yfoot)]

        return xfoot, yfoot

    @property
    def image_plane_header(self):
        """Create and return header for the ImagePlane."""
        x, y = self.footprint
        pixel_scale = from_currsys(self.meta["pixel_scale"], self.cmds)
        hdr = header_from_list_of_xy(x, y, pixel_scale, "D")

        return hdr

    def rectify_traces(self, hdulist, xi_min=None, xi_max=None, interps=None,
                       **kwargs):
        """Create rectified 2D spectra for all traces in the list.

        This method creates an HDU list with one extension per spectral
        trace, i.e. it essentially treats all traces independently.
        For the case of an IFU where the traces correspond to spatial
        slices for the same wavelength range, use method `rectify_cube`
        (not yet implemented).

        Parameters
        ----------
        hdulist : str or fits.HDUList
           The result of scopesim readout()
        xi_min, xi_max : float [arcsec]
           Spatial limits of the slit on the sky. This should be taken
           from the header of the hdulist, but this is not yet provided by
           scopesim. For the time being, these limits *must* be provided by
           the user.
        interps :  list of interpolation functions
           If provided, there must be one for each image extension in
           `hdulist`. The functions go from pixels to the images and can be
           created with, e.g. ``RectBivariateSpline``.
        """
        try:
            inhdul = fits.open(hdulist)
        except TypeError:
            inhdul = hdulist

        # Crude attempt to get a useful wavelength range
        # Problematic because different instruments use different
        # keywords for the filter... We try to make it work for METIS
        # and MICADO for the time being.
        try:
            filter_name = from_currsys("!OBS.filter_name", self.cmds)
        except ValueError:
            filter_name = from_currsys("!OBS.filter_name_fw1", self.cmds)

        filtcurve = FilterCurve(
            filter_name=filter_name,
            filename_format=from_currsys("!INST.filter_file_format", self.cmds))
        filtwaves = filtcurve.table["wavelength"]
        filtwave = filtwaves[filtcurve.table["transmission"] > 0.01]
        wave_min, wave_max = min(filtwave), max(filtwave)
        logger.info(
            "Full wavelength range: %.02f .. %.02f um", wave_min, wave_max)

        if xi_min is None or xi_max is None:
            try:
                xi_min = inhdul[0].header["HIERARCH INS SLIT XIMIN"]
                xi_max = inhdul[0].header["HIERARCH INS SLIT XIMAX"]
                logger.info(
                    "Slit limits taken from header: %.02f .. %.02f arcsec",
                    xi_min, xi_max)
            except KeyError:
                logger.error(
                    "Spatial slit limits (in arcsec) must be provided:\n"
                    "- either as method parameters xi_min and xi_max\n"
                    "- or as header keywords HIERARCH INS SLIT XIMIN/XIMAX"
                )
                return None

        bin_width = kwargs.get("bin_width", None)

        if interps is None:
            logger.debug("Computing interpolation functions")
            interps = make_image_interpolations(hdulist)

        pdu = fits.PrimaryHDU()
        pdu.header["FILETYPE"] = "Rectified spectra"
        # pdu.header["INSTRUME"] = inhdul[0].header["HIERARCH ESO OBS INSTRUME"]
        # pdu.header["FILTER"] = from_currsys("!OBS.filter_name_fw1", self.cmds)
        outhdul = fits.HDUList([pdu])

        for i, trace_id in tqdm(enumerate(self.spectral_traces, start=1),
                                desc=" Traces", total=len(self.spectral_traces)):
            hdu = self[trace_id].rectify(hdulist,
                                         interps=interps,
                                         bin_width=bin_width,
                                         xi_min=xi_min, xi_max=xi_max,
                                         wave_min=wave_min, wave_max=wave_max)
            if hdu is not None:   # ..todo: rectify does not do that yet
                outhdul.append(hdu)
                outhdul[0].header[f"EXTNAME{i}"] = trace_id

        outhdul[0].header.update(inhdul[0].header)

        return outhdul

    def rectify_cube(self, hdulist):
        """Rectify traces and combine into a cube."""
        raise NotImplementedError()

    def plot(self, wave_min=None, wave_max=None, axes=None, **kwargs):
        """Plot every spectral trace in the spectral trace list.

        Parameters
        ----------
        wave_min : float, optional
            Minimum wavelength, if any. If None, value from_currsys is used.
        wave_max : float, optional
            Maximum wavelength, if any. If None, value from_currsys is used.
        axes : matplotlib axes, optional
            The axes object to use for the plot. If None (default), a new
            figure with one axes will be created.
        **kwargs : dict
            Any other parameters passed along to the plot method of the
            individual spectral traces.

        Returns
        -------
        fig : matplotlib figure
            DESCRIPTION.

        """
        if wave_min is None:
            wave_min = from_currsys("!SIM.spectral.wave_min", self.cmds)
        if wave_max is None:
            wave_max = from_currsys("!SIM.spectral.wave_max", self.cmds)

        if axes is None:
            fig, axes = figure_factory()
        else:
            fig = axes.figure

        if self.spectral_traces is not None:
            for spt, c in zip(self.spectral_traces.values(), cycle("rgbcymk")):
                spt.plot(wave_min, wave_max, c=c, axes=axes, **kwargs)

        return fig

    def __str__(self) -> str:
        msg = (f"{self.__class__.__name__}: \"{self.display_name}\": "
               f"{len(self.spectral_traces)} traces")
        return msg

    def __getitem__(self, item):
        return self.spectral_traces[item]

    def __setitem__(self, key, value):
        self.spectral_traces[key] = value


class SpectralTraceListWheel(Effect):
    """
    A Wheel-Effect object for selecting between multiple gratings/grisms.

    See ``SpectralTraceList`` for the trace file format description.

    Parameters
    ----------
    trace_list_names : list
        The list of unique identifiers in the trace filenames

    filename_format : str
        ``f-string`` that directs scopesim to the folder containing the trace
        files. This can be a ``!-string`` if the trace names are shared with
        other ``*Wheel`` effect objects (e.g. a ``FilterWheel``). See examples.

    current_trace_list : str
        default trace file to use

    kwargs : key-value pairs
        Addition keywords that are passed to the ``SpectralTraceList`` objects
        See SpectralTraceList docstring

    Examples
    --------
    A simplified YAML file example taken from the OSIRIS instrument package::

        alias: INST
        name: OSIRIS_LSS

        properties:
          decouple_detector_from_sky_headers: True
          grism_names:
            - R300B
            - R500B
            - R1000B
            - R2500V

        effects:
          - name: spectral_trace_wheel
            description: grism wheel contining spectral trace geometries
            class: SpectralTraceListWheel
            kwargs:
              current_trace_list: "!OBS.grating_name"
              filename_format: "traces/LSS_{}_TRACE.fits"
              trace_list_names: "!INST.grism_names"

          - name: grating_efficiency
            description: OSIRIS grating efficiency curves, piggybacking on FilterWheel
            class: FilterWheel
            kwargs:
              minimum_throughput: !!float 0.
              filename_format: "gratings/{}.txt"
              current_filter: "!OBS.grating_name"
              filter_names: "!INST.grism_names"

    """

    required_keys = {
        "trace_list_names",
        "filename_format",
        "current_trace_list",
    }
    z_order: ClassVar[tuple[int, ...]] = (70, 270, 670)
    report_plot_include: ClassVar[bool] = True
    report_table_include: ClassVar[bool] = True
    report_table_rounding: ClassVar[int] = 4
    _current_str = "current_trace_list"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        check_keys(kwargs, self.required_keys, action="error")

        params = {
            "path": "",
        }
        self.meta.update(params)
        self.meta.update(kwargs)

        path = self._get_path()
        self.trace_lists = {}
        if "name" in kwargs:
            kwargs.pop("name")
        for name in from_currsys(self.meta["trace_list_names"], self.cmds):
            fname = str(path).format(name)
            self.trace_lists[name] = SpectralTraceList(filename=fname,
                                                       name=name,
                                                       **kwargs)

    def apply_to(self, obj, **kwargs):
        """Use apply_to of current trace list."""
        return self.current_trace_list.apply_to(obj, **kwargs)

    @property
    def current_trace_list(self):
        trace_list_eff = None
        trace_list_name = from_currsys(self.meta["current_trace_list"],
                                       self.cmds)
        if trace_list_name is not None:
            trace_list_eff = self.trace_lists[trace_list_name]
        return trace_list_eff


def _warn_if_echelle_design_res_inconsistent(
        prefix,
        design_res,
        echelle_angle,
        pix_per_res_elem,
        pix_size,
        dispersion_focal_len,
        tolerance=0.05):
    """Warn if analytical order-center R is inconsistent with design_res."""
    try:
        design_res = float(design_res)
        pix_per_res_elem = float(pix_per_res_elem)
        pix_size = u.Quantity(pix_size).to(u.mm)
        dispersion_focal_len = u.Quantity(dispersion_focal_len).to(u.mm)
        echelle_angle_rad = u.Quantity(echelle_angle).to_value(u.rad)
    except (TypeError, ValueError, u.UnitConversionError):
        return
    if (
            not np.isfinite(design_res) or design_res <= 0
            or not np.isfinite(pix_per_res_elem) or pix_per_res_elem <= 0
            or dispersion_focal_len <= 0 * u.mm
            or pix_size <= 0 * u.mm):
        return

    implied_res = (
        2.0 * np.tan(echelle_angle_rad)
        * (dispersion_focal_len / pix_size).to_value(u.dimensionless_unscaled)
        / pix_per_res_elem
    )
    relative_delta = abs(implied_res - design_res) / design_res
    if relative_delta > tolerance:
        logger.warning(
            "Analytical echelle row %s design_res %.0f differs from "
            "order-center R %.0f implied by echelle angle, pixel size, "
            "FWHM, and dispersion focal length.",
            prefix,
            design_res,
            implied_res,
        )


class EchelleSpectralTraceList(SpectralTraceList):
    """
    SpectralTraceList effect for echelle spectrographs. Unlike SpectralTraceList, it generates the trace definitions
    instead of loading them from FITS file. The arguments required to define the echelle traces are supplied through
    a txt file containing a table of parameters using the filename kwarg.

    Below is an example of how to define the echelle trace parameters (see irdb/ZShooter_v1/traces/echelle_trace_parameters.txt):
    ----------------------------------------------------------------
    # min_wave_unit : nm
    # max_wave_unit : nm
    # echelle_blaze_unit : deg
    # focal_length_unit : mm
    # dispersion_focal_length_unit : mm
    # fwhm_unit : pixel
    # nominal_slit_width_unit : arcsec
    # plate_scale_unit : arcsec/mm
    # detector_pad_unit : pixel
    # pixel_size_unit : mm
    # n_disp_unit : pixel
    # n_xdisp_unit : pixel
    # disp_freq_unit : mm
    # xdisp_freq_unit : mm
    # slitwidth_unit : arcsec

    prefix    aperture_id    image_plane_id    m0    n    min_wave    max_wave   design_res    echelle_blaze    focal_length    dispersion_focal_length    fwhm    nominal_slit_width    plate_scale    detector_pad    pixel_size    n_disp    n_xdisp     disp_freq    xdisp_freq    slitwidth    dispdir
    ub         0              2                29    11    315         515          20000        64.2             225             225                         4.7     0.7                   10.0           10              0.015         4096      4096        200          1000          10           x
    gri        1              1                36    18    490         1020         20000        64.2             225             225                         4.7     0.7                   10.0           10              0.015         4096      4096        100          500           10           x
    nIR        2              0                40    24    970         2500         20000        64.2             225             225                         4.7     0.7                   10.0           10              0.015         4096      4096        45           175           10           x
    ----------------------------------------------------------------

    The calculated traces are stored in the same HDUList format as required by SpectralTraceList,
    and supplied to the parent class through hdulist kwarg.

    """
    required_keys = {"filename"}
    z_order = (71, 271, 671)

    def __init__(self, cmds=None, **kwargs):
        check_keys(kwargs, self.required_keys, action="error")
        self.cmds = cmds

        trace_param_filename = kwargs.pop("filename")
        save_generated_hdulist = kwargs.pop("save_generated_hdulist", False)
        generated_hdulist_filename = kwargs.pop(
            "generated_hdulist_filename", "analytical_echelle_traces.fits")
        trace_params = DataContainer(filename=trace_param_filename)
        hdulist = self._generate_trace_hdulist(trace_params)
        if save_generated_hdulist:
            output_path = Path(generated_hdulist_filename).expanduser()
            if not output_path.is_absolute():
                output_path = Path.cwd() / output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            hdulist.writeto(output_path, overwrite=True)
        kwargs["hdulist"] = hdulist
        super().__init__(cmds=cmds, **kwargs)

    def _generate_trace_hdulist(self, trace_params):
        hdul = fits.HDUList()
        hdul.append(fits.PrimaryHDU())
        hdul[0].header["EXTNAME"] = "OVERVIEW"
        hdul[0].header["ECAT"] = 1
        hdul[0].header["EDATA"] = 2

        trace_ids, ap_ids, im_ids = [], [], []
        for row in trace_params.table:
            prefix = row["prefix"]
            min_order = row['m0'] - row['n']
            max_order = row['m0']
            min_wave = row['min_wave'] * u.Unit(trace_params.meta["min_wave_unit"])
            max_wave = row['max_wave'] * u.Unit(trace_params.meta["max_wave_unit"])
            design_res = row['design_res']
            focal_len = row['focal_length'] * u.Unit(trace_params.meta["focal_length_unit"])
            dispersion_focal_len = None
            if "dispersion_focal_length" in trace_params.table.colnames:
                dispersion_focal_len = row["dispersion_focal_length"] * u.Unit(
                    trace_params.meta.get(
                        "dispersion_focal_length_unit",
                        trace_params.meta["focal_length_unit"],
                    ))
            disp_npix = int(row['n_disp'])
            xdisp_npix = int(row['n_xdisp'])
            detector_pad = (
                int(row['detector_pad'])
                if 'detector_pad' in trace_params.table.colnames else 0
            )
            pix_size = row['pixel_size'] * u.Unit(trace_params.meta["pixel_size_unit"])
            echelle_angle = np.deg2rad(row['echelle_blaze'])*u.rad
            xdisp_beta_center = np.deg2rad(row['xbeta_center'])*u.rad

            xdisp_groove_length = u.Unit(trace_params.meta["xdisp_freq_unit"]) / row['xdisp_freq']
            echelle_groove_length = u.Unit(trace_params.meta["disp_freq_unit"]) / row['disp_freq']
            pix_per_res_elem = row['fwhm']

            ss = echelle.spectrograph_factory(min_wave, max_wave, focal_len,
                                              design_res, echelle_angle, min_order, max_order,
                                              echelle_groove_length, pix_per_res_elem, disp_npix, xdisp_npix,
                                              pix_size, xdisp_groove_length=xdisp_groove_length,
                                              xdisp_beta_center=xdisp_beta_center,
                                              dispersion_focal_len=dispersion_focal_len)
            _warn_if_echelle_design_res_inconsistent(
                prefix,
                design_res,
                echelle_angle,
                pix_per_res_elem,
                pix_size,
                ss.dispersion_focal_length,
            )

            slit_edge = (row['slitlength'] / 2) * u.Unit(trace_params.meta["slitlength_unit"])
            slit_pos = np.linspace(-slit_edge, slit_edge, num=3)
            if "plate_scale" in trace_params.table.colnames:
                plate_scale = row["plate_scale"] * u.Unit(
                    trace_params.meta["plate_scale_unit"])
                slit_offset_pix = (slit_pos / plate_scale / pix_size).to_value(
                    u.dimensionless_unscaled)
            else:
                slit_offset_pix = (
                    slit_pos /
                    (from_currsys('!INST.pixel_scale', self.cmds) * u.arcsec)
                ).to_value(u.dimensionless_unscaled)
            detector_angle = 0.0
            if "detector_angle" in trace_params.table.colnames:
                detector_angle = u.Quantity(
                    row["detector_angle"],
                    u.Unit(trace_params.meta.get("detector_angle_unit", "deg")),
                ).to_value(u.deg)
            cang = np.cos(np.deg2rad(detector_angle))
            sang = np.sin(np.deg2rad(detector_angle))
            detector_x_padding = max(
                0.0,
                (
                    disp_npix * abs(cang)
                    + xdisp_npix * abs(sang)
                    - disp_npix
                ) / 2,
            )
            slit_x_padding = (
                abs(np.tan(np.deg2rad(detector_angle)))
                * float(np.nanmax(np.abs(slit_offset_pix)))
            )
            raw_x_min = 0.5 - detector_x_padding - slit_x_padding
            raw_x_max = disp_npix - 0.5 + detector_x_padding + slit_x_padding

            def raw_detector_pixels(wave, order):
                x = ss.wavelength_to_x_pixel(wave, order)
                y = ss.wavelength_to_y_pixel(wave)
                x = u.Quantity(x, copy=False).to_value(
                    u.dimensionless_unscaled)
                y = u.Quantity(y, copy=False).to_value(
                    u.dimensionless_unscaled)
                xpix = np.broadcast_to(x, (slit_offset_pix.size, wave.size))
                ypix = y[None, :] + slit_offset_pix[:, None]
                return xpix, ypix

            rotated_x_min = rotated_y_min = np.inf
            rotated_x_max = rotated_y_max = -np.inf
            # The analytical echelle layout is relative, not yet detector
            # placed. Rotate the footprint first, then place that rotated
            # footprint on the detector. Detector padding is not part of the
            # optical trace geometry; downstream detector/FOV code is
            # responsible for clipping padded display or extraction regions.
            for i, order in enumerate(ss.orders):
                raw_x = np.linspace(
                    raw_x_min, raw_x_max, num=max(disp_npix, 2))
                wave = ss.x_pixel_to_wavelength(raw_x, order)
                xpix, ypix = raw_detector_pixels(wave, order)
                xrot = cang * xpix - sang * ypix
                yrot = sang * xpix + cang * ypix
                rotated_x_min = min(rotated_x_min, float(np.nanmin(xrot)))
                rotated_x_max = max(rotated_x_max, float(np.nanmax(xrot)))
                rotated_y_min = min(rotated_y_min, float(np.nanmin(yrot)))
                rotated_y_max = max(rotated_y_max, float(np.nanmax(yrot)))
            rotated_x_center = rotated_x_min + (rotated_x_max - rotated_x_min) / 2
            rotated_y_center = rotated_y_min + (rotated_y_max - rotated_y_min) / 2

            def rotated_detector_pixels(wave, order):
                xpix, ypix = raw_detector_pixels(wave, order)
                xrot = cang * xpix - sang * ypix
                yrot = sang * xpix + cang * ypix
                return (
                    xrot - rotated_x_center + disp_npix / 2,
                    yrot - rotated_y_center + xdisp_npix / 2,
                )

            def on_detector(wave, order):
                xpix, ypix = rotated_detector_pixels(wave, order)
                return (
                    (xpix >= 0)
                    & (xpix <= disp_npix)
                    & (ypix >= 0)
                    & (ypix <= xdisp_npix)
                )

            for i, order in enumerate(ss.orders):
                candidate_raw_x = np.linspace(
                    raw_x_min, raw_x_max, num=max(disp_npix, 2))
                candidate_wave = ss.x_pixel_to_wavelength(candidate_raw_x, order)
                valid_wave = np.any(
                    on_detector(candidate_wave, order), axis=0)
                valid_indices = np.flatnonzero(valid_wave)
                if valid_indices.size == 0:
                    logger.debug(
                        "Skipping analytical trace %s_%d: no samples inside "
                        "the rotated detector rectangle.",
                        prefix, order,
                    )
                    continue
                raw_x = np.linspace(
                    candidate_raw_x[valid_indices[0]],
                    candidate_raw_x[valid_indices[-1]],
                    num=max(int(disp_npix * .1), 2),
                )
                wave = ss.x_pixel_to_wavelength(raw_x, order)
                valid_wave = np.any(on_detector(wave, order), axis=0)
                wave = wave[valid_wave]
                if wave.size < 2:
                    logger.debug(
                        "Skipping analytical trace %s_%d: fewer than two "
                        "samples after rotated detector-edge clipping.",
                        prefix, order,
                    )
                    continue
                s = np.tile(slit_pos, wave.size).reshape(wave.size, slit_pos.size).T.ravel()
                w = np.tile(wave, slit_offset_pix.size)
                xpix, ypix = rotated_detector_pixels(wave, order)
                xval = (xpix.ravel() - disp_npix / 2) * pix_size.to(u.mm)
                yval = (ypix.ravel() - xdisp_npix / 2) * pix_size.to(u.mm)

                order_table = Table(
                    {'wavelength': w.to(u.um), 's': s,
                     'x': xval,
                     'y': yval,
                     'x_pix': xpix.ravel() * u.pixel,
                     'y_pix': ypix.ravel() * u.pixel})

                trace_hdu = fits.BinTableHDU(order_table)
                trace_hdu.header['DISPDIR'] = row['dispdir']
                trace_hdu.header["EXTNAME"] = f'{prefix}_{order:d}'
                trace_hdu.header["DESIGNR"] = (
                    float(design_res), "Analytical trace design resolving power")
                trace_hdu.header["FWHMPIX"] = (
                    float(pix_per_res_elem), "Analytical nominal FWHM [pix]")
                trace_hdu.header["PIXSIZE"] = (
                    pix_size.to_value(u.mm), "Detector pixel size [mm]")
                trace_hdu.header["DISPFLEN"] = (
                    ss.dispersion_focal_length.to_value(u.mm),
                    "Effective echelle dispersion focal length [mm]")
                trace_hdu.header["DETPAD"] = (
                    detector_pad, "Legacy detector padding [pix]; not applied")
                trace_hdu.header["DETANG"] = (
                    detector_angle, "Detector rotation angle [deg]")
                if "nominal_slit_width" in trace_params.table.colnames:
                    trace_hdu.header["SLITWID"] = (
                        float(row["nominal_slit_width"]),
                        "Analytical nominal slit width [arcsec]")
                if "plate_scale" in trace_params.table.colnames:
                    trace_hdu.header["PLTSCALE"] = (
                        plate_scale.to_value(u.arcsec / u.mm),
                        "Analytical sky-to-image plate scale [arcsec/mm]")
                trace_ids.append(f'{prefix}_{order:d}')
                ap_ids.append(row["aperture_id"])
                im_ids.append(row["image_plane_id"])
                hdul.append(trace_hdu)

        if not trace_ids:
            raise ValueError(
                "Analytical echelle trace generation produced no detector "
                "intersecting traces.")
        hdul.insert(1, fits.BinTableHDU(Table(
            {'description': trace_ids,
             'extension_id': np.arange(len(trace_ids), dtype=int)+2,
             'aperture_id': ap_ids,
             'image_plane_id': im_ids
             })))

        return hdul
