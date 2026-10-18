# -*- coding: utf-8 -*-
"""
Utility classes and functions for ``SpectralTraceList``.

This module contains
   - the definition of the `SpectralTrace` class. The visible effect should
     always be a `SpectralTraceList`, even if that contains only one
     `SpectralTrace`.
   - the definition of the `XiLamImage` class
   - utility functions for use with spectral traces
"""

import warnings

import numpy as np

from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.sparse import csr_matrix

from astropy.table import Table, vstack
from astropy.modeling import fitting
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
from astropy.modeling.models import Polynomial2D

from ..utils import (
    power_vector,
    quantify,
    from_currsys,
    close_loop,
    figure_factory,
    get_logger,
)
from .ter_curves import detector_qe_at


logger = get_logger(__name__)


def _clip_tiny_negative_trace_pixels(image: np.ndarray) -> int:
    """Set floating-point roundoff-scale negative trace pixels to zero."""
    if image is None or not np.issubdtype(image.dtype, np.floating):
        return 0

    negative = image < 0
    if not np.any(negative):
        return 0

    finite = np.isfinite(image)
    image_scale = np.nanmax(np.abs(image[finite])) if np.any(finite) else 0
    tolerance = 128 * np.finfo(float).eps * max(float(image_scale), 1.0)
    tiny_negative = negative & (image >= -tolerance)
    image[tiny_negative] = 0
    return int(np.sum(tiny_negative))


def apply_detector_qe_to_trace_image(
    image: np.ndarray,
    detector_qe,
    wave_um: np.ndarray,
    detector_x: np.ndarray,
    detector_y: np.ndarray,
    **kwargs,
) -> np.ndarray:
    """Apply detector QE to an already trace-mapped image."""
    if detector_qe is None:
        return image
    qe_values = detector_qe_at(
        detector_qe,
        wave_um * u.um,
        detector_x=detector_x,
        detector_y=detector_y,
        **kwargs,
    )
    return image * np.asarray(qe_values, dtype=float)


class SpectralTrace:
    """Definition of one spectral trace.

    A SpectralTrace describes the mapping of spectral slit coordinates
    to the focal plane. The class reads an order layout and fits several
    functions to describe the geometry of the trace.

    Slit coordinates are:
    - xi : spatial position along the slit [arcsec]
    - lam : Wavelength [um]
    Focal plane coordinates are:
    - x, y : [mm]
    """

    _pixel_quadrature_order = 4
    _class_params = {
        "x_colname": "x",
        "y_colname": "y",
        "s_colname": "s",
        "wave_colname": "wavelength",
        "dwave": 0.002,
        "aperture_id": 0,
        "image_plane_id": 0,
        "extension_id": 2,
        "spline_order": 4,
        "pixel_size": None,
        "description": "<no description>",
        "trace_flux_jacobian": "!SIM.spectral.trace_flux_jacobian",
        "fit_degree": 4,
    }

    def __init__(self, trace_tbl, cmds=None, **kwargs):
        # Within scopesim, the actual parameter values are
        # passed as kwargs from the SpectralTraceList.
        # The values need to be here for stand-alone use.
        self.meta = {}
        self.meta.update(self._class_params)
        self.meta.update(kwargs)
        self.cmds = cmds

        if isinstance(trace_tbl, (fits.BinTableHDU, fits.TableHDU)):
            self.table = Table.read(trace_tbl)
            self.meta["trace_id"] = trace_tbl.header.get("EXTNAME",
                                                         "<unknown trace id>")
            self.dispersion_axis = trace_tbl.header.get("DISPDIR", "unknown")
            for header_key, meta_key in (
                ("DESIGNR", "design_res"),
                ("FWHMPIX", "nominal_fwhm_pix"),
                ("PIXSIZE", "pixel_size"),
                ("DISPFLEN", "dispersion_focal_length"),
                ("SLITWID", "nominal_slit_width"),
                ("PLTSCALE", "plate_scale"),
                ("DETPAD", "detector_pad"),
                ("DETANG", "detector_angle"),
                ("FITDEG", "fit_degree"),
                ("LSFRES", "optical_resolving_power"),
                ("SLITCOL", "slit_dispersion_colname"),
            ):
                if header_key in trace_tbl.header:
                    self.meta[meta_key] = trace_tbl.header[header_key]
        elif isinstance(trace_tbl, Table):
            self.table = trace_tbl
            self.dispersion_axis = "unknown"
        else:
            raise ValueError("trace_tbl must be one of (fits.BinTableHDU, "
                             "fits.TableHDU, astropy.Table) but is "
                             f"{type(trace_tbl)}")
        self.compute_interpolation_functions()

        # Declaration of other attributes
        self._xilamimg = None
        self._slit_cache = {}
        self._pixel_cache = {}
        self.dlam_per_pix = None
        self.fit_inverse = True

    @property
    def trace_id(self):
        """Return the name of the trace."""
        return self.meta["trace_id"]

    def fov_grid(self):
        """
        Provide information on the source space volume required by the effect.

        Returns
        -------
        A dictionary with entries `wave_min` and `wave_max`.
        Spatial limits are determined by the `ApertureMask` effect
        and are not returned here.
        """
        warnings.warn("The fov_grid method is deprecated and will be removed "
                      "in a future release.", DeprecationWarning, stacklevel=2)
        aperture_id = self.meta["aperture_id"]
        image_plane_id = self.meta["image_plane_id"]
        lam_arr = self.table[self.meta["wave_colname"]]

        wave_max = np.max(lam_arr)
        wave_min = np.min(lam_arr)

        return {"wave_min": wave_min, "wave_max": wave_max,
                "trace_id": self.trace_id, "aperture_id": aperture_id,
               "image_plane_id": image_plane_id}


    def compute_interpolation_functions(self):
        """
        Compute various interpolation functions between slit and focal plane.

        Focal plane coordinates are `x` and `y`, in mm. Slit coordinates are
        `xi` (spatial coordinate along the slit, in arcsec) and `lam`
        (wavelength, in um).

        #The interpolation functions include a shift in the focal-plane
        #coordinates, determined from the CRVAL of the source FOV.
        """
        x_arr = self.table[self.meta["x_colname"]]
        y_arr = self.table[self.meta["y_colname"]]

        xi_arr = self.table[self.meta["s_colname"]]
        lam_arr = self.table[self.meta["wave_colname"]]

        self.wave_min = quantify(np.min(lam_arr), u.um).value
        self.wave_max = quantify(np.max(lam_arr), u.um).value

        degree = self.meta["fit_degree"]
        self.xy2xi = Transform2D.fit(x_arr, y_arr, xi_arr, degree=degree)
        self.xy2lam = Transform2D.fit(x_arr, y_arr, lam_arr, degree=degree)
        self.xilam2x = Transform2D.fit(xi_arr, lam_arr, x_arr, degree=degree)
        self.xilam2y = Transform2D.fit(xi_arr, lam_arr, y_arr, degree=degree)
        self._xiy2x = Transform2D.fit(xi_arr, y_arr, x_arr, degree=degree)
        self._xiy2lam = Transform2D.fit(xi_arr, y_arr, lam_arr, degree=degree)
        if "slit_dispersion_colname" in self.meta:
            center = xi_arr == 0
            derivative = self.table[self.meta["slit_dispersion_colname"]].quantity.to_value(u.um / u.arcsec)
            # FOVs are restricted to this trace; allow unit-conversion roundoff
            # at the wavelength endpoints when interpolating the derivative.
            self.slit_dispersion = interp1d(lam_arr[center], derivative[center], bounds_error=False, fill_value="extrapolate")

        if self.dispersion_axis == "unknown":
            dlam_dx, dlam_dy = self.xy2lam.gradient()
            wave_mid = 0.5 * (self.wave_min + self.wave_max)
            xi_mid = np.mean(xi_arr)
            x_mid = self.xilam2x(xi_mid, wave_mid)
            y_mid = self.xilam2y(xi_mid, wave_mid)
            if dlam_dx(x_mid, y_mid) > dlam_dy(x_mid, y_mid):
                self.dispersion_axis = "x"
            else:
                self.dispersion_axis = "y"
            logger.info(
                "Dispersion axis determined to be %s", self.dispersion_axis)

    def _focal_plane_grid(self, fov):
        """Return the exact focal-plane grid used to rasterize this FOV."""
        wave_min = fov.meta["wave_min"].value       # [um]
        wave_max = fov.meta["wave_max"].value       # [um]
        xi_min = fov.meta["xi_min"].value           # [arcsec]
        xi_max = fov.meta["xi_max"].value           # [arcsec]
        xlim_mm, ylim_mm = self.footprint(wave_min=wave_min, wave_max=wave_max,
                                          xi_min=xi_min, xi_max=xi_max)

        if xlim_mm is None or ylim_mm is None:
            # Cases where there is no overlap betwen slit/trace and
            # fov are acceptable.
            return None

        det_header = fov.detector_header
        fpa_wcsd = WCS(det_header, key="D")
        naxis1d, naxis2d = det_header["NAXIS1"], det_header["NAXIS2"]
        xlim_px, ylim_px = fpa_wcsd.all_world2pix(xlim_mm, ylim_mm, 0)
        xmin = np.floor(xlim_px.min()).astype(int)
        xmax = np.ceil(xlim_px.max()).astype(int)
        ymin = np.floor(ylim_px.min()).astype(int)
        ymax = np.ceil(ylim_px.max()).astype(int)

        # Check if spectral trace footprint is outside FoV
        if xmax < 0 or xmin > naxis1d or ymax < 0 or ymin > naxis2d:
            logger.info(
                "Spectral trace %s: footprint is outside FoV", fov.trace_id)
            return None

        # Only work on parts within the FoV
        xmin = max(xmin, 0)
        xmax = min(xmax, naxis1d)
        ymin = max(ymin, 0)
        ymax = min(ymax, naxis2d)

        det_wcs = WCS(det_header, key="D")
        det_wcs.wcs.crpix -= np.array([xmin, ymin])

        sub_naxis1 = xmax - xmin
        sub_naxis2 = ymax - ymin
        xmin_mm, ymin_mm = fpa_wcsd.all_pix2world(xmin, ymin, 0)
        xmax_mm, ymax_mm = fpa_wcsd.all_pix2world(xmax, ymax, 0)

        x_mm = np.linspace(xmin_mm, xmax_mm, sub_naxis1, endpoint=False)
        y_mm = np.linspace(ymin_mm, ymax_mm, sub_naxis2, endpoint=False)
        return xmin, ymin, x_mm, y_mm, det_wcs

    def spectral_padding(self, fov):
        """Input wavelength support for slit displacement and optical blur."""
        lower, upper = fov.meta['wave_min'].to_value(u.um), fov.meta['wave_max'].to_value(u.um)
        wave = np.unique(u.Quantity(self.table[self.meta['wave_colname']]).to_value(u.um))
        wave = np.r_[lower, wave[(wave > lower) & (wave < upper)], upper]
        width = (fov.meta['eta_max'] - fov.meta['eta_min']).to_value(u.arcsec)
        if 'slit_dispersion_colname' in self.meta:
            derivative = self.slit_dispersion(wave)
        else:
            pixsize = (fov.detector_header['CDELT1D']*u.Unit(fov.detector_header['CUNIT1D'])).to_value(u.mm)
            self._set_dispersion(lower, upper, pixsize=pixsize)
            step = abs(fov.header['CDELT2'])*3600
            derivative = self.dlam_per_pix(wave)/step
        padding = np.max(abs(derivative))*width/2
        # Resolve the physical slit profile rather than using one fixed
        # wavelength step for all orders. The PSF may request a finer step.
        fov.meta['slit_spectral_width'] = np.min(abs(derivative))*width*u.um
        if 'optical_resolving_power' in self.meta:
            padding += 5*upper/(self.meta['optical_resolving_power']*np.sqrt(8*np.log(2)))
        return padding*u.um

    def map_spectra_to_focal_plane(self, fov):
        """
        Apply the spectral trace mapping to a spectral cube.

        The cube is contained in a FieldOfView object, which also has
        world coordinate systems for the Source (sky coordinates and
        wavelengths) and for the focal plane.
        The method returns a section of the fov image along with info on
        where this image lies in the focal plane.
        """
        logger.debug("Mapping %s", fov.trace_id)
        wave_min = fov.meta["wave_min"].value       # [um]
        wave_max = fov.meta["wave_max"].value       # [um]
        xi_min = fov.meta["xi_min"].value           # [arcsec]
        xi_max = fov.meta["xi_max"].value           # [arcsec]
        fov_header = fov.header
        det_header = fov.detector_header
        pixsize = det_header["CDELT1D"] * u.Unit(det_header["CUNIT1D"])
        pixsize = pixsize.to_value(u.mm)
        pixscale = fov_header["CDELT1"] * u.Unit(fov_header["CUNIT1"])
        pixscale = pixscale.to_value(u.arcsec)

        focal_plane_grid = self._focal_plane_grid(fov)
        if focal_plane_grid is None:
            return None
        xmin, ymin, x_mm, y_mm, det_wcs = focal_plane_grid
        sub_naxis1 = len(x_mm)
        sub_naxis2 = len(y_mm)
        xmax = xmin + sub_naxis1
        ymax = ymin + sub_naxis2
        image = np.zeros((sub_naxis2, sub_naxis1), dtype=np.float32)

        self._set_dispersion(wave_min, wave_max, pixsize=pixsize)
        quadrature_order = self._pixel_quadrature_order
        if 'effective_psf_fwhm' in fov.meta:
            middle = (wave_min + wave_max)/2
            dx_dxi, _ = self.xilam2x.gradient()
            dy_dxi, _ = self.xilam2y.gradient()
            along_step = pixsize/np.hypot(dx_dxi(0., middle), dy_dxi(0., middle))
            across_step = abs(self.dlam_per_pix(middle)/self.slit_dispersion(middle)) if 'slit_dispersion_colname' in self.meta else pixscale
            psf_width = fov.meta['effective_psf_fwhm'].to_value(u.arcsec)
            quadrature_order = max(quadrature_order, int(np.ceil(8*max(along_step, across_step)/psf_width)))
        slit_dispersion = None
        if "slit_dispersion_colname" in self.meta:
            slit_dispersion = self.slit_dispersion
        xilam = XiLamImage(fov, self.dlam_per_pix, slit_dispersion=slit_dispersion, _cache=self._slit_cache)
        if "optical_resolving_power" in self.meta:
            xilam.convolve_lsf(self.meta["optical_resolving_power"])
        self._xilamimg = xilam

        # Integrate the density over each detector pixel. The inverse optical
        # map is locally affine within a pixel; its exact center and Jacobian
        # are cached, while quadrature samples are streamed to bound memory.
        xx, yy = np.meshgrid(np.arange(sub_naxis1), np.arange(sub_naxis2))
        ximg_fpa, yimg_fpa = det_wcs.all_pix2world(xx, yy, 0)
        lam_fpa = self.xy2lam(ximg_fpa, yimg_fpa)
        jacobian = from_currsys(self.meta['trace_flux_jacobian'], self.cmds)
        key = (det_header.tostring(), xmin, ymin, image.shape, wave_min, wave_max, xi_min, xi_max, pixscale, jacobian)
        if self._pixel_cache.get('key') != key:
            self._pixel_cache.clear()
            matrix = det_wcs.pixel_scale_matrix
            dxidx, dxidy = self.xy2xi.gradient()
            dlamdx, dlamdy = self.xy2lam.gradient()
            ax, ay = dxidx(ximg_fpa, yimg_fpa), dxidy(ximg_fpa, yimg_fpa)
            bx, by = dlamdx(ximg_fpa, yimg_fpa), dlamdy(ximg_fpa, yimg_fpa)
            self._pixel_cache.update(key=key, xi=self.xy2xi(ximg_fpa, yimg_fpa), lam=lam_fpa,
                                     xi_x=ax*matrix[0, 0] + ay*matrix[1, 0], xi_y=ax*matrix[0, 1] + ay*matrix[1, 1],
                                     lam_x=bx*matrix[0, 0] + by*matrix[1, 0], lam_y=bx*matrix[0, 1] + by*matrix[1, 1],
                                     scale=self._trace_flux_scale(ximg_fpa, yimg_fpa, pixsize, pixscale, det_header))
        geometry = self._pixel_cache
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
        for nx, wx in zip(nodes/2, weights/2):
            for ny, wy in zip(nodes/2, weights/2):
                xi = geometry['xi'] + nx*geometry['xi_x'] + ny*geometry['xi_y']
                lam = geometry['lam'] + nx*geometry['lam_x'] + ny*geometry['lam_y']
                inside = (xi >= xi_min) & (xi < xi_max) & (lam >= wave_min) & (lam < wave_max)
                image[inside] += xilam.interp(xi[inside], lam[inside], grid=False) * geometry['scale'][inside] * wx * wy

        detector_qe = fov.meta.get("detector_qe")
        if detector_qe is not None:
            xpix_img, ypix_img = np.meshgrid(
                np.arange(xmin, xmax, dtype=np.float32),
                np.arange(ymin, ymax, dtype=np.float32),
            )
            image = apply_detector_qe_to_trace_image(
                image,
                detector_qe,
                lam_fpa,
                xpix_img,
                ypix_img,
                detector_x_mm=ximg_fpa,
                detector_y_mm=yimg_fpa,
                trace=self,
                fov=fov,
            )

        # img_header = sub_wcs.to_header()
        # img_header.update(det_wcs.to_header())
        img_header = det_wcs.to_header()
        img_header["XMIN"] = xmin
        img_header["XMAX"] = xmax
        img_header["YMIN"] = ymin
        img_header["YMAX"] = ymax
        img_header["BUNIT"] = "ph s-1"

        _clip_tiny_negative_trace_pixels(image)
        if np.any(image < 0):
            logger.warning(
                "map_spectra_to_focal_plane: %d negative pixels",
                np.sum(image < 0))

        image_hdu = fits.ImageHDU(header=img_header, data=image)
        return image_hdu

    def _trace_flux_scale(self, x_mm, y_mm, pixsize, pixscale, det_header):
        """Return the local [arcsec um] per detector pixel flux scale."""
        use_jacobian = from_currsys(self.meta["trace_flux_jacobian"], self.cmds)

        if not use_jacobian:
            dlam_by_dx, dlam_by_dy = self.xy2lam.gradient()
            dlam_per_pix = pixsize * np.sqrt(
                dlam_by_dx(x_mm, y_mm)**2 + dlam_by_dy(x_mm, y_mm)**2)
            return pixscale * dlam_per_pix

        dxi_by_dx, dxi_by_dy = self.xy2xi.gradient()
        dlam_by_dx, dlam_by_dy = self.xy2lam.gradient()
        jacobian = (dxi_by_dx(x_mm, y_mm) * dlam_by_dy(x_mm, y_mm) -
                    dxi_by_dy(x_mm, y_mm) * dlam_by_dx(x_mm, y_mm))
        return np.abs(jacobian) * self._detector_pixel_area(det_header)

    @staticmethod
    def _detector_pixel_area(det_header):
        """Return detector WCS pixel area in mm2."""
        detector_wcs = WCS(det_header, key="D")
        pixel_area = np.abs(np.linalg.det(detector_wcs.pixel_scale_matrix))
        pixel_area *= u.Unit(detector_wcs.wcs.cunit[0]) * u.Unit(detector_wcs.wcs.cunit[1])
        return pixel_area.to_value(u.mm**2)

    def rectify(
        self,
        hdulist: fits.HDUList,
        interps: list | None = None,
        wcs: WCS | None = None,
        fit_inverse: bool = True,
        bin_width: float | None = None,
        wave_min: float | None = None,
        wave_max: float | None = None,
        xi_min: float | None = None,
        xi_max: float | None = None,
    ) -> fits.ImageHDU | None:
        """Create 2D spectrum for a trace.

        .. todo: Consider turning `interps` into a dict keyed on extname.

        .. todo: Proper typing on `interps`.

        .. todo: Rewrite and expand docstring (FH).

        .. todo: Consider using quantities for float parameters.

        Parameters
        ----------
        hdulist : HDUList
           The result of scopesim readout
        interps : list of interpolation functions
           If provided, there must be one for each image extension in `hdulist`.
           The functions go from pixels to the images and can be created with,
           e.g., RectBivariateSpline.
        wcs : The WCS describing the rectified XiLamImage. This can be created
           in a simple way from the fov included in the `OpticalTrain` used in
           the simulation run producing `hdulist`.

        The WCS can also be set up via the following keywords:

        bin_width : float [um]
           The spectral bin width. This is best computed automatically from the
           spectral dispersion of the trace.
        wave_min, wave_max : float [um]
           Limits of the wavelength range to extract. The default is the
           the full range on which the `SpectralTrace` is defined. This may
           extend significantly beyond the filter window.
        xi_min, xi_max : float [arcsec]
           Spatial limits of the slit on the sky. This should be taken from
           the header of the hdulist, but this is not yet provided by scopesim
        """
        # Note: fit_inverse used to be self.fit_inverse, but wasn't used
        #       outside this function.
        wave_min = wave_min if wave_min is not None else self.wave_min
        wave_max = wave_max if wave_max is not None else self.wave_max
        if wave_max < self.wave_min or wave_min > self.wave_max:
            logger.debug("   Outside filter range")
            return None

        wave_min = max(wave_min, self.wave_min)
        wave_max = min(wave_max, self.wave_max)
        logger.info("Rectifying %s (%.02f .. %.02f um)",
                    self.trace_id, wave_min, wave_max)

        # bin_width is taken as the minimum dispersion of the trace
        # TODO: if wcs is given take bin width from cdelt1
        if bin_width is None:
            self._set_dispersion(wave_min, wave_max)
            bin_width = np.abs(self.dlam_per_pix.y).min()
        logger.debug("   Bin width %.02g um", bin_width)

        pixscale = from_currsys(self.meta["pixel_scale"], self.cmds)

        # Temporary solution to get slit length
        if xi_min is None:
            try:
                xi_min = hdulist[0].header["HIERARCH INS SLIT XIMIN"]
            except KeyError as err:
                raise KeyError(
                    "xi_min must be in header as 'INS SLIT XIMIN' or provided "
                    "directly as an argument."
                ) from err

        if xi_max is None:
            try:
                xi_max = hdulist[0].header["HIERARCH INS SLIT XIMAX"]
            except KeyError as err:
                raise KeyError(
                    "xi_max must be in header as 'INS SLIT XIMAX' or provided "
                    "directly as an argument."
                ) from err

        if wcs is None:
            wcs = WCS(naxis=2)
            wcs.wcs.ctype = ["WAVE", "LINEAR"]
            wcs.wcs.cunit = ["um", "arcsec"]
            wcs.wcs.crpix = [1, 1]
            wcs.wcs.cdelt = [bin_width, pixscale]  # PIXSCALE

        # crval set to wave_min to catch explicitely set value
        wcs.wcs.crval = [wave_min, xi_min]   # XIMIN

        nlam = int((wave_max - wave_min) / bin_width) + 1
        nxi = int((xi_max - xi_min) / pixscale) + 1

        # Create interpolation functions if not provided
        if interps is None:
            logger.info("Computing image interpolations")
            interps = make_image_interpolations(hdulist, kx=1, ky=1)

        # Create Xi, Lam images (do I need Iarr and Jarr or can I build Xi, Lam directly?)
        Iarr, Jarr = np.meshgrid(
            np.arange(nlam, dtype=np.float32),
            np.arange(nxi, dtype=np.float32),
        )
        Lam, Xi = wcs.all_pix2world(Iarr, Jarr, 0)

        # Make sure that we do have microns
        Lam = Lam * u.Unit(wcs.wcs.cunit[0]).to(u.um)

        # TODO: Replace self.xilam2x and self.xilam2y by fitted
        #       functions fit_xilam2x and fit_xilam2y, which need
        #       to be determined by mapping a grid of (xi, lam) to (x,y)
        #       and back to (xip, lamp); then fit (x,y)(xip, lamp).
        if fit_inverse:
            fit_xilam2x, fit_xilam2y = self._fit_transform(
                xi_min,
                xi_max,
                wave_min,
                wave_max,
            )
        else:
            logger.debug("Using matrix transform")
            fit_xilam2x, fit_xilam2y = self.xilam2x, self.xilam2y

        # Convert Xi, Lam to focal plane units
        Xarr = fit_xilam2x(Xi, Lam)
        Yarr = fit_xilam2y(Xi, Lam)

        rect_spec = np.zeros_like(Xarr, dtype=np.float32)

        ihdu = 0
        # TODO: make this more iteratory
        for hdu in hdulist:
            if not isinstance(hdu, fits.ImageHDU):
                continue

            wcs_fp = WCS(hdu.header, key="D")
            n_x = hdu.header["NAXIS1"]
            n_y = hdu.header["NAXIS2"]
            iarr, jarr = wcs_fp.all_world2pix(Xarr, Yarr, 0)
            mask = (iarr > 0) * (iarr < n_x) * (jarr > 0) * (jarr < n_y)
            if np.any(mask):
                specpart = interps[ihdu](jarr, iarr, grid=False)
                rect_spec += specpart * mask

            ihdu += 1

        header = wcs.to_header()
        header["EXTNAME"] = self.trace_id
        return fits.ImageHDU(data=rect_spec, header=header)

    def _fit_transform(self, ximin, ximax, wmin, wmax):
        """Fit inverses to self.xy2xi and self.xy2lam

        The method replaces self.xilam2x and self.xilam2y by fitted
        functions fit_xilam2x and fit_xilam2y, determined by mapping a grid of
        (xi, lam) to (x,y) and back to (xip, lamp); then fit (x,y)(xip, lamp).
        """

        # Build input (xi, lam) grid
        xi = np.linspace(ximin, ximax, 51)
        lam = np.linspace(wmin, wmax, 201)

        # Do a full round using the matrix transforms
        x2d = self.xilam2x(xi, lam, grid=True)
        y2d = self.xilam2y(xi, lam, grid=True)
        xip = self.xy2xi(x2d, y2d, grid=False)
        lamp = self.xy2lam(x2d, y2d, grid=False)

        # Determine the fits
        pinit_x = Polynomial2D(degree=4)
        pinit_y = Polynomial2D(degree=4)
        fitter = fitting.LinearLSQFitter()
        xilam2x = fitter(pinit_x, xip, lamp, x2d)
        xilam2y = fitter(pinit_y, xip, lamp, y2d)

        return xilam2x, xilam2y

    def footprint(self, wave_min=None, wave_max=None, xi_min=None, xi_max=None):
        """
        Return corners of rectangle enclosing spectral trace.

        Parameters
        ----------
        wave_min, wave_max : float [um], Quantity
            Minimum and maximum wavelength to compute the footprint on.
            If `None`, use the full range that spectral trace is defined on.
            Float values are interpreted as microns.
        xi_min, xi_max : float [arcsec], Quantity
            Minimum and maximum slit position on the sky.
            If `None`, use the full range that spectral trace is defined on.
            Float values are interpreted as arcsec.
        """
        # Define the wavelength range of the footprint. This is a compromise
        # between the requested range (by method args) and the definition
        # range of the spectral trace
        # This is only relevant if the trace is given by a table of reference
        # points. Otherwise (METIS LMS!) we assume that the range is valid.
        if ("wave_colname" in self.meta and
                self.meta["wave_colname"] in self.table.colnames):
            # Here, the parameters are obtained from a table of reference points
            wave_unit = self.table[self.meta["wave_colname"]].unit
            wave_val = quantify(self.table[self.meta["wave_colname"]].data,
                                wave_unit)

            if wave_min is None:
                wave_min = np.min(wave_val)
            if wave_max is None:
                wave_max = np.max(wave_val)

            wave_min = quantify(wave_min, u.um)
            wave_max = quantify(wave_max, u.um)

            # Requested wavelenth range is entirely outside definition range:
            # no footprint
            if wave_min > np.max(wave_val) or wave_max < np.min(wave_val):
                logger.info("Trace not in wavelength range")
                return None, None

            # Restrict to overlap of requested range and definition range
            wave_min = max(wave_min, np.min(wave_val)).value
            wave_max = min(wave_max, np.max(wave_val)).value

            # Define the slit range of the footprint. This is a compromise
            # between the requested range (by method args) and the definition
            # range of the spectral trace
            try:
                xi_unit = self.table[self.meta["s_colname"]].unit
            except KeyError:
                xi_unit = u.arcsec

            xi_val = quantify(self.table[self.meta["s_colname"]].data,
                              xi_unit)

            if xi_min is None:
                xi_min = np.min(xi_val)
            if xi_max is None:
                xi_max = np.max(xi_val)

            xi_min = quantify(xi_min, u.arcsec)
            xi_max = quantify(xi_max, u.arcsec)

            # Requested slit range is entirely outside definition range:
            # no footprint
            if xi_min > np.max(xi_val) or xi_max < np.min(xi_val):
                logger.info("Slit outside range (spatial)")
                return None, None

            # Restrict to overlap of requested range and definition range
            xi_min = max(xi_min, np.min(xi_val)).value
            xi_max = min(xi_max, np.max(xi_val)).value

        # Map the edges of xi/lam to the focal plance
        n_edge = 512
        wave_edge = np.concatenate((
            np.linspace(wave_min, wave_max, n_edge),
            [wave_max] * n_edge,
            np.linspace(wave_min, wave_max, n_edge),
            [wave_min] * n_edge,
        ))
        xi_edge = np.concatenate((
            [xi_min] * n_edge,
            np.linspace(xi_min, xi_max, n_edge),
            [xi_max] * n_edge,
            np.linspace(xi_min, xi_max, n_edge),
        ))

        x_edge = self.xilam2x(xi_edge, wave_edge)
        y_edge = self.xilam2y(xi_edge, wave_edge)

        return ([x_edge.min(), x_edge.max(), x_edge.max(), x_edge.min()],
                [y_edge.min(), y_edge.min(), y_edge.max(), y_edge.max()])

    def plot(self, wave_min=None, wave_max=None, xi_min=None, xi_max=None, *,
             c="r", axes=None, plot_footprint=True, plot_wave=True,
             plot_ctrlpnts=True, plot_outline=False, plot_trace_id=False):
        """Plot control points (and/or footprint) of the SpectralTrace.

        Parameters
        ----------
        wave_min : float, optional
            Minimum wavelength, if any.
        wave_max : float, optional
            Maximum wavelength, if any.
        xi_min : float, optional
            Minimum slit, if any.
        xi_max : float, optional
            Maximum slit, if any.
        c : str, optional
            Colour, any valid matplotlib colour string. The default is "r".
        axes : matplotlib axes, optional
            The axes object to use for the plot. If None (default), a new
            figure with one axes will be created.

        Returns
        -------
        axes : matplotlib axes
            The axes object containing the plot.

        Other Parameters
        ----------------
        plot_footprint : bool, optional
            Plot a rectangle encompassing all control points, which may be
            larger than the area actually covered by the trace, if the trace is
            not exactly perpendicular to the detector. The default is True.
        plot_wave : bool, optional
            Annotate the wavelength points. The default is True.
        plot_ctrlpnts : bool, optional
            Plot the individual control points as makers. The default is True.
        plot_outline : bool, optional
            Plot the smallest tetragon encompassing all control points.
            The default is False.
        plot_trace_id : bool, optional
            Write the trace ID in the middle of the trace.
            The default is False.
        """
        if axes is None:
            _, axes = figure_factory()

        # Footprint (rectangle enclosing the trace)
        xlim, ylim = self.footprint(wave_min=wave_min, wave_max=wave_max)
        if xlim is None:
            return axes

        if plot_footprint:
            axes.plot(list(close_loop(xlim)), list(close_loop(ylim)))

        # for convenience...
        xname = self.meta["x_colname"]
        yname = self.meta["y_colname"]
        wname = self.meta["wave_colname"]
        sname = self.meta["s_colname"]

        # Control points
        waves = self.table[wname]
        if wave_min is None:
            wave_min = waves.min()
        if wave_max is None:
            wave_max = waves.max()
        xis = self.table[sname]
        if xi_min is None:
            xi_min = xis.min()
        if xi_max is None:
            xi_max = xis.max()

        mask = ((waves >= wave_min) & (waves <= wave_max)
                & (xis >= xi_min) & (xis <= xi_max))
        if sum(mask) <= 2:
            return axes

        w = waves[mask]

        x = self.table[xname][mask]
        y = self.table[yname][mask]
        if plot_ctrlpnts:
            axes.plot(x, y, "o", c=c)

        if plot_outline or plot_trace_id:
            blue_end = self.table[mask][w == w.min()]
            red_end = self.table[mask][w == w.max()]
            blue_end.sort(sname)
            red_end.sort(sname)
            corners = vstack([blue_end[[0, -1]][xname, yname],
                              red_end[[-1, 0]][xname, yname],
                              blue_end[0][xname, yname]])

        if plot_outline:
            axes.plot(corners[xname], corners[yname], c=c)

        if plot_trace_id:
            axes.text(corners[xname][:-1].mean(), corners[yname][:-1].mean(),
                      self.trace_id, c=c, rotation="vertical",
                      ha="center", va="center")

        for wave in np.unique(w):
            xx = x[w == wave]
            xx.sort()
            dx = xx[-1] - xx[-2]

            if plot_wave:
                axes.text(x[w == wave].max() + 0.5 * dx,
                          y[w == wave].mean(),
                          str(wave), va="center", ha="left")

        axes.set_aspect("equal")
        return axes

    def _set_dispersion(self, wave_min, wave_max, pixsize=None):
        """Compute of dispersion dlam_per_pix along xi=0."""
        # ..todo: This may have to be generalised - xi=0 is at the centre
        # of METIS slits and the short MICADO slit.

        xi = np.array([0] * 1001)
        lam = np.linspace(wave_min, wave_max, 1001)
        x_mm = self.xilam2x(xi, lam)
        y_mm = self.xilam2y(xi, lam)
        if self.dispersion_axis == "x":
            dlam_grad = self.xy2lam.gradient()[0]  # dlam_by_dx
        else:
            dlam_grad = self.xy2lam.gradient()[1]  # dlam_by_dy
        if pixsize is None:
            pixsize = (from_currsys(self.meta["pixel_scale"], self.cmds) /
                       from_currsys(self.meta["plate_scale"], self.cmds))
        elif isinstance(pixsize, u.Quantity):
            pixsize = pixsize.to_value(u.mm)

        self.dlam_per_pix = interp1d(lam,
                                     dlam_grad(x_mm, y_mm) * pixsize,
                                     fill_value="extrapolate")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.table!r}, **{self.meta!r})"

    def __str__(self):
        msg = (f"<SpectralTrace> \"{self.trace_id}\" : "
               f"[{self.wave_min:.4f}, {self.wave_max:.4f}]um : "
               f"Ext {self.meta['extension_id']} : "
               f"Aperture {self.meta['aperture_id']} : "
               f"ImagePlane {self.meta['image_plane_id']}")
        return msg


def _uniform_sum_cdf(position, width_a, width_b):
    """CDF of two centered uniform intervals, including a zero-width interval."""
    wide, narrow = np.maximum(width_a, width_b), np.minimum(width_a, width_b)
    probability = np.clip(.5 + position / wide, 0, 1)
    finite = narrow > 0
    a, b, x = wide[finite], narrow[finite], position[:, finite]
    tail = np.clip((a + b)/2 - abs(x), 0, b)**2 / (2*a*b)
    probability[:, finite] = np.where(abs(x) <= (a - b)/2, .5 + x/a, np.where(x < 0, tail, 1 - tail))
    return probability


def _slit_cell_operator(wavelength, derivative, eta, deta):
    """Conservative overlap of wavelength bins sheared across one finite slit cell.

    Wavelength and its derivative use um and um/arcsec; eta uses arcsec.
    Locally affine dispersion maps each rectangular (lambda, eta) cell to a
    trapezoidal wavelength profile. Integrating its CDF at output edges keeps
    the input photon integral, including lines narrower than a wavelength bin.
    """
    step = wavelength[1] - wavelength[0]
    center = np.arange(len(wavelength)) + derivative * eta / step
    spectral_width = 1 + eta * np.gradient(derivative, wavelength)
    slit_width = abs(derivative * deta / step)
    assert np.all(spectral_width > 0), 'Slit displacement must preserve wavelength ordering.'
    span = spectral_width + slit_width
    first = np.floor(center - span/2 + .5).astype(int)
    output = first[None, :] + np.arange(int(np.ceil(span.max())) + 1)[:, None]
    lower = output - .5 - center
    probability = _uniform_sum_cdf(lower + 1, spectral_width, slit_width) - _uniform_sum_cdf(lower, spectral_width, slit_width)
    source = np.broadcast_to(np.arange(len(wavelength)), output.shape)
    valid = (output >= 0) & (output < len(wavelength)) & (probability > 0)
    return csr_matrix((probability[valid], (output[valid], source[valid])), shape=(len(wavelength), len(wavelength)))


class XiLamImage:
    """
    Class to compute a rectified 2D spectrum.

    The class produces and holds an image of xi (relative position along
    the spatial slit direction) and wavelength lambda.

    Parameters
    ----------
    fov : FieldOfView
    dlam_per_pix : a 1-D interpolation function from wavelength (in um) to
          dispersion (in um/pixel); alternatively a number giving an average
          dispersion
    slit_dispersion : callable, optional
          Physical cross-slit derivative d(lambda)/d(eta) in um/arcsec as a
          function of wavelength in um. When supplied, this replaces the
          legacy coupling between sky-grid sampling and detector dispersion.
    """

    def __init__(self, fov, dlam_per_pix, slit_dispersion=None, _cache=None):
        # ..todo: we assume that we always have a cube. We use SpecCADO's
        #         add_cube_layer method
        cube_wcs = WCS(fov.cube.header, key=" ")
        wcs_lam = cube_wcs.sub([3])

        d_xi = fov.cube.header["CDELT1"]
        d_xi *= u.Unit(fov.cube.header["CUNIT1"]).to(u.arcsec)
        d_eta = fov.cube.header["CDELT2"]
        d_eta *= u.Unit(fov.cube.header["CUNIT2"]).to(u.arcsec)
        d_lam = fov.cube.header["CDELT3"]
        d_lam *= u.Unit(fov.cube.header["CUNIT3"]).to(u.um)

        # This is based on the cube shape and assumes that the cube's spatial
        # dimensions are set by the slit aperture
        (n_lam, n_eta, n_xi) = fov.cube.data.shape

        # arrays of cube coordinates
        cube_xi = cube_wcs.all_pix2world(np.arange(n_xi), np.zeros(n_xi), np.zeros(n_xi), 0)[0]
        cube_eta = cube_wcs.all_pix2world(np.zeros(n_eta), np.arange(n_eta), np.zeros(n_eta), 0)[1]
        if cube_wcs.has_celestial:
            # Slit coordinates are offsets about zero; RA wraps at 360 degrees.
            cube_xi = (cube_xi + 180) % 360 - 180
        cube_xi *= u.Unit(cube_wcs.wcs.cunit[0]).to(u.arcsec)
        cube_eta *= u.Unit(cube_wcs.wcs.cunit[1]).to(u.arcsec)
        # Each IFU slice has its own optical axis, independent of sky position.
        eta_center = (fov.meta["eta_min"] + fov.meta["eta_max"]) / 2
        cube_eta -= eta_center.to_value(u.arcsec)
        cube_lam = wcs_lam.all_pix2world(np.arange(n_lam), 0)[0]
        cube_lam *= u.Unit(wcs_lam.wcs.cunit[0]).to(u.um)

        # Initialise the array to hold the xi-lambda image
        self.image = np.zeros((n_xi, n_lam), dtype=np.float32)
        self.lam = cube_lam
        try:
            dlam_per_pix_val = dlam_per_pix(np.asarray(self.lam))
        except TypeError:
            dlam_per_pix_val = dlam_per_pix
            logger.warning("Using scalar dlam_per_pix = %.2g",
                           dlam_per_pix_val)

        derivative = -dlam_per_pix_val / d_eta if slit_dispersion is None else slit_dispersion(self.lam)
        derivative = np.broadcast_to(derivative, self.lam.shape)
        key = (cube_lam[0], d_lam, n_lam, cube_eta[0], d_eta, n_eta, derivative.tobytes())
        if _cache is None:
            _cache = {}
        if _cache.get('key') != key:
            _cache.clear()
            _cache['operators'] = [_slit_cell_operator(cube_lam, derivative, eta, d_eta) for eta in cube_eta]
            _cache['key'] = key
        for i, operator in enumerate(_cache['operators']):
            self.image += (operator @ fov.cube.data[:, i, :]).T * abs(d_eta)

        # WCS for the xi-lambda image, i.e. the rectified 2D spectrum
        # Default WCS with xi in arcsec
        self.wcs = WCS(naxis=2)
        self.wcs.wcs.crpix = [1, 1]
        self.wcs.wcs.crval = [self.lam[0], cube_xi[0]]
        self.wcs.wcs.pc = [[1, 0], [0, 1]]
        self.wcs.wcs.cdelt = [d_lam, d_xi]
        self.wcs.wcs.ctype = ["LINEAR", "LINEAR"]
        self.wcs.wcs.cname = ["WAVELEN", "SLITPOS"]
        self.wcs.wcs.cunit = ["um", "arcsec"]

        # Alternative: xi = [0, 1], dimensionless
        self.wcsa = WCS(naxis=2)
        self.wcsa.wcs.crpix = [1, 1]
        self.wcsa.wcs.crval = [self.lam[0], 0]
        self.wcsa.wcs.pc = [[1, 0], [0, 1]]
        self.wcsa.wcs.cdelt = [d_lam, 1./n_xi]
        self.wcsa.wcs.ctype = ["LINEAR", "LINEAR"]
        self.wcsa.wcs.cname = ["WAVELEN", "SLITPOS"]
        self.wcsa.wcs.cunit = ["um", ""]

        self.xi = self.wcs.all_pix2world(self.lam[0], np.arange(n_xi), 0)[1]
        self.npix_xi = n_xi
        self.npix_lam = n_lam

    def interp(self, xi, wavelength, grid=True):
        """Bilinear density interpolation on the existing uniform WCS grid."""
        xi, wavelength = np.meshgrid(xi, wavelength, indexing='ij') if grid else np.broadcast_arrays(xi, wavelength)
        row = (xi - self.xi[0])/self.wcs.wcs.cdelt[1]
        column = (wavelength - self.lam[0])/self.wcs.wcs.cdelt[0]
        return map_coordinates(self.image, [row, column], order=1, mode='nearest', prefilter=False)

    def convolve_lsf(self, resolving_power):
        """Apply an optical Gaussian LSF at constant R, preserving photon density.

        This is the optical blur alone; slit illumination is integrated above.
        Convolving lambda * I_lambda on a uniform log(lambda) grid preserves
        integrated flux rather than treating I_lambda as a log-bin count.
        """
        if resolving_power <= 0:
            raise ValueError("Optical resolving power must be positive")
        log_wave = np.log(self.lam)
        n = int(np.ceil((log_wave[-1] - log_wave[0]) / np.min(np.diff(log_wave)))) + 1
        log_grid = np.linspace(log_wave[0], log_wave[-1], n)
        sigma = 1 / (resolving_power * np.sqrt(8 * np.log(2)) * (log_grid[1] - log_grid[0]))
        density = interp1d(log_wave, self.image * self.lam, axis=1)(log_grid)
        blurred = gaussian_filter1d(density, sigma, axis=1, mode="nearest")
        self.image = interp1d(log_grid, blurred, axis=1)(log_wave) / self.lam


class Transform2D():
    """
    2-dimensional polynomial transform.

    The class is instantiated from a m x n matrix A that contains the
    coefficients of the polynomial. Along rows, the power of x increases;
    along columns, the power of y increases, such that A[j, i] is
    the coefficient of x^i y^j.

    The functions `pretransform_x` and `pretransform_y` can be used to
    transform the input variables before the matrix is applied. The function
    `posttransform` can be applied to the output after application of the
    matrix.

    In Scopesim, a usecase for the pre- and post-transform functions is the
    METIS LMS, where the matrices are applied to phases while Scopesim
    operates on wavelengths. The functions to pass are `lam2phase` and
    `phase2lam`.

    Parameters
    ----------
    matrix : np.array
        matrix of polynomial coefficients
    pretransform_x : function, tuple
    pretransform_y : function, tuple
        If not None, the function is applied to the input
        variable `x` or `y` before the actual 2D transform is computed
    posttransform : function, tuple
        If not None, the function is applied to the output variable
        after the 2D transform is computed

    When passed as a tuple, the first element is the function itself,
    the second element is a dictionary of arguments to the function.
    Example:
    ```
    def rescale(x, scale=1.):
        return x * scale
    pretransform_x = (rescale, {"scale": 0.5})
    ```
    """

    def __init__(self, matrix, pretransform_x=None,
                 pretransform_y=None, posttransform=None,
                 dpretransform_x=1.0, dpretransform_y=1.0):
        self.matrix = np.asarray(matrix)
        self.ny, self.nx = self.matrix.shape
        self.pretransform_x = self._repackage(pretransform_x)
        self.pretransform_y = self._repackage(pretransform_y)
        self.posttransform = self._repackage(posttransform)
        self.dpretransform_x = dpretransform_x
        self.dpretransform_y = dpretransform_y

    def _repackage(self, trafo):
        """Make sure `trafo` is a tuple."""
        if trafo is not None and not isinstance(trafo, tuple):
            trafo = (trafo, {})
        return trafo

    def __call__(self, x, y, grid=False, **kwargs):
        """
        Apply the polynomial transform.

        The transformation is a polynomial based on the simple
        monomials x^i y^j. When grid=True, the transform is applied to the grid
        formed by the tensor product of the vectors x and y. When grid=False,
        the vectors of x and y define the components of a number of points (in
        this case, x and y must be of the same length).

        Functions `pretransform_x`, `pretransform_y` and `posttransform`
        can be supplied to override the instance values for this call only;
        the instance itself is not modified.

        Parameters
        ----------
        x, y : np.array
            x and y values to transform
        grid : boolean
            If true, return result for all pairs of components in x and y.

        Return
        ------
        When grid=True, a matrix with results for all pairings of components
        in x and y. When grid=False, a vector. In this case, x and y must
        have the same length.
        """
        # Per-call overrides must not modify the instance
        pretransform_x = self.pretransform_x
        pretransform_y = self.pretransform_y
        posttransform = self.posttransform
        if "pretransform_x" in kwargs:
            pretransform_x = self._repackage(kwargs["pretransform_x"])
        if "pretransform_y" in kwargs:
            pretransform_y = self._repackage(kwargs["pretransform_y"])
        if "posttransform" in kwargs:
            posttransform = self._repackage(kwargs["posttransform"])

        x = np.array(x)
        y = np.array(y)
        orig_shape = x.shape

        if not grid and x.shape != y.shape:
            raise ValueError("x and y must have the same length when grid "
                             "is False")

        # Apply pre transforms
        if pretransform_x is not None:
            x = pretransform_x[0](x, **pretransform_x[1])
        if pretransform_y is not None:
            y = pretransform_y[0](y, **pretransform_y[1])

        xvec = power_vector(x.flatten(), self.nx - 1)
        yvec = power_vector(y.flatten(), self.ny - 1)

        temp = self.matrix @ xvec

        if grid:
            result = yvec.T @ temp
        else:
            # Compute the scalar product of each column in yvec with the
            # corresponding column in temp. This gives the diagonal of the
            # expression in the "grid" branch.
            result = (yvec * temp).sum(axis=0)
            if not orig_shape:
                result = result.item()
            else:
                result = result.reshape(orig_shape)

        # Apply posttransform
        if posttransform is not None:
            result = posttransform[0](result, **posttransform[1])

        return result

    @classmethod
    def fit(cls, xin, yin, xout, degree=4, normalize=True):
        """Determine polynomial fits."""
        xin = np.asarray(xin, dtype=float)
        yin = np.asarray(yin, dtype=float)
        xout = np.asarray(xout, dtype=float)
        pretransform_x = pretransform_y = None
        dpretransform_x = dpretransform_y = 1.0
        if normalize:
            x_offset, x_scale = _fit_normalization(xin)
            y_offset, y_scale = _fit_normalization(yin)
            xin_fit = _linear_rescale(xin, x_offset, x_scale)
            yin_fit = _linear_rescale(yin, y_offset, y_scale)
            pretransform_x = (
                _linear_rescale,
                {"offset": x_offset, "scale": x_scale},
            )
            pretransform_y = (
                _linear_rescale,
                {"offset": y_offset, "scale": y_scale},
            )
            dpretransform_x = 1.0 / x_scale
            dpretransform_y = 1.0 / y_scale
        else:
            xin_fit = xin
            yin_fit = yin
        pinit = Polynomial2D(degree=degree)
        fitter = fitting.LinearLSQFitter()
        fit = fitter(pinit, xin_fit, yin_fit, xout)
        return Transform2D(
            fit2matrix(fit),
            pretransform_x=pretransform_x,
            pretransform_y=pretransform_y,
            dpretransform_x=dpretransform_x,
            dpretransform_y=dpretransform_y,
        )

    def gradient(self):
        """Compute the gradient of a 2d polynomial transformation."""
        mat = self.matrix

        dmat_x = (mat * np.arange(self.nx))[:, 1:] * self.dpretransform_x
        dmat_y = (mat.T * np.arange(self.ny)).T[1:, :] * self.dpretransform_y

        return (
            Transform2D(
                dmat_x,
                pretransform_x=self.pretransform_x,
                pretransform_y=self.pretransform_y,
            ),
            Transform2D(
                dmat_y,
                pretransform_x=self.pretransform_x,
                pretransform_y=self.pretransform_y,
            ),
        )


def _fit_normalization(values):
    """Return a stable offset and scale for polynomial fitting."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    minimum = float(np.nanmin(finite))
    maximum = float(np.nanmax(finite))
    offset = minimum + (maximum - minimum) / 2
    scale = (maximum - minimum) / 2
    if not np.isfinite(scale) or scale == 0:
        scale = 1.0
    return offset, scale


def _linear_rescale(values, offset=0.0, scale=1.0):
    """Linearly rescale values for polynomial fitting/evaluation."""
    return (np.asarray(values, dtype=float) - offset) / scale


def fit2matrix(fit):
    """
    Return coefficients from a polynomial fit as a matrix.

    The Polynomial2D fits of degree n have coefficients for all i, j
    with i + j <= n.
    How would one rearrange those?
    """
    coeffs = dict(zip(fit.param_names, fit.parameters))
    deg = fit.degree
    mat = np.zeros((deg + 1, deg + 1), dtype=float)
    for i in range(deg + 1):
        for j in range(deg + 1):
            try:
                mat[j, i] = coeffs[f"c{i}_{j}"]
            except KeyError:
                pass
    return mat


# ..todo: should the next three functions be combined and return a dictionary of fits?
def xilam2xy_fit(layout, params):
    """
    Determine polynomial fits of FPA position.

    Fits are of degree 4 as a function of slit position and wavelength.
    """
    xi_arr = layout[params["s_colname"]]
    lam_arr = layout[params["wave_colname"]]
    x_arr = layout[params["x_colname"]]
    y_arr = layout[params["y_colname"]]

    # Filter the lists: remove any points with x==0
    # ..todo: this may not be necessary after sanitising the table
    # good = x != 0
    # xi = xi[good]
    # lam = lam[good]
    # x = x[good]
    # y = y[good]

    # compute the fits
    pinit_x = Polynomial2D(degree=4)
    pinit_y = Polynomial2D(degree=4)
    fitter = fitting.LinearLSQFitter()
    xilam2x = fitter(pinit_x, xi_arr, lam_arr, x_arr)
    xilam2y = fitter(pinit_y, xi_arr, lam_arr, y_arr)

    return xilam2x, xilam2y


def xy2xilam_fit(layout, params):
    """
    Determine polynomial fits of wavelength/slit position.

    Fits are of degree 4 as a function of focal plane position
    """
    xi_arr = layout[params["s_colname"]]
    lam_arr = layout[params["wave_colname"]]
    x_arr = layout[params["x_colname"]]
    y_arr = layout[params["y_colname"]]

    pinit_xi = Polynomial2D(degree=4)
    pinit_lam = Polynomial2D(degree=4)
    fitter = fitting.LinearLSQFitter()
    xy2xi = fitter(pinit_xi, x_arr, y_arr, xi_arr)
    xy2lam = fitter(pinit_lam, x_arr, y_arr, lam_arr)

    return xy2xi, xy2lam


def _xiy2xlam_fit(layout, params):
    """Determine polynomial fits of wavelength/slit position.

    Fits are of degree 4 as a function of focal plane position
    """
    # These are helper functions to allow fitting of left/right edges
    # for the purpose of checking whether a trace is on a chip or not.

    xi_arr = layout[params["s_colname"]]
    lam_arr = layout[params["wave_colname"]]
    x_arr = layout[params["x_colname"]]
    y_arr = layout[params["y_colname"]]

    pinit_x = Polynomial2D(degree=4)
    pinit_lam = Polynomial2D(degree=4)
    fitter = fitting.LinearLSQFitter()
    xiy2x = fitter(pinit_x, xi_arr, y_arr, x_arr)
    xiy2lam = fitter(pinit_lam, xi_arr, y_arr, lam_arr)
    return xiy2x, xiy2lam


def make_image_interpolations(hdulist, **kwargs):
    """Create 2D interpolation functions for images.

    The interpolation functions are called with (j, i) pixel coordinates,
    i.e. the first argument corresponds to the row (NAXIS2) axis of the
    image, the second to the column (NAXIS1) axis.
    """
    interps = []
    for hdu in hdulist:
        if isinstance(hdu, fits.ImageHDU):
            interps.append(
                RectBivariateSpline(np.arange(hdu.header["NAXIS2"]),
                                    np.arange(hdu.header["NAXIS1"]),
                                    hdu.data, **kwargs)
            )
    return interps

# ..todo: Check whether the following functions are actually used


def rolling_median(x, n):
    """Calculate the rolling median of a sequence for +/- n entries."""
    y = [np.median(x[max(0, i-n):min(len(x), i+n+1)]) for i in range(len(x))]
    return np.array(y)


def fill_zeros(x):
    """Fill in zeros in a sequence with the previous non-zero number."""
    for i in range(1, len(x)):
        if x[i] == 0:
            x[i] = x[i-1]
    return x


def get_affine_parameters(coords):
    """
    Return rotation and shear for each MTC point along a ``SpectralTrace``.

    .. note: Restrictions of this method:

       * only uses the left most coordinates for the shear
       * rotation angle is calculated using the trace extremes

    Parameters
    ----------
    coords : dict of 2D arrays
        Each dict entry ["x", "y", "s"] contains a [N, M] 2D array of
        coordinates, where:

        * N is the number of points along the slit (e.g. ~5), and
        * M is the number of positions along the trace (e.g. >100)

    Returns
    -------
    rotations : array
        [deg] Rotation angles for M positions along the Trace
    shears : array
        [deg] Shear angles for M positions along the Trace

    """
    rad2deg = 180 / np.pi
    dxs = coords["x"][-1, :] - coords["x"][0, :]
    dys = coords["y"][-1, :] - coords["y"][0, :]
    rotations = np.arctan2(dys, dxs) * rad2deg

    dxs = np.diff(coords["x"], axis=1)
    dys = np.diff(coords["y"], axis=1)
    shears = np.array([np.arctan2(dys[i], dxs[i])
                      for i in range(dxs.shape[0])])
    shears = np.array(list(shears.T) + [shears.T[-1]]).T
    shears = (np.average(shears, axis=0) * rad2deg) - (90 + rotations)

    return rotations, shears

def det_offset(x, offset):
    """Apply offset to x"""
    return x + offset
