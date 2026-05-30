from typing import ClassVar, Any
from functools import partial
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.table import Table
from astropy.wcs import WCS
from astropy.time import Time
from astropy.coordinates import SkyCoord, HADec
from scipy.interpolate import make_interp_spline, RegularGridInterpolator
from scipy.ndimage import map_coordinates

from .effects import Effect
from ..optics.fov import FieldOfView3D
from ..utils import (from_currsys, get_zenith_angle, get_observation_info_from_cmds, parallactic_angle,
                     get_logger, check_keys, is_night, quantity_from_table)

logger = get_logger(__name__)

class ShiftFoV3D(Effect):
    """
    Shift the FoV by dx(wavelength) and dy(wavelength), that are provided in a file, or as arrays.
    Examples
    --------
    As arrays:-
    ::

        - name: shift
          class: ShiftFoV3D
          kwargs:
              array_dict:
                  wavelength: [0.3, 3.0]
                  dx: [0.05, 0.05]
                  dy: [0.05, 0.05]
              wavelength_unit: um
              dx_unit: arcsec
              dy_unit: arcsec
              method: map_coords

    As file:-
    ::

        - name: shift
          class: ShiftFoV3D
          kwargs:
              filename: "shifts.dat"
              method: map_coords

    which references this ASCII file:-
    ::

        # wavelength_unit: um
        # dx_unit: arcsec
        # dy_unit: arcsec
        wavelength  dx    dy
        0.3         0.05  0.05
        3.0         0.05  0.05

    """
    z_order: ClassVar[tuple[int, ...]] = (50, 650)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.meta.get("method", None) not in [None, "roll", "map_coords"]:
            raise ValueError("Method must be 'roll' or 'map_coords' if provided.")

    def get_shifts(self, obj):
        """
        Shifts calculated from obj (FoV3D) properties or from init data.
        Overwrite this function when sub-classing.
        Should return interp spline values for dx and dy at data wavelength bins in arcsec
        """
        if self.data is not None:
            if isinstance(self.data, Table):
                dx = make_interp_spline(self.data["wavelength"], self.data["dx"])
                dy = make_interp_spline(self.data["wavelength"], self.data["dy"])
            else:
                raise ValueError("Provided shifts data must be in Table format")
        else:
            logger.warning("No shifts as a function of wavelength provided in init.")
            raise NotImplementedError

        swcs = WCS(obj.hdu.header).spectral
        with u.set_enabled_equivalencies(u.spectral()):
            wave = swcs.pixel_to_world(np.arange(swcs.pixel_shape[0])) << u.um
            return dx(wave), dy(wave)

    def apply_to(self, obj, **kwargs):
        if isinstance(obj, FieldOfView3D) and obj.hdu is not None and obj.hdu.header['NAXIS'] == 3:
            dxs, dys = self.get_shifts(obj)

            # get bin widths in spatial axis and divide to get dxs and dys in steps
            dxs = np.round((dxs / (obj.hdu.header["CDELT1"]*u.deg)).decompose().value, 1)
            dys = np.round((dys / (obj.hdu.header["CDELT2"]*u.deg)).decompose().value, 1)

            if self.meta.get("method", None) is None:
                if np.sqrt(np.median(dxs) ** 2 + np.median(dys) ** 2) >= 10:
                    self.meta["method"] = "roll"
                else:
                    self.meta["method"] = "map_coords"
            shift_func = getattr(self, self.meta["method"]+"_shift")

            obj.hdu.data = shift_func(obj.data, dxs, dys)

        return obj

    @staticmethod
    def roll_shift(data, dx, dy):
        """
        Shift each wavelength plane data[w, :, :] by (dx[w], dy[w]) using roll.
        This is a simpler method for when the shift amount is much more than pixel size
        The rolled data is clipped out.
        """
        nw, ny, nx = data.shape
        dx = np.round(np.asarray(dx, dtype=np.float32)).astype(int)
        dy = np.round(np.asarray(dy, dtype=np.float32)).astype(int)
        if dx.shape != (nw,) or dy.shape != (nw,):
            raise ValueError("dx and dy must have shape (nwave,)")

        out = np.empty_like(data)
        for w in range(nw):
            out[w] = np.roll(data[w], (dy[w], dx[w]), axis=(0, 1))
            # zero out the rolled in part, select the correct slice based on sign of dxint, dyint
            xslice = slice(0, dx[w]) if dx[w] > 0 else slice(nx-dx[w], nx)
            yslice = slice(0, dy[w]) if dy[w] > 0 else slice(ny-dy[w], ny)
            out[w][yslice, xslice] = 0
        return out

    @staticmethod
    def map_coords_shift(data, dx, dy, chunk_w: int=8, order: int=3):
        """
        Shift each wavelength plane data[w, :, :] by (dx[w], dy[w]) using
        scipy.ndimage.map_coordinates with chunked processing.

        Parameters
        ----------
        data : np.ndarray, shape (nwave, ny, nx)
            Input data cube.
        dx : np.ndarray, shape (nwave,)
            Shift along x (axis=1) in pixels. Positive moves content toward +x.
        dy : np.ndarray, shape (nwave,)
            Shift along y (axis=0) in pixels. Positive moves content toward +y.
        chunk_w : int
            Number of wavelength planes to process per chunk. Default 8.
        order : int
            Spline interpolation order. Default 3.

        Returns
        -------
        out : np.ndarray, shape (nwave, ny, nx)
        """
        nw, ny, nx = data.shape
        dx = np.asarray(dx, dtype=np.float32)
        dy = np.asarray(dy, dtype=np.float32)
        if dx.shape != (nw,) or dy.shape != (nw,):
            raise ValueError("dx and dy must have shape (nwave,)")

        # Base grids for a single chunk (will be broadcast)
        yi = np.arange(ny, dtype=np.float32)[None, :, None]  # (1, ny, 1)
        xi = np.arange(nx, dtype=np.float32)[None, None, :]  # (1, 1, nx)

        out = np.empty_like(data)
        prefilter = (order > 1)

        for w0 in range(0, nw, chunk_w):
            w1 = min(w0 + chunk_w, nw)
            k = w1 - w0

            # wave index grid for this chunk: (k, 1, 1)
            wi = np.arange(w0, w1, dtype=np.float32)[:, None, None]

            # To shift content by +dx/+dy, sample input at (x - dx, y - dy)
            xcoords = xi - dx[w0:w1][:, None, None]  # (k, 1, nx) -> broadcast to (k, ny, nx)
            ycoords = yi - dy[w0:w1][:, None, None]  # (k, ny, 1) -> broadcast to (k, ny, nx)

            # Broadcast to full chunk volume
            xcoords = np.broadcast_to(xcoords, (k, ny, nx))
            ycoords = np.broadcast_to(ycoords, (k, ny, nx))
            wcoords = np.broadcast_to(wi,      (k, ny, nx))

            # Axis order in coords must match axis order of data: (wave, y, x)
            coords = np.array([wcoords, ycoords, xcoords], dtype=np.float32)  # (3, k, ny, nx)

            out[w0:w1, :, :] = map_coordinates(
                data, coords, order=order, mode='nearest', prefilter=prefilter
            )
        return out


class ADShift(ShiftFoV3D):
    """
    Shift the FoV by the amount of atmospheric dispersion, which depends on the zenith angle, temperature (in deg_C),
    pressure (in bar), humidity (fractional), and wavelength (um).
    Zenith angle of target is determined by !OBS information.

    Example
    -------
    ::

        - name: ad_shift
          class: ADShift
          kwargs:
            temperature: !ATMO.temperature
            pressure: !ATMO.pressure
            humidity: !ATMO.humidity
            wave_ref_um: 0.5

    """
    z_order: ClassVar[tuple[int, ...]] = (651, )
    required_keys = {
        "temperature",
        "humidity",
        "pressure",
        "wave_ref_um",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.target, self.location, self.time = get_observation_info_from_cmds(self.cmds)

        self.zenith_angle = get_zenith_angle(self.target, self.location, self.time) * u.deg

        # partially execute calculation of refractive index for non-changing parameters
        temp = from_currsys(self.meta["temperature"], self.cmds) * u.deg_C
        pressure = from_currsys(self.meta["pressure"], self.cmds) * u.bar
        humidity = from_currsys(self.meta["humidity"], self.cmds)
        humidity = humidity if humidity <=1 else humidity / 100.
        xc = from_currsys(self.meta.get("x_co2", 450.), self.cmds)
        self.n = partial(refractive_index, temp=temp, pressure=pressure, frac_humidity=humidity, xc=xc)

    def apply_to(self, obj, **kwargs):
        return super().apply_to(obj, **kwargs)

    def get_shifts(self, obj: FieldOfView3D):
        shift_arcsec = self._get_shifts_arcsec(obj)

        pos_angle_y = field_rotation_pa_y(obj.hdu.header)
        par_angle = get_parallactic_angle(self.target, self.location, self.time)
        theta = np.deg2rad(par_angle - pos_angle_y) # angle of zenith direction from +y-axis counter-clockwise
        logger.info(f'Direction of zenith, measured counter-clockwise from +y axis is {int(par_angle - pos_angle_y)}.')

        dy = shift_arcsec * np.cos(theta)
        dx = -1 * shift_arcsec * np.sin(theta)
        return dx, dy

    def _get_shifts_arcsec(self, obj: FieldOfView3D):
        wave_ref = from_currsys(self.meta["wave_ref_um"], self.cmds) * u.um
        n_ref = self.n(wave_ref)

        swcs = WCS(obj.hdu.header).spectral
        with u.set_enabled_equivalencies(u.spectral()):
            wave = swcs.pixel_to_world(np.arange(swcs.pixel_shape[0])) << u.um
        n_wave = self.n(wave)
        delta_n = n_wave - n_ref
        return 206265 * delta_n * np.tan(self.zenith_angle.to(u.rad).value) * u.arcsec


class ADCShift(ShiftFoV3D):
    """
    Shift the FoV3D by the amount of ADC.
    Supports:
      1. Supplying the residuals using 'filename' kwarg. No need for AD shift to be applied first, directly shift by
         correction residuals
      2. Calculating ADC shifts as opposite of AD shift but with an added error in zenith angle value. The residuals
         are then calculated as AD_Shift(zenith_angle) - AD_Shift(zenith_angle + zenith_angle_error). Used if
         'zenith_angle_error' is supplied. If kwargs required by ADShift effect are not supplied, currsys values are used.
      3. Both of the above.

    Example
    --------
    ::

        - name: adc_shift
          class: ADCShift
          kwargs:
            filename: "adc_residuals.dat"
            use_broadband: False
            zenith_angle_error: 0.1  # in degrees
            wave_ref_um: 0.5

    where adc_residuals.dat has wavelength as first column, and residuals at different zenith angles in subsequent
    columns with zenith angle (in degrees) as the column name-
    ::

        # wavelength_unit: um
        # shifts_unit: arcsec
        wavelength  0.0   60.0
        0.3         0      0.1
        3.0         0      0.1

    """
    z_order = (652, )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if 'filename' not in kwargs and 'zenith_angle_error' not in kwargs:
            raise ValueError("Residuals must be supplied through either filename or zenith_angle_error.")

        self.target, self.location, self.time = get_observation_info_from_cmds(self.cmds)

        self.zenith_angle = get_zenith_angle(self.target, self.location, self.time) * u.deg
        self._ad_shift_cache_key = None
        self._ad_shift_cache = None
        self._residual_interpolator_cache = None

    def get_shifts(self, obj: FieldOfView3D):
        """
        Computes an AD correction residual for a given zenith angle.
        """
        swcs = WCS(obj.hdu.header).spectral
        with u.set_enabled_equivalencies(u.spectral()):
            wave = swcs.pixel_to_world(np.arange(swcs.pixel_shape[0])) << u.um
        shifts = np.zeros_like(wave.value) * u.arcsec

        if self.data is not None and isinstance(self.data, Table):
            logger.info(f'Residuals supplied by {self.meta["filename"]}')
            R_unit, adc_opt_resid = self._get_residual_interpolator()

            if self.meta.get("use_broadband", False):
                use = wave < 1.0 * u.um
                shifts[use] += (adc_opt_resid(
                    (self.zenith_angle.to_value(u.deg), wave[use].to_value(u.um))) * R_unit).to(u.arcsec)
            else:
                use = wave < 0.5 * u.um
                res1 = (self._adc_ub_resid(adc_opt_resid,
                    (self.zenith_angle.to_value(u.deg), wave[use].to_value(u.um))) * R_unit).to(u.arcsec)
                shifts[use] += res1
                use = ~use & (wave < 1.0 * u.um)
                res2 = (self._adc_gri_resid(adc_opt_resid,
                    (self.zenith_angle.to_value(u.deg), wave[use].to_value(u.um))) * R_unit).to(u.arcsec)
                shifts[use] += res2
            use = wave >= 1.0 * u.um
            res3 = (self._adc_nir_resid(adc_opt_resid,
                    (self.zenith_angle.to_value(u.deg), wave[use].to_value(u.um))) * R_unit).to(u.arcsec)
            shifts[use] += res3

        if self.meta.get('zenith_angle_error', 0.0) != 0.0:
            logger.info(f'Additional residual from zenith angle error: {self.meta["zenith_angle_error"]}')
            ad_kwargs = {}
            ad_defaults = {'temperature':'!ATMO.temperature', 'pressure':'!ATMO.pressure', 'humidity':'!ATMO.humidity',
                       'x_co2':'!ATMO.x_co2', 'wave_ref_um':0.5}
            for k, v in ad_defaults.items():
                if k not in self.meta:
                    ad_kwargs[k] = v
                else:
                    ad_kwargs[k] = self.meta[k]
            ad = self._get_ad_shift(ad_kwargs)
            ad.zenith_angle = self.zenith_angle
            ad_shift = ad._get_shifts_arcsec(obj)  # get shift at zenith angle
            ad.zenith_angle = ad.zenith_angle + self.meta.get('zenith_angle_error', 0.0) * u.deg # update zenith angle
            ad_shift -= ad._get_shifts_arcsec(obj)  # subtract shift at (zenith angle + error) to get residual
            shifts += ad_shift
            ad.zenith_angle = self.zenith_angle

        pos_angle_y = field_rotation_pa_y(obj.hdu.header)
        par_angle = get_parallactic_angle(self.target, self.location, self.time)
        theta = np.deg2rad(par_angle - pos_angle_y) # angle of zenith direction from +y-axis counter-clockwise
        logger.info(f'Direction of zenith, measured counter-clockwise from +y axis is {int(par_angle - pos_angle_y)}.')
        dy = shifts * np.cos(theta)
        dx = -1 * shifts * np.sin(theta)
        return dx, dy

    def _get_residual_interpolator(self):
        if self._residual_interpolator_cache is None:
            Z = list(self.data.colnames)
            Z.remove("wavelength")
            lam_um = quantity_from_table("wavelength", self.data, "um").value
            R = np.array([np.array(self.data[z]).astype(float) for z in Z])
            R_unit = u.Unit(self.data.meta.get("shifts_unit", "arcsec"))
            Z = np.array(Z).astype(float)
            adc_opt_resid = RegularGridInterpolator(
                (Z, lam_um), R, method="linear", bounds_error=False,
                fill_value=None,
            )
            self._residual_interpolator_cache = (R_unit, adc_opt_resid)
        return self._residual_interpolator_cache

    @staticmethod
    def _adc_nir_resid(adc_opt_resid, xy):
        return adc_opt_resid((xy[0], (xy[1] - 1) / 1.5 * (1.1 - .31) + .31))

    @staticmethod
    def _adc_ub_resid(adc_opt_resid, xy):
        return adc_opt_resid((xy[0], (xy[1] - .3) / .2 * (1.1 - .31) + .31)) / 2

    @staticmethod
    def _adc_gri_resid(adc_opt_resid, xy):
        return adc_opt_resid((xy[0], (xy[1] - .5) / .5 * (1.1 - .31) + .31)) / 2

    def _get_ad_shift(self, ad_kwargs):
        cache_key = tuple(
            (key, str(from_currsys(value, self.cmds)))
            for key, value in sorted(ad_kwargs.items())
        )
        if self._ad_shift_cache is None or cache_key != self._ad_shift_cache_key:
            self._ad_shift_cache = ADShift(**ad_kwargs, cmds=self.cmds)
            self._ad_shift_cache_key = cache_key
        return self._ad_shift_cache


########################### AD utils ###############################
def field_rotation_pa_y(header) -> float:
    """
    Returns PA of detector +Y axis on sky, in deg (N->E convention).
    0 deg = Y points north, 90 deg = Y points east
    """
    w = WCS(header).dropaxis(-1)
    # local anchor (image center)
    x0, y0 = w.wcs.crpix
    lon0, lat0 = w.pixel_to_world(x0, y0)
    lon1, lat1 = w.pixel_to_world(x0, y0+1)
    p0 = SkyCoord(lon0, lat0, unit=(u.deg, u.deg))
    p1 = SkyCoord(lon1, lat1, unit=(u.deg, u.deg))
    return p0.position_angle(p1).to(u.deg).value

def get_parallactic_angle(target: SkyCoord, location: EarthLocation, time: Time) -> float:
    """
    Returns parallactic angle in degrees.
    """
    hadec_target = target.transform_to(HADec(obstime=time, location=location))
    return parallactic_angle(hadec_target.ha.hour, hadec_target.dec.deg, location.lat.deg)

def compressibility(temp: u.Quantity, pressure: u.Quantity, x_w: float):
    """
    See Ciddor 1996, Appendix A, Equation 12, https://doi.org/10.1364/AO.35.001566
    :param temp: temperature
    :param pressure: total pressure
    :param x_w: molar fraction of water vapor in moist air
    :return:
    """
    t = temp.to(u.deg_C, equivalencies=u.temperature()).value
    T = temp.to(u.K, equivalencies=u.temperature()).value
    p = pressure.to(u.Pa).value
    a0 = 1.58123e-6  # K·Pa^-1
    a1 = -2.9331e-8  # Pa^-1
    a2 = 1.1043e-10  # K^-1·Pa^-1
    b0 = 5.707e-6  # K·Pa^-1
    b1 = -2.051e-8  # Pa^-1
    c0 = 1.9898e-4  # K·Pa^-1
    c1 = -2.376e-6  # Pa^-1
    d = 1.83e-11  # K^2·Pa^-2
    e = -0.765e-8  # K^2·Pa^-2
    return 1 - (p / T) * (a0 + a1 * t + a2 * t ** 2 + (b0 + b1 * t) * x_w + (c0 + c1 * t) * x_w ** 2) + (p / T) ** 2 * (
                d + e * x_w ** 2)

def refractive_index(wavelength: u.Quantity, temp: u.Quantity, pressure: u.Quantity, frac_humidity: float,
                     xc: float):
    """
    See Ciddor 1996, https://doi.org/10.1364/AO.35.001566
    :param wavelength: wavelength, 0.3 to 1.69 um
    :param temp: temperature, -40 to +100 °C
    :param pressure: pressure, 80000 to 120000 Pa
    :param frac_humidity: fractional humidity, 0 to 1
    :param xc: CO2 concentration in ppm, 0 to 2000 ppm
    :return: nprop: refractive index
    """
    lam = wavelength.to(u.um).value
    σ = 1 / lam  # μm^-1

    t = temp.to(u.deg_C, equivalencies=u.temperature()).value  # t in °C
    T = temp.to(u.K, equivalencies=u.temperature()).value  # Temperature °C -> K
    p = pressure.to(u.Pa).value

    R = 8.314510  # gas constant, J/(mol·K)

    k0 = 238.0185  # μm^-2
    k1 = 5792105  # μm^-2
    k2 = 57.362  # μm^-2
    k3 = 167917  # μm^-2

    w0 = 295.235  # μm^-2
    w1 = 2.6422  # μm^-2
    w2 = -0.032380  # μm^-4
    w3 = 0.004028  # μm^-6

    A = 1.2378847e-5  # K^-2
    B = -1.9121316e-2  # K^-1
    C = 33.93711047
    D = -6.3431645e3  # K

    α = 1.00062
    β = 3.14e-8  # Pa^-1,
    γ = 5.6e-7  # °C^-2

    # saturation vapor pressure of water vapor in air at temperature T (Pa)
    svp = np.where(t >= 0,
                   np.exp(A * T ** 2 + B * T + C + D / T),  # if t>=0
                   10 ** (-2663.5 / T + 12.537))  # if t<0

    # enhancement factor of water vapor in air
    f = α + β * p + γ * t ** 2

    # molar fraction of water vapor in moist air
    xw = f * frac_humidity * svp / p

    # refractive index of standard air at 15 °C, 101325 Pa, 0% humidity, 450 ppm CO2
    nas = 1 + (k1 / (k0 - σ ** 2) + k3 / (k2 - σ ** 2)) * 1e-8

    # refractive index of standard air at 15 °C, 101325 Pa, 0% humidity, xc ppm CO2
    naxs = 1 + (nas - 1) * (1 + 0.534e-6 * (xc - 450))

    # refractive index of water vapor at standard conditions (20 °C, 1333 Pa)
    nws = 1 + 1.022 * (w0 + w1 * σ ** 2 + w2 * σ ** 4 + w3 * σ ** 6) * 1e-8

    Ma = 1e-3 * (28.9635 + 12.011e-6 * (xc - 400))  # molar mass of dry air, kg/mol
    Mw = 0.018015  # molar mass of water vapor, kg/mol

    Za = compressibility(288.15*u.K, 101325*u.Pa, 0)  # compressibility of dry air
    Zw = compressibility(293.15*u.K, 1333*u.Pa, 1)  # compressibility of pure water vapor

    # Eq.4 with (T,P,xw) = (288.15, 101325, 0)
    ρaxs = 101325 * Ma / (Za * R * 288.15)  # density of standard air

    # Eq 4 with (T,P,xw) = (293.15, 1333, 1)
    ρws = 1333 * Mw / (Zw * R * 293.15)  # density of standard water vapor

    # two parts of Eq.4: ρ=ρa+ρw
    Z = compressibility(temp, pressure, xw)
    ρa = p * Ma / (Z * R * T) * (1 - xw)  # density of the dry component of the moist air
    ρw = p * Mw / (Z * R * T) * xw  # density of the water vapor component

    nprop = 1 + (ρa / ρaxs) * (naxs - 1) + (ρw / ρws) * (nws - 1)

    return nprop
