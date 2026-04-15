from typing import ClassVar, Any
import numpy as np
from astropy.coordinates import SkyCoord, EarthLocation, AltAz, get_sun, get_body, HeliocentricTrueEcliptic
import astropy.units as u
from astropy.time import Time
from astropy.table import Table

from palace import palace

from ..utils import (check_keys, get_logger, airmass2zendist, zendist2airmass, from_currsys, from_rc_config, find_file,
                     get_target, get_location, get_zenith_angle)
from .. import rc
from ..effects import Effect, SkycalcTERCurve, TERCurve

logger = get_logger(__name__)

class PalaceLineEmission(Effect):
    """
    Applies the Paranal Airglow Line and Continuum Emission (PALACE) model TER curves to FoV objects.
    Ref: https://arxiv.org/pdf/2504.10683
    Following input params are needed to run the PALACE model:
    ::
        species=all    # (all, OH, O2, HO2, FeO, Na, K, O, N, H)
        z=0.0          # zenith angle >=0 and <90 deg
        mbin=0         # month number, 0 for all, 1-12 for specific month
        tbin=0         # local (solar mean time at Paranal) time bin, 0 for all, 1 for 18-19h and 12 for 5-6h
        srf=100.0      # solar radio flux at 10.7 cm in sfu (solar flux units), >=0
        isair=True     # True for wavelength in standard air, False for vacuum wavelengths
        isatm=True     # True to include absorption and scattering effects
        pwv=2.5        # precipitable water vapor in mm, >=0
        lammin=0.3     # minimum wavelength in microns, >=0.3
        lammax=2.5     # maximum wavelength in microns, <=2.5
        dlam=1e-06     # wavelength step size in microns, >=1e-7
        resol=100000.0        # spectral resolution, >0
        outdir=zshooter/      # output directory for the model results (not necessary as outputs will be read-in and converted to TER curves in ScopeSim, but can be set if users want to save the model outputs)
        outname=palace_zs     # output file name prefix for the model results
        specsuffix=dat        # output file format suffix for the model results, e.g., dat, fits
        showplot=True         # True to show the model plot, False to skip plotting

    This effect sets the following parameters from simulation settings and instrument configuration:
    :: z, pwv, lammin, lammax, resol, and outdir.

    For remaining parameters, the defaults are as follows:
    - outname is set to "palace" by default, specsuffix to "dat" and showplot to False.
    - mbin and tbin are set by !OBS.mjdobs if available, otherwise default to 0 (all months and times).
    - isair and isatm are set to True by default.
    - species is set to "all", and srf to 130.0 by default, but can be set to specific species if desired.
    - dlam is set to 1e-6 by default.

    Use kwargs to change/set the input parameters to the PALACE model, the parameters set from simulation settings and
    instrument configs will be overwritten by the values from kwargs if provided.
    ::
        name: palace_ter_curves
        class: PalaceLineEmission
        kwargs:
          parlist:
            mbin: 1
            srf: 120.0
          save_model_output: False
          only_line: False
          only_continuum: False

    The output from PALACE are two tables of wavelength and fluxes for the line emission and continuum emission.
    These are read-in and converted to TER curves in ScopeSim's format.
    """
    z_order: ClassVar[tuple[int, ...]] = (112, 512)
    parlist = {"species": "all",
               "z": 0.0,
               "mbin": 0,
               "tbin": 0,
               "srf": 130.0,
               "isair": True,
               "isatm": True,
               "pwv": 2.5,
               "lammin": 0.3,
               "lammax": 2.5,
               "dlam": 1e-6,
               "resol": 100000.0,
               "outdir": "",
               "outname": "palace",
               "specsuffix": "dat",
               "showplot": False
               }

    def __init__(self, **kwargs):
        check_keys(kwargs, self.required_keys, action="error")
        super().__init__(**kwargs)

        ## Set PALACE model output directory if save_model_output is True.
        if self.meta.get("save_model_output", False):
            try:
                atmo_name = [dt.name for dt in self.cmds.yaml_dicts if dt.alias == "!ATMO"][0]
                self.parlist["outdir"] = [pth for pth in rc.__search_path__ if atmo_name in pth][0]
            except:
                self.parlist["outdir"] = f"{from_rc_config("!SIM.file.local_packages_path")}/{self.cmds.package_name}"

        ## Set PALACE model spectral resolution input from simulation settings if provided.
        if check_keys(self.cmds, {"!SIM.spectral.spectral_resolution"}, action="warn"):
            resol = 2*from_currsys("!SIM.spectral.spectral_resolution", self.cmds)
            _dlam = 1.0/resol
            if _dlam < 1e-7:
                logger.warning(f"Spectral resolution {resol} is too high for the PALACE model minimum dlam (1e-7). Setting to 100,000.")
                resol = 100000.0
            elif 1e-7 <= _dlam < 1e-6:
                logger.warning(f"Spectral resolution {resol} is higher than default PALACE dlam (1e-6), setting dlam to 1e-7")
                self.parlist["dlam"] = 1e-7
            self.parlist["resol"] = resol
        if check_keys(self.cmds, {"!SIM.spectral.wave_min", "!SIM.spectral.wave_max"}, action="warn"):
            wave_unit = from_currsys("!SIM.spectral.wave_unit", self.cmds) if "!SIM.spectral.wave_unit" in self.cmds else "um"
            self.parlist["lammin"] = max(from_currsys("!SIM.spectral.wave_min", self.cmds)*u.Unit(wave_unit).to(u.um), 0.3)
            self.parlist["lammax"] = min(from_currsys("!SIM.spectral.wave_max", self.cmds)*u.Unit(wave_unit).to(u.um), 2.5)

        ## Set the rest of the PALACE model input params from sim and config settings.
        #### month and time
        if check_keys(self.cmds, {"!OBS.mjdobs"}, action="warn"):
            obstime = Time(from_currsys("!OBS.mjdobs", self.cmds), format="mjd") ## assuming local time at !ATMO.location
            mbin, tbin = self.get_mbin_tbin(obstime)
        else:
            obstime = None
            mbin, tbin = 0, 0
        self.parlist["mbin"] = mbin
        self.parlist["tbin"] = tbin
        #### zenith
        if check_keys(self.cmds, {"!OBS.alt"}, action="warn"):
            logger.info("Using !OBS.alt to determine zenith angle for PALACE model input.")
            z = 90 - from_currsys("!OBS.alt", self.cmds)
        elif (obstime is not None and
              check_keys(self.cmds, {"!OBS.ra", "!OBS.dec", "!ATMO.longitude", "!ATMO.latitude", "!ATMO.altitude"}, action="warn")):
            logger.info("Using !OBS.ra, !OBS.dec, and !ATMO location info to determine zenith angle for PALACE model input.")
            target = get_target(ra=from_currsys("!OBS.ra", self.cmds),
                                dec=from_currsys("!OBS.dec", self.cmds))
            location = get_location(lon=from_currsys("!ATMO.longitude", self.cmds),
                                    lat=from_currsys("!ATMO.latitude", self.cmds),
                                    alt=from_currsys("!ATMO.altitude", self.cmds))
            z = get_zenith_angle(target, location, obstime)
        elif check_keys(self.cmds, {"!OBS.airmass"}, action="warn"):
            logger.info("Using !OBS.airmass to determine zenith angle for PALACE model input.")
            z = airmass2zendist(from_currsys("!OBS.airmass", self.cmds))
        else:
            logger.warning("No valid input found to determine zenith angle for PALACE model. Defaulting to z=0 (zenith).")
            z = 0.0
        self.parlist["z"] = z
        #### pwv
        self.parlist["pwv"] = from_currsys("!ATMO.pwv", self.cmds) if check_keys(self.cmds, {"!ATMO.pwv"}, action="warn") else 2.5

        ## Update parlist with values from kwargs if provided, otherwise keep the above.
        kwpars = self.meta.get("parlist", {})
        for k, v in kwpars.items():
            if k in self.parlist and k not in ["lammin", "lammax", "resol"]:  # do not allow overwriting of lammin, lammax and resol from sim settings
                self.parlist[k] = v
            else:
                logger.warning(f"Parameter '{k}' is not a valid input parameter for the PALACE model and will be ignored.")
        self.meta["parlist"] = self.parlist

        ## Run palace model and get line and continuum spectra
        spec_cont, spec_line = self.run_palace() # astropy tables with columns "lam" and "flux"
        # assign units to the spectra
        ray_per_nm = (1e10 / (4 * np.pi)) * (u.photon / (u.s * u.m ** 2 * u.steradian * u.nm))

        ## make TER curves and add to the effect
        cont_kwargs = {"array_dict": {"wavelength": spec_cont["lam"].data,
                                      "transmission": [1.0]*len(spec_cont),
                                      "emission": spec_cont["flux"].data * ray_per_nm.to(u.photon / (u.s * u.m ** 2 * u.um * u.arcsec**2))},
                       "wavelength_unit": "um",
                       "emission_unit": "ph s-1 m-2 um-1"}
        self.continuum_TER = TERCurve(**cont_kwargs)
        line_kwargs = {"array_dict": {"wavelength": spec_line["lam"].data,
                                      "transmission": [1.0]*len(spec_line),
                                      "emission": spec_line["flux"].data * ray_per_nm.to(u.photon / (u.s * u.m ** 2 * u.um * u.arcsec**2))},
                       "wavelength_unit": "um",
                       "emission_unit": "ph s-1 m-2 um-1"}
        self.line_TER = TERCurve(**line_kwargs)

    def apply_to(self, obj, **kwargs):
        if self.meta.get("only_line", False):
            obj = self.line_TER.apply_to(obj)
        elif self.meta.get("only_continuum", False):
            obj = self.continuum_TER.apply_to(obj)
        else:
            obj = self.line_TER.apply_to(obj)
            obj = self.continuum_TER.apply_to(obj)
        return obj

    @staticmethod
    def get_mbin_tbin(obstime):
        mbin = obstime.datetime.month
        tbin = obstime.datetime.hour
        if not ((0 <= tbin <= 6) or (18 <= tbin <= 24)):
            logger.warning("Local time is outside of the range covered by the PALACE model (18-6h). Defaulting to tbin=0 (all times).")
            tbin = 0
        return mbin, tbin

    def run_palace(self):
        _, spec_cont, spec_line = palace.model(**self.meta["parlist"])
        if len(spec_cont) == 0:
            logger.warning("PALACE model returned empty continuum spectrum.")
            ## Try loading cont emission from saved model output if available
            spec_cont_file = find_file("palace_zs_cont.dat")
            if spec_cont_file:
                logger.info("Loading PALACE continuum spectrum from saved model output file.")
                spec_cont = Table.read(spec_cont_file, format="ascii")
        if len(spec_line) == 0:
            logger.warning("PALACE model returned empty line spectrum.")
            ## Try loading line emission from saved model output if available
            spec_line_file = find_file("palace_zs_line.dat")
            if spec_line_file:
                logger.info("Loading PALACE line spectrum from saved model output file.")
                spec_line = Table.read(spec_line_file, format="ascii")

        if self.meta.get("save_model_output", False):
            if len(spec_cont) > 0:
                self.meta["parlist"]["outname"] = self.meta["parlist"]["outname"] + "_cont"
                palace.output(spec_cont, **self.meta["parlist"])
            if len(spec_line) > 0:
                self.meta["parlist"]["outname"] = self.meta["parlist"]["outname"].replace("_cont", "_line")
                palace.output(spec_line, **self.meta["parlist"])
            logger.info(f"Saved PALACE models in {self.meta["parlist"]["outdir"]}")

        return spec_cont, spec_line


class SkyBackgroundTERCurve(SkycalcTERCurve):
    """
    Obtains TERCurve for continuum sky background emission from scattered moonlight, starlight and zodiacal light
    from SkyCalc.
    ** DOES NOT INCLUDE AIRGLOW LINES AND RESIDUAL CONTINUUM EMISSION, use Palace_TERCurve for that. **
    ** BY DEFAULT, TRANSMISSION IS DISABLED TO AVOID DOUBLE-COUNTING OF ATMOSPHERIC ABSORPTION EFFECTS, SET disable_transmission TO False TO ENABLE. **

    required parameters:
    - time_str: the time of observation, used to determine moon phase, position, and season and time of night for SkyCalc.
        EITHER "bright", "gray" or "dark", OR a specific time in ISOT or MJD format, e.g., "2024-01-01T00:00:00", 59000, or !OBS.mjdobs.

    The following SkyCalc input parameters are set for this effect:
    - airmass: !OBS.airmass if available, otherwise 1.0, range 1.0-3.0
    - pwv_mode: 'pwv' (or 'season')
    - season: 0 (0 for all, [1-6] for specific, 1=dec/jan ..., used if pwv_mode is 'season')
    - time: 0 (0 for all, [1-3] for specific third of the night, used if pwv_mode is 'season')
    - pwv: !ATMO.pwv if available, otherwise 2.5 mm
    - msolflux: 130.0 sfu
    - incl_moon: "Y" if |z_target-z_moon| < moon-target-sep < |z_target+z_moon|, otherwise "N"
    - moon_sun_sep: set by time input
    - moon_target_sep: set by time input and target coordinates if available, otherwise 30 deg by default
    - moon_alt: set by time input
    - moon_earth_dist: set by time input
    - incl_starlight: "Y"
    - incl_zodiacal: "Y"
    - ecl_lon: target heliocentric ecliptic longitude if available, otherwise 135 deg by default
    - ecl_lat: target ecliptic latitude if available, otherwise 90 deg by default
    - incl_loweratm: "N"   (DO NOT SET TO "Y" UNLESS PALACE MODEL IS DISABLED)
    - incl_upperatm: "N"   (DO NOT SET TO "Y" UNLESS PALACE MODEL IS DISABLED)
    - incl_airglow: "N"    (DO NOT SET TO "Y" UNLESS PALACE MODEL IS DISABLED)
    - vacair: "air"
    - wmin: !SIM.spectral.wave_min if available, otherwise 300 nm
    - wmax: !SIM.spectral.wave_max if available, otherwise 2500 nm
    - wgrid_mode: 'fixed_wavelength_step'
    - wdelta: !SIM.spectral.spectral_bin_width if available, otherwise 0.1 nm
    - wres: !SIM.spectral.spectral_resolution if available, otherwise 80000
    - lsf_type: 'none'
    - observatory: "paranal"  ["paranal, "lasilla", "3060m"]

    e.g.
    ::
    name: continuum_sky_background
    class: SkyBackgroundTERCurve
    kwargs:
        time_str: "bright"
        disable_transmission: True
        target:    # provide either (ra,dec) or (alt,az) or (airmass) for target, otherwise defaults to alt=90,az=0
          ra: !OBS.ra
          dec: !OBS.dec
          alt: !OBS.alt
          az: !OBS.az
          airmass: !OBS.airmass

    """
    z_order: ClassVar[tuple[int, ...]] = (112, 512)
    required_keys = {"time_str"}

    def __init__(self, **kwargs):
        check_keys(kwargs, self.required_keys, action="error")
        self.cmds = kwargs.get("cmds")
        self.time = self.resolve_time(kwargs["time_str"])
        self.location = None
        if check_keys(self.cmds, {"!ATMO.longitude", "!ATMO.latitude", "!ATMO.altitude"}, action="error"):
            self.location = get_location(lon=from_currsys("!ATMO.longitude", self.cmds),
                                    lat=from_currsys("!ATMO.latitude", self.cmds),
                                    alt=from_currsys("!ATMO.altitude", self.cmds))

        target_kwargs: dict[str, Any] = kwargs.get("target", {'alt':90, 'az':0})
        target_kwargs['obstime'] = self.time
        target_kwargs['location'] = self.location
        for k, v in target_kwargs.items():
            if isinstance(v, str) and '!' in v:
                target_kwargs[k] = from_currsys(v, self.cmds)
        self.target = get_target(**target_kwargs)

        skycalc_params = self.get_skycalc_inputs(**kwargs)
        kwargs.update(skycalc_params)
        super().__init__(**kwargs)

        if self.meta.get("disable_transmission", True):
            self.surface.table["transmission"] = np.ones_like(self.surface.table["wavelength"])

    def get_skycalc_inputs(self, **kwargs):
        params = {
            "pwv_mode": "pwv",
            "pwv": from_currsys("!ATMO.pwv", self.cmds) if check_keys(self.cmds, {"!ATMO.pwv"}, action="warn") else 2.5,
            "msolflux": 130.0,
            "incl_starlight": "Y",
            "incl_zodiacal": "Y",
            "incl_loweratm": "N",
            "incl_upperatm": "N",
            "incl_airglow": "N",
            "incl_therm": "N",
            "vacair": "air",
            "wunit": "nm",
            "wgrid_mode": 'fixed_wavelength_step',
            "wres": from_currsys("!SIM.spectral.spectral_resolution", self.cmds) if check_keys(self.cmds, {"!SIM.spectral.spectral_resolution"}, action="warn") else 80000,
        }
        if check_keys(self.cmds, {"!SIM.spectral.wave_min", "!SIM.spectral.wave_max", "!SIM.spectral.spectral_bin_width"}, action="warn"):
            wave_unit = from_currsys("!SIM.spectral.wave_unit", self.cmds) if "!SIM.spectral.wave_unit" in self.cmds else "um"
            params["wmin"] = max(from_currsys("!SIM.spectral.wave_min", self.cmds)*u.Unit(wave_unit).to(u.nm), 300.0)
            params["wmax"] = min(from_currsys("!SIM.spectral.wave_max", self.cmds)*u.Unit(wave_unit).to(u.nm), 2500.0)
            params["wdelta"] = (from_currsys("!SIM.spectral.spectral_bin_width", self.cmds)*u.Unit(wave_unit).to(u.nm))
        else:
            params["wmin"] = 300.0
            params["wmax"] = 2500.0
            params["wdelta"] = 0.01

        ec_frame = HeliocentricTrueEcliptic(obstime=self.time)
        target_ecl = self.target.transform_to(ec_frame)
        params["ecl_lon"] = target_ecl.lon.deg
        params["ecl_lat"] = target_ecl.lat.deg

        moon = get_body("moon", self.time)
        alt_moon = moon.transform_to(AltAz(obstime=self.time, location=self.location)).alt.deg
        z_moon = 90 - alt_moon
        z_target = get_zenith_angle(self.target, self.location, self.time)
        moon_target_sep = moon.separation(self.target).deg
        if (abs(z_target - z_moon) < moon_target_sep) and (moon_target_sep < abs(z_target + z_moon)):
            params.update({
                "airmass": zendist2airmass(z_target),
                "incl_moon": "Y",
                "moon_sun_sep": get_moon_phase(self.time, get_elongation=True).deg,
                "moon_target_sep": moon_target_sep,
                "moon_alt": alt_moon,
                "moon_earth_dist": max(0.91, min(moon.distance.km / 384400.0, 1.08))
            })
        else:
            params["incl_moon"] = "N"
            params["airmass"] = zendist2airmass(z_target)

        if kwargs.get('pwv_mode', None) == 'season':
            if from_currsys("!ATMO.location", self.cmds) not in ["Paranal", "Armazones"]:
                logger.warning("Seasonal PWV mode is only calibrated for Paranal/Armazones. Defaulting to pwv_mode='pwv'.")
            else:
                params["pwv_mode"] = "season"
                params["season"] = self.time.datetime.month//2 + 1 if self.time.datetime.month != 12 else 1
                if 18 <= self.time.datetime.hour <= 24:
                    params["time"] = 1
                elif 0 <= self.time.datetime.hour < 6:
                    params["time"] = 2
                else:
                    params["time"] = 3
        return params

    def resolve_time(self, time_str):
        if isinstance(time_str, int) or isinstance(time_str, float): ## if time_str is a numeric MJD value
            try:
                return Time(time_str, format="mjd")
            except Exception as e:
                logger.warning(f"Failed to parse time from {time_str}: {e}. Defaulting to 'dark'.")
                time_str = "dark"

        if isinstance(time_str, str):
            if '!' in time_str:  ## if time_str is a reference to !OBS.mjdobs
                try:
                    return Time(from_currsys(time_str, self.cmds), format="mjd")
                except Exception as e:
                    logger.warning(f"Failed to parse time from {time_str}: {e}. Defaulting to 'dark'.")
                    time_str = "dark"
            if ('T' in time_str) and (':' in time_str): ## ISOT format
                try:
                    return Time(time_str, format="isot")
                except Exception as e:
                    logger.warning(f"Failed to parse time from {time_str}: {e}. Defaulting to 'dark'.")
                    time_str = "dark"
            if time_str not in ["bright", "gray", "dark"]:
                logger.warning(f"Unrecognized time input: {time_str}. Defaulting to 'dark'.")
                time_str = "dark"
        else:
            logger.warning(f"Invalid time input type: {type(time_str)}, should be a string or numeric MJD value. Defaulting to 'dark'.")
            time_str = "dark"

        kw = {"bright":"full", "gray":"half", "dark":"new"}
        return get_next_moon(kw[time_str])

################################# UTILS FOR MOON ILLUMINATION CALCULATIONS #################################

def get_moon_phase(time: Time, get_elongation=False):
    """
    Calculates moon phase angle and fraction of lunar illumination (FLI) for a given time.
    Phase angle ranges from 0 to 180 degrees, with 0 being full moon and 180 being new moon.
    FLI ranges from 0 to 1, with 0 being new moon and 1 being full moon.
    """
    if isinstance(time, Time):
        sun = get_sun(time)
        moon = get_body("moon", time)
        elongation = sun.separation(moon)
        if get_elongation:
            return elongation
        else:
            return np.arctan2(sun.distance * np.sin(elongation), moon.distance - sun.distance * np.cos(elongation))
    else:
        raise ValueError(f"Invalid time type: {type(time)}, should be astropy Time object.")

def get_moon_fli(phase_angle: u.Quantity):
    """
    Calculates fraction of lunar illumination (FLI) from moon phase angle.
    FLI ranges from 0 to 1, with 0 being new moon and 1 being full moon.
    """
    if isinstance(phase_angle, u.Quantity):
        return (1 + np.cos(phase_angle.to(u.rad).value)) / 2
    else:
        raise ValueError(f"Invalid phase angle type: {type(phase_angle)}, should be astropy Quantity object with angle units.")

def get_next_moon(moontype="full"):
    times = Time.now() + np.linspace(0, 30, 1000)*u.day
    phases = get_moon_phase(times)
    flis = get_moon_fli(phases)
    next_full = times[np.argmax(flis)]
    next_new = times[np.argmin(flis)]
    prev_full = next_full - 29.53*u.day
    prev_new = next_new - 29.53*u.day
    if min(next_new, next_full) - Time.now() > Time.now() - max(prev_new, prev_full):
        next_half = min(next_new, next_full) - 7.38*u.day
    else:
        next_half = min(next_new, next_full) + 7.38*u.day
    if moontype == "full":
        return Time(next_full.isot.split('T')[0]+"T00:00:00")
    elif moontype == "new":
        return Time(next_new.isot.split('T')[0]+"T00:00:00")
    elif moontype == "half":
        return Time(next_half.isot.split('T')[0]+"T00:00:00")
    else:
        raise ValueError(f"Invalid moon type: {moontype}, should be 'full', 'new' or 'half'.")


