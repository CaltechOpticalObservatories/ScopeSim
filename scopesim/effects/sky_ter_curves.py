from typing import ClassVar
import numpy as np
from astropy.coordinates import AltAz, get_body, HeliocentricTrueEcliptic
import astropy.units as u
from astropy.table import Table

from palace import palace

from ..utils import (get_logger, from_currsys, from_rc_config, find_file,
                    zendist2airmass, get_zenith_angle, get_moon_phase, get_observation_info_from_cmds)
from .. import rc
from ..effects import Effect, SkycalcTERCurve, TERCurve

logger = get_logger(__name__)

class PalaceAirglowEmission(Effect):
    """
    Applies the Paranal Airglow Line and Continuum Emission (PALACE) model TER curves to FoV objects.
    Ref: https://arxiv.org/pdf/2504.10683

    The following PALACE input parameters are settable directly through effect kwargs:

    - species: "all" by default # (all, OH, O2, HO2, FeO, Na, K, O, N, H)
    - srf: 130.0 by default # solar radio flux at 10.7 cm in sfu (solar flux units), >=0
    - isair: True by default # True for wavelength in standard air, False for vacuum wavelengths
    - isatm: True by default # True to include absorption and scattering effects
    - pwv: 2.5 mm by default  # precipitable water vapor in mm, >=0
    - lammin: 0.3 um by default  # minimum wavelength in microns, >=0.3
    - lammax: 2.5 um by default  # maximum wavelength in microns, <=2.5
    - outname: "palace" by default # output file name prefix for the model results

    The following PALACE input parameters are set automatically and cannot be set directly:

    - z: set by target info from !OBS # zenith angle >=0 and <90 deg
    - mbin: set by !OBS.mjdobs or !OBS.brightness # month number, 0 for all, 1-12 for specific month
    - tbin: set by !OBS.mjdobs or !OBS.brightness  # local (solar mean time at Paranal) time bin, 0 for all, 1 for 18-19h and 12 for 5-6h
    - resol: set to (2 * !SIM.spectral.spectral_resolution) if provided, otherwise default to 100,000.
    - dlam: 1e-6 by default, 1e-7 (min) if (1/resol < 1e-6), resol changed to 1e6 if (1/resol < 1e-7)
    - outdir: set by package name, used if 'save_model_output' is True
    - specsuffix: "dat" by default
    - showplot: False by default


    Other kwargs:

    - save_model_output: False by default.
    - only_line: False by default. Only applies airglow line emission curve.
    - only_continuum: False by default. Only applies airglow continuum emission curve.
    - lamunit: um by default, update accordingly if setting lammin, lammax

    Example
    --------
    ::

        - name: palace_ter_curves
          class: PalaceAirglowEmission
          kwargs:
            save_model_output: False
            only_line: False
            only_continuum: False
            pwv: "!ATMO.pwv"
            lammin: "!SIM.spectral.wave_min"
            lammax: "!SIM.spectral.wave_max"
            lamunit: "!SIM.spectral.wave_unit"

    The output from PALACE are two tables of wavelength and fluxes for the line emission and continuum emission.
    These are read-in and converted to TER curves in ScopeSim's format.
    """
    z_order: ClassVar[tuple[int, ...]] = (112, 512)
    required_keys = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.target, self.location, self.time = get_observation_info_from_cmds(self.cmds)

        self.parlist = self.get_palace_inputs(**kwargs)

        ## Run palace model and get line and continuum spectra
        spec_cont, spec_line = self.run_palace() # astropy tables with columns "lam" and "flux"
        # assign units to the spectra
        ray_per_nm = (1e10 / (4 * np.pi)) * (u.photon / (u.s * u.m ** 2 * u.steradian * u.nm))

        ## make TER curves and add to the effect
        cont_kwargs = {"name": "palace_airglow_continuum",
                       "array_dict": {"wavelength": spec_cont["lam"].data,
                                      "transmission": [1.0]*len(spec_cont),
                                      "emission": spec_cont["flux"].data * ray_per_nm.to(u.photon / (u.s * u.m ** 2 * u.um * u.arcsec**2))},
                       "wavelength_unit": "um",
                       "emission_unit": "ph s-1 m-2 um-1 arcsec-2"}
        self.continuum_TER = TERCurve(**cont_kwargs)
        line_kwargs = {"name": "palace_airglow_line",
                       "array_dict": {"wavelength": spec_line["lam"].data,
                                      "transmission": [1.0]*len(spec_line),
                                      "emission": spec_line["flux"].data * ray_per_nm.to(u.photon / (u.s * u.m ** 2 * u.um * u.arcsec**2))},
                       "wavelength_unit": "um",
                       "emission_unit": "ph s-1 m-2 um-1 arcsec-2"}
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

    def get_palace_inputs(self, **kwargs):
        parlist = {"species": kwargs.get("species", "all"),
                        "srf": kwargs.get("srf", 130.0),
                        "isair": kwargs.get("isair", True),
                        "isatm": kwargs.get("isatm", True),
                        "pwv": from_currsys(kwargs.get("pwv", 2.5), self.cmds),
                        "outname": kwargs.get("outname", "palace"),
                        "specsuffix": "dat",
                        "showplot": False}

        if "lamunit" in kwargs:
            lamunit = u.Unit(from_currsys(kwargs["lamunit"], self.cmds))
        else:
            logger.warning("Wavelength unit not provided for lammin/lammax, assuming um")
            lamunit = u.um
        parlist["lammin"] = max((from_currsys(kwargs.get("lammin", 0.3), self.cmds) * lamunit).to(u.um).value, 0.3)
        parlist["lammax"] = min((from_currsys(kwargs.get("lammax", 2.5), self.cmds) * lamunit).to(u.um).value, 2.5)

        ## Set PALACE model output directory if save_model_output is True.
        if self.meta.get("save_model_output", False):
            try:
                atmo_name = [dt.name for dt in self.cmds.yaml_dicts if dt.alias == "!ATMO"][0]
                parlist["outdir"] = [pth for pth in rc.__search_path__ if atmo_name in pth][0]
            except:
                parlist["outdir"] = f"{from_rc_config("!SIM.file.local_packages_path")}/{self.cmds.package_name}"

        ## Set PALACE model spectral resolution input from simulation settings if provided.
        resol = 2 * from_currsys("!SIM.spectral.spectral_resolution", self.cmds) if (
                "!SIM.spectral.spectral_resolution" in self.cmds) else 100000.0
        _dlam = 1.0/resol
        if _dlam < 1e-7:
            logger.warning(f"Spectral resolution {resol} is too high for the PALACE model minimum dlam (1e-7). "
                           f"Setting to 1e6.")
            resol = 1000000.0
            dlam = 1e-7
        elif 1e-7 <= _dlam < 1e-6:
            logger.warning(f"Spectral resolution {resol} is higher than default PALACE dlam (1e-6), "
                           f"setting dlam to 1e-7")
            dlam = 1e-7
        else:
            dlam = 1e-6
        parlist["resol"] = resol
        parlist["dlam"] = dlam

        ## month and time
        mbin, tbin = self.get_mbin_tbin(self.time)
        parlist["mbin"] = mbin
        parlist["tbin"] = tbin

        ## zenith
        parlist["z"] = get_zenith_angle(self.target, self.location, self.time)

        return parlist

    @staticmethod
    def get_mbin_tbin(obstime):
        mbin = obstime.datetime.month
        tbin = obstime.datetime.hour
        if not ((0 <= tbin <= 6) or (18 <= tbin <= 24)):
            logger.warning("Local time is outside of the range covered by the PALACE model (18-6h). Defaulting to tbin=0 (all times).")
            tbin = 0
        return mbin, tbin

    def run_palace(self):
        _, spec_cont, spec_line = palace.model(**self.parlist)
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
                self.parlist["outname"] = self.parlist["outname"] + "_cont"
                palace.output(spec_cont, **self.parlist)
            if len(spec_line) > 0:
                self.parlist["outname"] = self.parlist["outname"].replace("_cont", "_line")
                palace.output(spec_line, **self.parlist)
            logger.info(f"Saved PALACE models in {self.parlist["outdir"]}")

        return spec_cont, spec_line


class SkyBackgroundTERCurve(SkycalcTERCurve):
    """
    Applies SkycalcTERCurve for continuum sky background emission from scattered moonlight, starlight and zodiacal light.

    Airglow emission is disabled in the query by default. Transmission is not applied by default.

    Optional kwargs:

    * disable_transmission: True by default
    * disable_airglow: True by default

    The following SkyCalc input parameters can be supplied in kwargs:

    - pwv_mode: "pwv" by default
    - pwv: 2.5 mm by default
    - msolflux: 130.0 sfu by default
    - incl_loweratm: "N" by default, "Y" if 'disable_airglow' is False (DO NOT SET TO "Y" UNLESS PALACE MODEL IS DISABLED)
    - incl_upperatm: "N" by default, "Y" if 'disable_airglow' is False (DO NOT SET TO "Y" UNLESS PALACE MODEL IS DISABLED)
    - incl_airglow: "N" by default, "Y" if 'disable_airglow' is False (DO NOT SET TO "Y" UNLESS PALACE MODEL IS DISABLED)
    - vacair: "air" by default
    - wmin: 0.3 um by default
    - wmax: 2.5 um by default
    - wgrid_mode: 'fixed_wavelength_step' by default
    - wdelta: 0.00001 um by default
    - wres: 80000 by default
    - wunit: "um" by default
    - lsf_type: 'none' by default
    - observatory: "paranal" by default  ["paranal, "lasilla", "3060m"]

    The following SkyCalc input parameters are set automatically and CANNOT be overridden:

    - airmass: set by target info from !OBS
    - season: set by !OBS.mjdobs or !OBS.brightness (used if pwv_mode is 'season')
    - time: set by !OBS.mjdobs or !OBS.brightness (used if pwv_mode is 'season')
    - incl_moon: "Y" if |z_target-z_moon| < moon-target-sep < |z_target+z_moon|, otherwise "N"
    - moon_sun_sep: set by !OBS info
    - moon_target_sep: set by !OBS info
    - moon_alt: set by !OBS info
    - moon_earth_dist: set by !OBS info
    - incl_starlight: "Y"
    - incl_zodiacal: "Y"
    - ecl_lon: set by !OBS info
    - ecl_lat: set by !OBS info
    - incl_therm: "N"

    Example
    --------
    ::

        - name: continuum_sky_background
          class: SkyBackgroundTERCurve
          kwargs:
            disable_transmission: True
            disable_airglow: True
            pwv: "!ATMO.pwv"
            wmin: "!SIM.spectral.wave_min"
            wmax: "!SIM.spectral.wave_max"
            wdelta: "!SIM.spectral.spectral_bin_width"
            wres: "!SIM.spectral.spectral_resolution"
            wunit: "!SIM.spectral.wave_unit"

    """
    z_order: ClassVar[tuple[int, ...]] = (112, 512)
    required_keys = {}

    def __init__(self, **kwargs):
        self.cmds = kwargs.get("cmds")

        self.target, self.location, self.time = get_observation_info_from_cmds(self.cmds)

        skycalc_params = self.get_skycalc_inputs(**kwargs)
        kwargs.update(skycalc_params)
        super().__init__(**kwargs)

        if self.meta.get("disable_transmission", True):
            self.surface.table["transmission"] = np.ones_like(self.surface.table["wavelength"])

    def get_skycalc_inputs(self, **kwargs):
        params = {
            "pwv_mode": kwargs.get("pwv_mode", "pwv"),
            "pwv": from_currsys(kwargs.get("pwv", 2.5), self.cmds),
            "msolflux": kwargs.get("msolflux", 130.0),
            "incl_starlight": "Y",
            "incl_zodiacal": "Y",
            "incl_therm": "N",
            "vacair": kwargs.get("vacair", "air"),
            "wunit": from_currsys(kwargs.get("wunit", "nm"), self.cmds),
            "wgrid_mode": kwargs.get("wgrid_mode", "fixed_wavelength_step"),
            "wmin": from_currsys(kwargs.get("wmin", 300.), self.cmds),
            "wmax": from_currsys(kwargs.get("wmax", 2500.), self.cmds),
            "wdelta": from_currsys(kwargs.get("wdelta", 0.01), self.cmds),
            "wres": from_currsys(kwargs.get("wres", 80000), self.cmds),
        }
        scale_factor = u.Unit(params["wunit"]).to(u.nm)
        for k in ["wmin", "wmax", "wdelta"]:
            params[k] = params[k] * scale_factor
        params["wunit"] = "nm"
        if params["wmin"] < 300.:
            logger.warning(f"wmin {params['wmin']} is below the minimum wavelength covered by SkyCalc. Setting to 300 nm.")
            params["wmin"] = 300.
        if params["wmax"] > 30000.:
            logger.warning(f"wmax {params['wmax']} is above the maximum wavelength covered by SkyCalc. Setting to 30000 nm.")
            params["wmax"] = 30000.

        if kwargs.get("disable_airglow", True):
            params.update({"incl_upperatm": "N", "incl_loweratm": "N", "incl_airglow": "N"})
        else:
            params.update({"incl_upperatm": kwargs.get("incl_upperatm", "N"),
                           "incl_loweratm": kwargs.get("incl_loweratm", "N"),
                           "incl_airglow": kwargs.get("incl_airglow", "N")})

        ec_frame = HeliocentricTrueEcliptic(obstime=self.time)
        target_ecl = self.target.transform_to(ec_frame)
        params["ecl_lon"] = target_ecl.lon.wrap_at(180*u.deg).deg
        params["ecl_lat"] = target_ecl.lat.wrap_at(90*u.deg).deg

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

        if params["pwv_mode"] == 'season':
            if from_currsys("!ATMO.location", self.cmds) not in ["Paranal", "Armazones"]:
                logger.warning("Seasonal PWV mode is only calibrated for Paranal/Armazones. Defaulting to pwv_mode='pwv'.")
                params["pwv_mode"] = 'pwv'
            else:
                params["pwv_mode"] = 'season'
                params["season"] = self.time.datetime.month//2 + 1 if self.time.datetime.month != 12 else 1
                if 18 <= self.time.datetime.hour <= 24:
                    params["time"] = 1
                elif 0 <= self.time.datetime.hour < 6:
                    params["time"] = 2
                else:
                    params["time"] = 3
        return params



