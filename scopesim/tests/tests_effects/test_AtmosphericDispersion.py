import pytest
from pytest import approx

import numpy as np

from astropy import units as u

from scopesim.effects.atmo_dispersion import (ADCShift, ShiftFoV3D,
                                              refractive_index, compressibility)

from scopesim.tests.tests_optics.test_slit_spatial_integration import physical_fov
from scopesim.effects.shifts import AtmosphericDispersion, \
                                    get_pixel_border_waves_from_atmo_disp

from ..mocks.py_objects.yaml_objects import _atmo_yaml_dict


def test_repeated_integer_adc_shifts_preserve_positions_and_clip_edges():
    cube = np.zeros((10, 5, 7))
    cube[:, 2, 3] = 1
    dx = np.array([-1.2, -1.2, 0., 0., .9, .9, 8., -8., 0., 0.])
    dy = np.array([1., 1., 0., 0., -1., -1., 0., 0., 6., -6.])
    shifted = ADCShift.roll_shift(cube, dx, dy)
    expected = np.zeros_like(cube)
    expected[:2, 3, 2] = 1
    expected[2:4, 2, 3] = 1
    expected[4:6, 1, 4] = 1
    np.testing.assert_array_equal(shifted, expected)


@pytest.fixture(scope="function")
def atmo_yaml_dict():
    atmo_dict = _atmo_yaml_dict()
    atmo_dict["properties"].update({"pupil_angle": 0, "airmass": 1.14,
                                    "altitude": 2400, "temperature": 7,
                                    "pressure": 0.755,
                                    "pixel_scale": 0.004,
                                    "wave_min": 0.5,
                                    "wave_mid": 1.5,
                                    "wave_max": 2.5,
                                    "sub_pixel_fraction": 1,
                                    "pixel_scale": 0.004,
                                    })      # "!INST.pixel_scale"

    return atmo_dict


class TestInit:
    def test_throws_error_when_initialised_with_nothing(self):
        with pytest.raises(ValueError):
            isinstance(AtmosphericDispersion(), AtmosphericDispersion)

    def test_passes_if_all_keywords_are_present(self, atmo_yaml_dict):
        kwargs_dict = atmo_yaml_dict["properties"]
        atmo_disp = AtmosphericDispersion(**kwargs_dict)

        assert isinstance(atmo_disp, AtmosphericDispersion)

    def test_throws_error_if_one_keyword_missing(self, atmo_yaml_dict):
        kwargs_dict = atmo_yaml_dict["properties"]
        kwargs_dict.pop("airmass")
        with pytest.raises(ValueError):
            isinstance(AtmosphericDispersion(), AtmosphericDispersion)


class TestGetTable:
    def test_returns_wavelength_dx_dy_columns(self, atmo_yaml_dict):
        atmo_disp = AtmosphericDispersion(**atmo_yaml_dict["properties"])
        tbl = atmo_disp.get_table()
        assert tbl.colnames == ["wavelength", "dx", "dy"]
        assert all(isinstance(tbl[col], np.ndarray) for col in tbl.colnames)

    def test_returns_similar_values_to_lasilla_website(self, atmo_yaml_dict):
        atmo_disp = AtmosphericDispersion(**atmo_yaml_dict["properties"])
        tbl = atmo_disp.get_table()
        waves, dx, dy = tbl["wavelength"], tbl["dx"], tbl["dy"]
        assert dy[0] - dy[-1] == approx(0.53, rel=1e-2)
        assert all(dx == 0)
        assert waves[0] == 0.5 and waves[-1] == 2.5

    def test_returns_same_results_when_turned_90_degrees(self, atmo_yaml_dict):
        atmo_yaml_dict["properties"]["pupil_angle"] = 90
        atmo_disp = AtmosphericDispersion(**atmo_yaml_dict["properties"])
        tbl = atmo_disp.get_table()
        dx, dy = tbl["dx"], tbl["dy"]
        assert dx[0] - dx[-1] == approx(0.53, rel=1e-2)
        assert all([y == approx(0) for y in dy])

    def test_returns_same_results_when_turned_30_degrees(self, atmo_yaml_dict):
        atmo_yaml_dict["properties"]["pupil_angle"] = 30
        atmo_disp = AtmosphericDispersion(**atmo_yaml_dict["properties"])
        tbl = atmo_disp.get_table()
        dx, dy = tbl["dx"], tbl["dy"]
        dr = ((dx[0] - dx[-1])**2 + (dy[0] - dy[-1])**2)**0.5
        assert dr == approx(0.53, rel=1e-2)


class TestGetPixelBorderWavesFromAtmoDisp:
    """
    http://gtc-phase2.gtc.iac.es/science/astroweb/atmosRefraction.php
    airmass = 1.14
    altitude = 2400m
    temperature = 7 deg
    pressure = 0.755 bar
    humidity = ?

    Approx atmospheric refraction at 500nm = 24.8 arcsec
    Diff atmo refr relative to 500nm
    - 0.5 : 0 arcsec
    - 1.5 : -0.49 arcsec
    - 2.5 : -0.53 arcsec

    """

    def test_returns_sensible_data(self):
        atmo_params = {"z0": 28.7,               # in deg
                       "temp": 7,                # in degC
                       "rel_hum": 100,              # in %
                       "pres": 755,              # in mbar
                       "lat": -26,              # in deg
                       "h": 2400,             # in m
                       "wave_min": 0.5,             # in um
                       "wave_mid": 1.5,
                       "wave_max": 2.5,
                       "pixel_scale": 0.004,        # in arcsec
                       "sub_pixel_fraction": 1,
                       "num_steps": 1000}

        waves, shifts = get_pixel_border_waves_from_atmo_disp(**atmo_params)

        assert shifts[0] - shifts[-1] == approx(0.53, rel=2e-2)
        assert waves[0] == 0.5 and waves[-1] == 2.5

        atmo_params["wave_max"] = 1.5
        waves, shifts = get_pixel_border_waves_from_atmo_disp(**atmo_params)
        assert shifts[0] - shifts[-1] == approx(0.49, rel=2e-2)
        assert waves[0] == 0.5 and waves[-1] == 1.5



class TestCiddorRefractiveIndex:
    """Reference values from Ciddor 1996, https://doi.org/10.1364/AO.35.001566"""

    def test_standard_air_matches_published_refractive_index(self):
        # Ciddor's reference condition: 633 nm, 15 deg C, 101325 Pa, dry air,
        # 450 ppm CO2, for which the published n is 1.00027653.
        n = refractive_index(0.633*u.um, 15*u.deg_C, 101325*u.Pa, 0.0, 450.)
        assert n == approx(1.00027653, abs=5e-9)

    @pytest.mark.parametrize("wavelength, expected", [(0.35, 1.00028612), (0.633, 1.00027653), (1.5, 1.00027330)])
    def test_normal_dispersion_across_the_band(self, wavelength, expected):
        # n falls monotonically with wavelength away from any resonance.
        n = refractive_index(wavelength*u.um, 15*u.deg_C, 101325*u.Pa, 0.0, 450.)
        assert n == approx(expected, abs=5e-9)

    def test_moist_air_is_less_refractive_than_dry_air(self):
        dry = refractive_index(0.633*u.um, 20*u.deg_C, 101325*u.Pa, 0.0, 450.)
        saturated = refractive_index(0.633*u.um, 20*u.deg_C, 101325*u.Pa, 1.0, 450.)
        assert saturated < dry
        assert (dry - saturated) == approx(8.503e-7, rel=1e-3)

    def test_refractivity_is_nearly_proportional_to_pressure(self):
        full = refractive_index(0.633*u.um, 15*u.deg_C, 101325*u.Pa, 0.0, 450.)
        half = refractive_index(0.633*u.um, 15*u.deg_C, 50000*u.Pa, 0.0, 450.)
        assert (half - 1)/(full - 1) == approx(50000/101325, rel=1e-3)

    def test_dry_air_compressibility_matches_ciddor_appendix(self):
        # Ciddor quotes Z_a = 0.9995922115 at 288.15 K and 101325 Pa.
        assert compressibility(288.15*u.K, 101325*u.Pa, 0) == approx(0.9995922115, abs=1e-10)

    def test_compressibility_approaches_the_ideal_gas_limit(self):
        assert compressibility(288.15*u.K, 1*u.Pa, 0) == approx(1.0, abs=1e-8)


class TestShiftMethodIsConfigured:
    def test_missing_method_is_an_error(self):
        # The interpolation method must not be guessed from the first FOV.
        with pytest.raises(ValueError, match="method"):
            ShiftFoV3D()

    def test_unknown_method_is_an_error(self):
        with pytest.raises(ValueError, match="roll"):
            ShiftFoV3D(method="nearest")

    @pytest.mark.parametrize("method", ["roll", "map_coords"])
    def test_configured_method_is_accepted(self, method):
        assert ShiftFoV3D(method=method).meta["method"] == method

    def test_adc_residual_shifts_inherit_the_configured_method(self):
        # ADCShift builds an inner ADShift to get the uncorrected dispersion;
        # that inner effect must not need its own method configured.
        from scopesim.commands import UserCommands
        cmds = UserCommands(properties={
            "!ATMO.longitude": -70.19, "!ATMO.latitude": -24.59, "!ATMO.altitude": 3060,
            "!ATMO.temperature": 0., "!ATMO.pressure": 0.6, "!ATMO.humidity": 0.1,
            "!ATMO.x_co2": 450., "!OBS.mjdobs": 60310.25, "!OBS.alt": 60, "!OBS.az": 0,
        })
        adc = ADCShift(method="roll", zenith_angle_error=0.1, wave_ref_um=0.5, cmds=cmds)
        assert adc.meta["method"] == "roll"

        # Exercising get_shifts is the point: it is what builds the inner
        # ADShift, which used to be constructed without a method.
        dxs, dys = adc.get_shifts(physical_fov(.7))
        assert len(dxs) == len(dys)
        assert np.any(np.abs(dys.to_value(u.arcsec)) > 0)
