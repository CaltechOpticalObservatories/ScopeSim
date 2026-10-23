import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table
from scopesim.effects import SkycalcTERCurve


@pytest.mark.parametrize('wave_unit', [u.nm, u.um])
def test_local_skycalc_respects_wavelength_unit(tmp_path, wave_unit):
    wave = np.array([.5, 1., 2.]) * u.um
    table = Table([wave.to(wave_unit), [.5, .8, .9], [100., 200., 300.]], names=['lam', 'trans', 'flux'])
    path = tmp_path / 'sky.fits'
    table.write(path)
    sky = SkycalcTERCurve(use_local_skycalc_file=str(path))
    np.testing.assert_allclose(sky.skycalc_table['wavelength'], wave.value)
    assert sky.surface.transmission(1*u.um).value == pytest.approx(.8)



def test_skycalc_missing_wavelength_unit_fails(tmp_path):
    path = tmp_path / 'sky.fits'
    Table([[500., 1000., 2000.], [.5, .8, .9], [100., 200., 300.]], names=['lam', 'trans', 'flux']).write(path)
    with pytest.raises(TypeError, match='Unit'):
        SkycalcTERCurve(use_local_skycalc_file=str(path))
