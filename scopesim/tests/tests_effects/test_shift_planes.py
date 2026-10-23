import numpy as np
from scipy.ndimage import map_coordinates
from scopesim.effects.atmo_dispersion import ShiftFoV3D


def test_dispersion_does_not_mix_wavelengths():
    data = np.zeros((19, 15, 17))
    data[:, 7, 8] = np.arange(19)
    dx, dy = np.linspace(-.2, .2, 19), np.linspace(.4, .6, 19)
    shifted = ShiftFoV3D.map_coords_shift(data, dx, dy)
    yy, xx = np.indices(data.shape[1:])
    for wave in range(19):
        expected = map_coordinates(data[wave], [yy-dy[wave], xx-dx[wave]], order=3, mode='nearest')
        np.testing.assert_allclose(shifted[wave], expected, atol=2e-5)
