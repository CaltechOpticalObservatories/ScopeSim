import numpy as np
import pytest
from scipy.ndimage import map_coordinates
from scopesim.effects.atmo_dispersion import ShiftFoV3D


@pytest.mark.parametrize('shape', [(13, 5, 7), (9, 1, 4), (9, 8, 1)])
def test_chunked_cubic_shift_matches_independent_spatial_planes(shape):
    data = np.random.default_rng(127).uniform(size=shape)
    dx = np.linspace(-2.3, 3.7, shape[0], dtype=np.float32)
    dy = np.linspace(1.8, -.4, shape[0], dtype=np.float32)
    yy, xx = np.indices(shape[1:], dtype=np.float32)
    expected = np.stack([map_coordinates(plane, [yy - sy, xx - sx], order=3, mode='nearest') for plane, sx, sy in zip(data, dx, dy)])
    actual = ShiftFoV3D.map_coords_shift(data, dx, dy, chunk_w=4)
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-6)


def test_spatial_shift_does_not_broaden_a_spectral_line():
    cube = np.zeros((11, 5, 7))
    cube[5, 2, 3] = 1.
    shifted = ShiftFoV3D.map_coords_shift(cube, np.full(11, .3), np.full(11, -.2))
    np.testing.assert_array_equal(shifted[:5], 0.)
    np.testing.assert_array_equal(shifted[6:], 0.)


@pytest.mark.parametrize('dx,dy', [(1, 0), (-1, 0), (0, 1), (0, -1), (2, -2)])
def test_integer_roll_does_not_wrap_light_at_edges(dx, dy):
    cube = np.ones((1, 5, 7))
    shifted = ShiftFoV3D.roll_shift(cube, [dx], [dy])
    assert shifted.sum() == (7 - abs(dx))*(5 - abs(dy))
