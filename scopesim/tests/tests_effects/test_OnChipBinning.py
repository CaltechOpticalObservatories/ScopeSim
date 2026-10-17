import numpy as np
import pytest
from scopesim.detector import Detector
from scopesim.optics.image_plane_utils import header_from_list_of_xy
from astropy.wcs import WCS
from scopesim.effects.electronic import OnChipBinning, BasicReadoutNoise


def detector(nx=120, ny=80):
    header = header_from_list_of_xy([-.015*nx/2, .015*nx/2], [-.015*ny/2, .015*ny/2], .015, 'D')
    return Detector(header)



@pytest.mark.parametrize('bx,by', [(1, 1), (1, 2), (2, 2)])
def test_binning_charge_and_focal_coordinates(bx, by):
    det = detector()
    det.data[:] = 7
    original = WCS(det.header, key='D')
    OnChipBinning(binx=bx, biny=by).apply_to(det)
    assert det.data.shape == (80//by, 120//bx)
    assert np.all(det.data == 7*bx*by)
    assert det.data.sum() == 7*120*80
    binned = WCS(det.hdu.header, key='D')
    expected = original.all_pix2world((np.array([10, 20])+.5)*bx-.5, (np.array([12, 23])+.5)*by-.5, 0)
    np.testing.assert_allclose(binned.all_pix2world([10, 20], [12, 23], 0), expected, atol=1e-12)
    det.reset()
    assert det.data.shape == (80, 120)
    assert det.header['CDELT2D'] == .015
    OnChipBinning(binx=bx, biny=by).apply_to(det)
    assert det.data.shape == (80//by, 120//bx)



def test_binning_rotated_cd_matrix():
    det = detector()
    det.header['CD1_1D'], det.header['CD1_2D'] = .012, -.009
    det.header['CD2_1D'], det.header['CD2_2D'] = .009, .012
    wcs = WCS(det.header, key='D')
    OnChipBinning(binx=1, biny=2).apply_to(det)
    actual = WCS(det.header, key='D').all_pix2world(9, 15, 0)
    np.testing.assert_allclose(actual, wcs.all_pix2world(9, 30.5, 0), atol=1e-12)



def test_binning_rejects_unsupported_modes():
    with pytest.raises(ValueError, match='Unsupported'):
        OnChipBinning(binx=2, biny=1, allowed_bins=[[1, 1], [1, 2], [2, 2]]).apply_to(detector())
    with pytest.raises(ValueError):
        OnChipBinning(binx=1, biny=3).apply_to(detector())



def test_ccd_bin_does_not_multiply_amplifier_noise():
    det = detector(600, 400)
    OnChipBinning(binx=2, biny=2).apply_to(det)
    BasicReadoutNoise(noise_std=3, ndit=1, random_seed=17).apply_to(det)
    assert det.data.std() == pytest.approx(3, rel=.02)

