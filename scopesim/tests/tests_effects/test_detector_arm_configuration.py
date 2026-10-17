import numpy as np
import pytest
from scopesim.detector import Detector
from scopesim.optics.image_plane_utils import header_from_list_of_xy
from astropy.io import fits
from scopesim.commands import UserCommands
from scopesim.effects import DetectorList, ExtraFitsKeywords, DetectorModePropertiesSetter
from scopesim.effects.electronic import ADConversion


def detector(nx=120, ny=80):
    header = header_from_list_of_xy([-.015*nx/2, .015*nx/2], [-.015*ny/2, .015*ny/2], .015, 'D')
    return Detector(header)



def test_gain_uses_selected_arm_and_missing_gain_fails():
    det = detector()
    det.data[:] = 12
    ADConversion(gain='!DET.gain_uvb', dtype='float32', cmds={'!DET.gain_uvb': .6}).apply_to(det)
    np.testing.assert_allclose(det.data, 20)
    with pytest.raises(ValueError, match='gain_uvb'):
        ADConversion(gain='!DET.gain_uvb', dtype='float32', cmds={}).apply_to(det)



def test_detector_specific_primary_headers():
    eff = ExtraFitsKeywords(header_dict=[{'detector_id': 1, 'ext_number': [0, 1], 'keywords': {'ARM': 'VIS'}}])
    for detector_id in [0, 1, 2]:
        hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(header=fits.Header({'DET_ID': detector_id}))])
        eff.apply_to(hdul)
        assert ('ARM' in hdul[0].header) == (detector_id == 1)



def test_detector_layout_sets_real_id_header():
    effect = DetectorList(array_dict={'id': [2], 'x_cen': [0.], 'y_cen': [0.], 'x_size': [16], 'y_size': [8], 'pixel_size': [.018], 'angle': [0.], 'gain': [2.29]}, x_cen_unit='mm', y_cen_unit='mm', x_size_unit='pixel', y_size_unit='pixel', pixel_size_unit='mm', angle_unit='deg', gain_unit='electron/adu')
    header = effect.detector_headers()[0]
    assert header['DET_ID'] == 2
    assert header['EXTNAME'] == 'DET_2'
    det = Detector(header)
    assert det.meta['ID'] == 2
    assert det.hdu.header['DET_ID'] == 2



def test_independent_detector_readout_modes():
    cmds = UserCommands(properties={'!DET.mode_blue': 'slow', '!DET.mode_red': 'fast'})
    for arm in ['blue', 'red']:
        modes = {'slow': {f'!DET.gain_{arm}': .6}, 'fast': {f'!DET.gain_{arm}': 1.8}}
        DetectorModePropertiesSetter(mode_properties=modes, mode_key=f'!DET.mode_{arm}', cmds=cmds).apply_to(None)
    assert cmds['!DET.gain_blue'] == .6
    assert cmds['!DET.gain_red'] == 1.8

