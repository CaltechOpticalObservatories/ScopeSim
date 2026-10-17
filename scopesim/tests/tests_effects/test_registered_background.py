import numpy as np
import pytest
from scopesim.detector import Detector
from scopesim.optics.image_plane_utils import header_from_list_of_xy
from astropy.io import fits
from scopesim.effects import ImagePlaneBackground
from scopesim.optics.image_plane import ImagePlane


def detector(nx=120, ny=80):
    header = header_from_list_of_xy([-.015*nx/2, .015*nx/2], [-.015*ny/2, .015*ny/2], .015, 'D')
    return Detector(header)



def test_background_uses_current_named_calibration(tmp_path):
    path = tmp_path / 'background.fits'
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.full((80, 120), .15), name='0.9'), fits.ImageHDU(np.full((80, 120), .04), name='0.9JH')]).writeto(path)
    cmds = {'!INST.slit': '0.9'}
    effect = ImagePlaneBackground(filename=str(path), extname='!INST.slit', cmds=cmds)
    plane = ImagePlane(detector().header)
    effect.apply_to(plane)
    np.testing.assert_allclose(plane.data, .15)
    cmds['!INST.slit'] = '0.9JH'
    plane.data[:] = 0
    effect.apply_to(plane)
    np.testing.assert_allclose(plane.data, .04)
    with pytest.raises(ValueError, match='shape'):
        effect.apply_to(ImagePlane(detector(60, 40).header))
    cmds['!INST.slit'] = 'missing'
    with pytest.raises(KeyError):
        effect.apply_to(plane)

