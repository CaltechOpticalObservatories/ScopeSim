import numpy as np
import pytest
from scopesim.detector import Detector
from scopesim.optics.image_plane_utils import header_from_list_of_xy
from scopesim.effects.electronic import ReadoutNoiseFromTable, ExposureOutput


def detector(nx=120, ny=80):
    header = header_from_list_of_xy([-.015*nx/2, .015*nx/2], [-.015*ny/2, .015*ny/2], .015, 'D')
    return Detector(header)



@pytest.mark.parametrize('ndit', [1, 4, 9])
def test_nir_averaging_preserves_signal_and_reduces_noise(tmp_path, ndit):
    noise_file = tmp_path / 'noise.dat'
    noise_file.write_text('dit read_noise\n10 12\n100 6\n')
    det = detector(500, 400)
    det.data[:] = 30*ndit
    ReadoutNoiseFromTable(filename=str(noise_file), dit=55, ndit=ndit, random_seed=21).apply_to(det)
    ExposureOutput(dit=55, ndit=ndit, mode='average').apply_to(det)
    assert det.data.mean() == pytest.approx(30, abs=.1)
    assert det.data.std() == pytest.approx(9/np.sqrt(ndit), rel=.02)
    with pytest.raises(ValueError, match='outside'):
        ReadoutNoiseFromTable(filename=str(noise_file), dit=101, ndit=ndit).apply_to(det)
