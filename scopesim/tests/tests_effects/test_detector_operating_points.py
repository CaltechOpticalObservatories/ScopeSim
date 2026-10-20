import warnings

import numpy as np
import pytest

from scopesim import UserCommands
from scopesim.detector import Detector
from scopesim.effects.electronic import AvalancheShotNoise, DarkCurrent, ExposureIntegration, ExposureLimit, OnChipBinning, BasicReadoutNoise
from scopesim.optics.image_plane_utils import header_from_list_of_xy


@pytest.mark.parametrize('factor', [1., 1.2])
@pytest.mark.parametrize('photons,dark', [(0., 4.), (.2, .01), (50., 3.)])
def test_avalanche_preserves_physical_mean_and_only_amplifies_photon_variance(factor, photons, dark):
    cmds = UserCommands(properties={'!SIM.random.seed': 7143})
    effect = AvalancheShotNoise(excess_noise_factor=factor, dark_current=dark, dit=1., ndit=1, cmds=cmds)
    data = effect(np.full(400000, photons + dark), dark_charge=dark, det_id=2)
    mean, variance = photons + dark, factor * photons + dark
    assert data.mean() == pytest.approx(mean, abs=5*np.sqrt(variance/data.size))
    assert data.var() == pytest.approx(variance, rel=.025)


def test_independent_integrations_and_read_noise_after_avalanche():
    cmds = UserCommands(properties={'!SIM.random.seed': 736})
    count, photons, dark, ron, factor = 12, 7., .4, .5, 1.2
    effect = AvalancheShotNoise(excess_noise_factor=factor, dark_current=dark, dit=1., ndit=count, cmds=cmds)
    data = effect(np.full(400000, count*(photons+dark)), dark_charge=count*dark, det_id=3)
    data = BasicReadoutNoise(noise_std=ron, ndit=count, cmds=cmds)(data, ndit=count, det_id=3)
    assert data.mean() == pytest.approx(count*(photons+dark), abs=.1)
    assert data.var() == pytest.approx(count*(factor*photons + dark + ron**2), rel=.02)
    cmds['!OBS.roid'] = 2
    other = effect(np.full(400000, count*(photons+dark)), dark_charge=count*dark, det_id=3)
    assert not np.array_equal(data, other)


def test_exposure_limit_warns_after_charge_binning_without_changing_counts_or_commands():
    cmds = UserCommands(properties={'!OBS.dit': 10., '!OBS.ndit': 4})
    header = header_from_list_of_xy([-2, 2], [-2, 2], 1., 'D')
    header['ID'] = 0
    detector = Detector(header, cmds=cmds)
    detector.data[:] = 400.
    limit = ExposureLimit(usable_well=15000., ndit='!OBS.ndit', cmds=cmds)
    ExposureIntegration(dit='!OBS.dit', ndit='!OBS.ndit', cmds=cmds).apply_to(detector)
    DarkCurrent(value=0., dit='!OBS.dit', ndit='!OBS.ndit', cmds=cmds).apply_to(detector)
    with warnings.catch_warnings(record=True) as caught:
        limit.apply_to(detector)
    assert not caught  # Summed 16000 e is only 4000 e per integration.
    OnChipBinning(binx=2, biny=2).apply_to(detector)
    before = detector.data.copy()
    with pytest.warns(UserWarning, match='peak 16000.0 e.*DIT/NDIT are unchanged'):
        limit.apply_to(detector)
    np.testing.assert_array_equal(detector.data, before)
    assert detector.header['PEAKDIT'] == 16000
    assert cmds['!OBS.dit'] == 10 and cmds['!OBS.ndit'] == 4



def test_avalanche_supports_unseeded_readouts():
    cmds = UserCommands(properties={'!SIM.random.seed': None})
    effect = AvalancheShotNoise(excess_noise_factor=1.2, dark_current=0., dit=1., ndit=1, cmds=cmds)
    first = effect(np.full(1000, 10.), det_id=3)
    second = effect(np.full(1000, 10.), det_id=3)
    assert not np.array_equal(first, second)


def test_detector_ids_produce_independent_read_noise():
    cmds = UserCommands(properties={'!SIM.random.seed': 7143})
    effect = BasicReadoutNoise(noise_std=2., ndit=1, cmds=cmds)
    header = header_from_list_of_xy([-50, 50], [-50, 50], 1., 'D')
    header['ID'] = 1
    first = Detector(header.copy(), cmds=cmds)
    header['ID'] = 2
    second = Detector(header.copy(), cmds=cmds)
    effect.apply_to(first)
    effect.apply_to(second)
    assert abs(np.corrcoef(first.data.ravel(), second.data.ravel())[0, 1]) < .05


def test_selector_detector_properties_keep_exposure_references_live():
    from scopesim.effects import SelectorWheel
    from scopesim.utils import from_currsys
    cmds = UserCommands(properties={'!DET.noise_class': 'BasicReadoutNoise', '!DET.noise_kwargs': {'noise_std': 2., 'ndit': '!OBS.ndit'}, '!OBS.ndit': 1})
    effect = SelectorWheel(selector_key='detector_id', wheel=[{'selector_value': 0, 'effect_class': '!DET.noise_class', 'effect_kwargs': '!DET.noise_kwargs'}], cmds=cmds)
    cmds['!OBS.ndit'] = 7
    assert from_currsys(effect.get_effect(0).meta['ndit'], cmds) == 7
