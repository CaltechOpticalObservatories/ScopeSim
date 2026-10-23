"""Branch throughput must be applied before integrating an imaging spectrum."""

from types import SimpleNamespace

import numpy as np
import pytest

from scopesim.commands import UserCommands
from scopesim.effects import DichroicTree, SelectorWheel, TERCurve
from scopesim.optics.image_plane import ImagePlane
from scopesim.optics.optical_train import OpticalTrain, extract_source
from scopesim.source.source import Source
from scopesim.tests.mocks.py_objects.fov_objects import _centre_fov
from scopesim.tests.mocks.py_objects.source_objects import _table_source


@pytest.fixture
def tree(tmp_path):
    (tmp_path / 'tree.dat').write_text('image_plane_id first second\n0 R X\n1 T R\n2 T T\n')
    for name, transmission in [('first', 0.75), ('second', 0.6)]:
        curve = '# wavelength_unit: um\nwavelength transmission reflection emissivity\n'
        curve += f'0.5 {transmission} {1 - transmission} 0\n2.5 {transmission} {1 - transmission} 0\n'
        (tmp_path / f'{name}.dat').write_text(curve)
    return DichroicTree(name='tree', tree_filename=str(tmp_path / 'tree.dat'), dichroic_names=['first', 'second'], filename_format=str(tmp_path / '{}.dat'))


@pytest.fixture
def selector():
    wheel = []
    for arm, transmission in enumerate([0.25, 0.75]):
        curve = {'wavelength': [0.5, 2.5], 'transmission': [transmission, transmission], 'emissivity': [0, 0]}
        wheel.append({'selector_value': arm, 'effect_class': 'QuantumEfficiencyCurve', 'effect_kwargs': {'array_dict': curve, 'wavelength_unit': 'um'}})
    return SelectorWheel(selector_key='image_plane_id', wheel=wheel, allowed_missing_selector_values=[2])


@pytest.fixture
def source():
    source = _table_source()
    background = TERCurve(array_dict={'wavelength': [0.5, 2.5], 'transmission': [1, 1], 'emission': [1e-8, 1e-8]}, wavelength_unit='um', emission_unit='photlam')
    source.append(background.background_source[0])
    return source


@pytest.mark.parametrize('effect_name, transmissions', [('tree', [0.25, 0.3, 0.45]), ('selector', [0.25, 0.75, 1.0])])
def test_branch_attenuates_coloured_sources_and_background_independently(request, effect_name, transmissions, source):
    effect = request.getfixturevalue(effect_name)
    reference = extract_source(_centre_fov(n=10), source).make_hdu().data
    for arm, transmission in enumerate(transmissions):
        fov = _centre_fov(n=10)
        fov.meta['image_plane_id'] = arm
        extract_source(fov, source, [effect])
        np.testing.assert_allclose(fov.make_hdu().data, transmission * reference)
    # Rendering each branch must leave the shared source, including its sky, intact.
    np.testing.assert_array_equal(extract_source(_centre_fov(n=10), source).make_hdu().data, reference)


@pytest.mark.parametrize('effect_name', ['tree', 'selector'])
def test_source_without_fov_context_is_unchanged(request, effect_name, source):
    effect = request.getfixturevalue(effect_name)
    reference = extract_source(_centre_fov(n=10), source).make_hdu().data
    effect.apply_to(source)
    np.testing.assert_array_equal(extract_source(_centre_fov(n=10), source).make_hdu().data, reference)


@pytest.mark.parametrize('effect_name', ['tree', 'selector'])
def test_missing_branch_id_fails(request, effect_name, source):
    effect = request.getfixturevalue(effect_name)
    fov = _centre_fov(n=10)
    fov.meta.clear()
    with pytest.raises(KeyError, match='image_plane_id'):
        effect.apply_to(source, fov=fov)


def test_observe_keeps_downstream_emission_after_branch_split(tree, source):
    common = TERCurve(array_dict={'wavelength': [0.5, 2.5], 'transmission': [0.8, 0.8], 'emissivity': [0, 0]}, wavelength_unit='um')
    downstream = TERCurve(array_dict={'wavelength': [0.5, 2.5], 'transmission': [0.5, 0.5], 'emission': [2e-8, 2e-8]}, wavelength_unit='um', emission_unit='photlam')
    train = OpticalTrain()
    train.cmds = UserCommands()
    fovs = [_centre_fov(n=10) for arm in range(3)]
    for arm, fov in enumerate(fovs):
        fov.meta['image_plane_id'] = arm
    train.fov_manager = SimpleNamespace(fovs=fovs, is_spectroscope=False)
    train.optics_manager = SimpleNamespace(source_effects=[common, tree, downstream], fov_effects=[], image_plane_effects=[])
    train.image_planes = [ImagePlane(fov.header) for fov in fovs]

    prepared = train.prepare_source(source.make_copy())
    reference = extract_source(_centre_fov(n=10), prepared).make_hdu().data
    emission = extract_source(_centre_fov(n=10), downstream.background_source[0]).make_hdu().data
    train.observe(source, update=False)
    for arm, transmission in enumerate([0.25, 0.3, 0.45]):
        expected = reference * 0.8 * transmission * 0.5 + emission
        np.testing.assert_allclose(train._last_fovs[arm].hdu.data, expected)
        assert train._last_fovs[arm].image_plane_id == arm


@pytest.mark.parametrize('spectrum_id, expected', [(0, 0.5), (1, 0.5 + 1 / 24), (2, 0.5 - 1 / 24)])
def test_wavelength_dependent_branch_preserves_source_colour(spectrum_id, expected):
    spectrum = _table_source().fields[0].spectra[spectrum_id]
    source = Source(x=[0], y=[0], ref=[0], weight=[1], spectra=[spectrum])
    curve = {'wavelength': [0.5, 2.5], 'transmission': [0, 1], 'emissivity': [0, 0]}
    selector = SelectorWheel(selector_key='image_plane_id', wheel=[{'selector_value': 0, 'effect_class': 'QuantumEfficiencyCurve', 'effect_kwargs': {'array_dict': curve, 'wavelength_unit': 'um'}}])
    reference = extract_source(_centre_fov(n=10), source).make_hdu().data.sum()
    fov = _centre_fov(n=10)
    fov.meta['image_plane_id'] = 0
    measured = extract_source(fov, source, [selector]).make_hdu().data.sum() / reference
    # Linear spectra and throughput give these analytic integrals over 1--2 um.
    # The input spectra are sampled every 0.02 um.
    np.testing.assert_allclose(measured, expected, rtol=0, atol=1e-4)


@pytest.mark.parametrize('z_order, expected', [([117, 217, 517], (117, 217, 517)), (517, (517,))])
def test_dichroic_source_stage_is_opt_in(tree, z_order, expected):
    assert tree.z_order == (117, 217, 617)
    configured = DichroicTree(z_order=z_order, **{key: tree.meta[key] for key in tree.required_keys})
    assert configured.z_order == expected
