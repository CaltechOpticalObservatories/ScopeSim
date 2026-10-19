import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from synphot import SourceSpectrum, ConstFlux1D
from synphot.units import PHOTLAM

from scopesim import Source
from scopesim.effects.atmo_dispersion import ShiftFoV3D, _cube_wavelengths
from scopesim.effects.psfs.analytical import MoffatPSF
from scopesim.source.source_fields import BackgroundSourceField
from scopesim.optics.image_plane_utils import header_from_list_of_xy
from scopesim.tests.tests_optics.test_slit_spatial_integration import physical_fov


class Displacement(ShiftFoV3D):
    def __init__(self, dx, dy, **kwargs):
        super().__init__(**kwargs)
        self.dx, self.dy = dx*u.arcsec, dy*u.arcsec

    def get_shifts(self, obj):
        wave = _cube_wavelengths(obj)
        return np.broadcast_to(self.dx, wave.shape, subok=True), np.broadcast_to(self.dy, wave.shape, subok=True)


def render(fov, source, psf, shifts):
    support = psf.padded_header(fov, point_sources=not (source.image_fields or source.cube_fields))
    for effect in shifts:
        support = effect._padded_header(support, effect.get_shifts(fov))
    fov.extract_from(source, spatial_header=support)
    psf._apply_convolution(fov, spatial_shifts=shifts)
    return fov.hdu.data*fov.pixel_area.value/1e4


@pytest.mark.parametrize('width', [.33, .7, .9])
@pytest.mark.parametrize('position,steps', [((0., 0.), (0, 1)), ((.037, .4), (0, -2)), ((5.6, .1), (-2, 1)), ((7., 0.), (-15, 0))])
@pytest.mark.parametrize('sign', [-1, 1])
def test_shifted_point_uses_incoming_psf_before_slit_and_ends(width, position, steps, sign):
    fov = physical_fov(width)
    fov.header['CDELT2'] *= sign
    original = fov.header.copy()
    cell = np.array([fov.header['CDELT1'], fov.header['CDELT2']])*3600
    displacement = np.array(steps)*cell
    shift = Displacement(*displacement, method='roll')
    psf = MoffatPSF(alpha=4.765, fwhm=.7)
    source = Source(x=[position[0]], y=[position[1]], ref=[0], weight=[1.], spectra=[SourceSpectrum(ConstFlux1D, amplitude=1*PHOTLAM)])
    actual = render(fov, source, psf, [shift])
    expected = psf.point_source_image(fov, *(position + displacement))
    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), rtol=2e-6, atol=1e-12)
    for key in ('NAXIS1', 'NAXIS2', 'CRPIX1', 'CRPIX2', 'CDELT1', 'CDELT2', 'CRPIX1D', 'CRPIX2D'):
        assert fov.header[key] == original[key]
        assert fov.hdu.header[key] == original[key]


@pytest.mark.parametrize('method', ['roll', 'map_coords'])
def test_uniform_sky_is_not_lost_at_shifted_slit_edges(method):
    fov = physical_fov(.7)
    source = Source()
    source.fields.append(BackgroundSourceField(field=None, header=fits.Header(), spectra={0: SourceSpectrum(ConstFlux1D, amplitude=1*PHOTLAM)}))
    psf = MoffatPSF(alpha=4.765, fwhm=.7)
    shift = Displacement(.23, -.17, method=method)
    actual = render(fov, source, psf, [shift])
    np.testing.assert_allclose(actual, fov.pixel_area.value, rtol=1e-12)


def test_wavelength_dependent_shift_keeps_spectral_planes_independent():
    fov = physical_fov(.7)
    count = len(fov.cube_waveset)
    cell_y = fov.header['CDELT2']*3600
    displacement = np.resize([-1., 0., 1.], count)*cell_y
    shift = Displacement(0., displacement, method='roll')
    psf = MoffatPSF(alpha=4.765, fwhm=.7)
    source = Source(x=[0], y=[.2], ref=[0], weight=[1.], spectra=[SourceSpectrum(ConstFlux1D, amplitude=1*PHOTLAM)])
    actual = render(fov, source, psf, [shift])
    expected = np.stack([psf.point_source_image(fov, 0., .2 + dy) for dy in displacement])
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_two_displacements_keep_light_that_reenters_slit():
    fov = physical_fov(.7)
    step = fov.header['CDELT2']*3600
    shifts = [Displacement(0., 4*step, method='roll'), Displacement(0., -4*step, method='roll')]
    psf = MoffatPSF(alpha=4.765, fwhm=.7)
    source = Source(x=[0], y=[0], ref=[0], weight=[1.], spectra=[SourceSpectrum(ConstFlux1D, amplitude=1*PHOTLAM)])
    actual = render(fov, source, psf, shifts)
    expected = psf.point_source_image(fov, 0., 0.)
    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), rtol=1e-6)


def test_shifted_resolved_image_and_cube_agree():
    header = header_from_list_of_xy([-1/3600, 1/3600], [-1/3600, 1/3600], .05/3600)
    yy, xx = np.indices((header['NAXIS2'], header['NAXIS1']))
    density = np.exp(-((xx-20)**2 + (yy-28)**2)/8)
    image_source = Source(image_hdu=fits.ImageHDU(data=density*.05**2, header=header), spectra=[SourceSpectrum(ConstFlux1D, amplitude=1*PHOTLAM)])
    cube_header = header.copy()
    cube_header.update(CRPIX3=1., CRVAL3=.49, CDELT3=.01, CUNIT3='um', CTYPE3='WAVE', BUNIT='photlam arcsec-2')
    cube_source = Source(cube=fits.ImageHDU(data=np.broadcast_to(density, (13, *density.shape)).copy(), header=cube_header))
    psf = MoffatPSF(alpha=4.765, fwhm=.7)
    shift = Displacement(.15, -.2, method='roll')
    images = [render(physical_fov(.7), source, psf, [shift]) for source in (image_source, cube_source)]
    np.testing.assert_allclose(images[0], images[1], rtol=2e-6, atol=1e-10)
    fov = physical_fov(.7)
    cell = np.array([fov.header['CDELT1'], fov.header['CDELT2']])*3600
    displacement = np.rint(np.round(np.array([.15, -.2])/cell, 1))*cell
    moved = image_source.make_copy()
    moved.shift(*displacement)
    expected = render(fov, moved, psf, [])
    np.testing.assert_allclose(images[0], expected, rtol=2e-6, atol=1e-10)


@pytest.mark.parametrize('method', ['roll', 'map_coords'])
def test_zero_shift_reproduces_unshifted_render(method):
    psf = MoffatPSF(alpha=4.765, fwhm=.7)
    source = Source(x=[.1], y=[.2], ref=[0], weight=[1.], spectra=[SourceSpectrum(ConstFlux1D, amplitude=1*PHOTLAM)])
    unshifted = render(physical_fov(.7), source, psf, [])
    shift = Displacement(0., 0., method=method)
    fov = physical_fov(.7)
    assert shift._padded_header(fov.header, shift.get_shifts(fov)) == fov.header
    shifted = render(fov, source, psf, [shift])
    np.testing.assert_allclose(unshifted, shifted, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize('position', [(0., 0.), (.1, .4), (5.6, 0.)])
def test_fractional_shift_agrees_with_pre_slit_psf_flux_and_centroid(position):
    fov = physical_fov(.7)
    cell = np.array([fov.header['CDELT1'], fov.header['CDELT2']])*3600
    shift = Displacement(*(np.array([.3, -.4])*cell), method='map_coords')
    psf = MoffatPSF(alpha=4.765, fwhm=.7)
    source = Source(x=[position[0]], y=[position[1]], ref=[0], weight=[1.], spectra=[SourceSpectrum(ConstFlux1D, amplitude=1*PHOTLAM)])
    actual = render(fov, source, psf, [shift])[0]
    expected = psf.point_source_image(fov, *(position + np.array([.3, -.4])*cell))
    assert actual.sum() == pytest.approx(expected.sum(), rel=1e-3)
    yy, xx = np.indices(actual.shape)
    for coordinate in (xx, yy):
        measured = np.sum(actual*coordinate)/actual.sum()
        reference = np.sum(expected*coordinate)/expected.sum()
        assert measured == pytest.approx(reference, abs=.01)


@pytest.mark.parametrize('intervening_throughput', [False, True])
def test_observe_applies_shift_before_slit_through_wavelength_throughput(intervening_throughput):
    from types import SimpleNamespace
    from scopesim import OpticalTrain
    from scopesim.effects import TERCurve

    psf = MoffatPSF(alpha=4.765, fwhm=.7)
    images = []
    for move_source in (False, True):
        fov = physical_fov(.7)
        displacement = -2*fov.header['CDELT2']*3600
        source_y = .4 + displacement if move_source else .4
        source = Source(x=[0], y=[source_y], ref=[0], weight=[1.], spectra=[SourceSpectrum(ConstFlux1D, amplitude=1*PHOTLAM)])
        effects = [psf]
        if intervening_throughput:
            effects.append(TERCurve(array_dict={'wavelength': [.4, .7], 'transmission': [.2, .8]}, wavelength_unit='um'))
        if not move_source:
            effects.append(Displacement(0., displacement, method='roll'))
        train = OpticalTrain()
        train.cmds = fov.cmds
        train.fov_manager = SimpleNamespace(fovs=[fov])
        train.optics_manager = SimpleNamespace(source_effects=[], fov_effects=effects, image_plane_effects=[])
        train.image_planes = [SimpleNamespace(add=lambda hdu, **kwargs: images.append(hdu.data.copy()))]
        train.observe(source, update=False)
        train.observe(source, update=False)
    for image in images[1:]:
        np.testing.assert_allclose(images[0], image, rtol=2e-6, atol=1e-12)
