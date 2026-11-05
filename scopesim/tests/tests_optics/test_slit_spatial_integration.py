from types import SimpleNamespace
import numpy as np
import pytest
from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits
from synphot import SourceSpectrum, ConstFlux1D
from synphot.units import PHOTLAM
from scipy.special import stdtr
from scopesim import Source, UserCommands
from scopesim.effects import TERCurve, DetectorWindow
from scopesim.effects.psfs.analytical import MoffatPSF
from scopesim.optics.fov import FieldOfView3D
from scopesim.optics.fov_manager import FOVManager
from scopesim.optics.image_plane_utils import header_from_list_of_xy, det_wcs_from_sky_wcs, project_spatial_density


def slit_fov(width, step=.1570545):
    nx, ny = int(np.ceil(11/step)), int(np.ceil(width/step))
    header = header_from_list_of_xy([-5.5/3600, 5.5/3600], [-width/7200, width/7200], step/3600)
    header.update(NAXIS1=nx, NAXIS2=ny, CRPIX1=(nx+1)/2, CRPIX2=(ny+1)/2, CRVAL1=0., CRVAL2=0., CDELT1=11/nx/3600, CDELT2=width/ny/3600)
    return SimpleNamespace(header=header, waveset=np.linspace(.5, .6, 16)*u.um)


@pytest.mark.parametrize('width', [.30, .33, .70])
@pytest.mark.parametrize('fwhm', [.10, .14, .7])
@pytest.mark.parametrize('offset', [0, .13, .4])
def test_integrated_point_psf_matches_analytic_slit_loss(width, fwhm, offset):
    alpha = 4.765
    psf = MoffatPSF(alpha=alpha, fwhm=fwhm, rounded_edges=False)
    image = psf.point_source_image(slit_fov(width), .037, offset)
    gamma = fwhm/(2*np.sqrt(2**(1/alpha) - 1))
    scale = np.sqrt(2*alpha - 2)/gamma
    expected = stdtr(2*alpha - 2, (width/2 - offset)*scale) - stdtr(2*alpha - 2, (-width/2 - offset)*scale)
    # The analytic comparison has an infinite slit length; the model has 11 arcsec.
    assert image.sum() == pytest.approx(expected, abs=2e-5)


def test_off_slit_source_remains_in_psf_extraction_support():
    fov = slit_fov(.33)
    psf = MoffatPSF(alpha=4.765, fwhm=.7)
    padded = psf.padded_header(fov)
    x, y = WCS(padded).all_world2pix(0, .4/3600, 0)
    assert -.5 < x < padded['NAXIS1'] - .5
    assert -.5 < y < padded['NAXIS2'] - .5
    assert psf.point_source_image(fov, 0, .4).sum() > .1


def test_kernel_cache_responds_to_seeing_and_pixel_scale():
    fov = slit_fov(.33)
    psf = MoffatPSF(alpha=4.765, fwhm=.7, rounded_edges=False)
    first = psf.get_kernel(fov)
    np.testing.assert_array_equal(psf.get_kernel(fov), first)
    psf.meta['fwhm'] = .2
    narrower = psf.get_kernel(fov)
    assert narrower.max() > first.max()
    fov.header['CDELT2'] /= 2
    finer = psf.get_kernel(fov)
    assert finer.max() < narrower.max()


def physical_fov(width=.33):
    header = slit_fov(width).header
    detector_wcs, _ = det_wcs_from_sky_wcs(WCS(header), .1570545, 10.)
    header.update(detector_wcs.to_header())
    cmds = UserCommands(properties={'!SIM.spectral.spectral_bin_width': .01})
    return FieldOfView3D(header, [.5, .6], cmds=cmds, area=1*u.cm**2)


def test_throughput_then_psf_admits_off_slit_point_light():
    fov = physical_fov()
    psf = MoffatPSF(alpha=4.765, fwhm=.7, rounded_edges=False)
    source = Source(x=[.037], y=[.4], ref=[0], weight=[1.], spectra=[SourceSpectrum(ConstFlux1D, amplitude=1*PHOTLAM)])
    fov.extract_from(source, spatial_header=psf.padded_header(fov))
    transmission = TERCurve(array_dict={'wavelength': [.4, .7], 'transmission': [.5, .5]}, wavelength_unit='um')
    transmission.apply_to(fov)
    psf.apply_to(fov)
    expected = psf.point_source_image(fov, .037, .4).sum()*.5
    actual = fov.hdu.data.sum(axis=(1, 2))*fov.pixel_area.value/1e4
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


@pytest.mark.parametrize('sign', [-1, 1])
def test_surface_brightness_uses_signed_wcs_and_pixel_overlaps(sign):
    header = header_from_list_of_xy([-.8/3600, .8/3600], [-.6/3600, .6/3600], .1/3600)
    header['CDELT1'] *= sign
    field = fits.ImageHDU(data=np.full((header['NAXIS2'], header['NAXIS1']), 7.), header=header)
    target = slit_fov(.33).header
    projected = project_spatial_density(field, target)
    # Only the small input's length illuminates this 11-arcsec slit.
    expected_area = header['NAXIS1']*.1*.33
    assert projected.sum()*abs(target['CDELT1']*target['CDELT2'])*3600**2 == pytest.approx(7*expected_area, rel=1e-6)


def test_resolved_image_and_cube_have_same_psf_and_spectral_normalization():
    header = header_from_list_of_xy([-1/3600, 1/3600], [-1/3600, 1/3600], .05/3600)
    yy, xx = np.indices((header['NAXIS2'], header['NAXIS1']))
    density = np.exp(-((xx - 20)**2 + (yy - 26)**2)/8)
    image = fits.ImageHDU(data=density*.05**2, header=header)
    spectrum = SourceSpectrum(ConstFlux1D, amplitude=1*PHOTLAM)
    image_source = Source(image_hdu=image, spectra=[spectrum])
    cube_header = header.copy()
    cube_header.update(CRPIX3=1., CRVAL3=.49, CDELT3=.01, CUNIT3='um', CTYPE3='WAVE', BUNIT='photlam arcsec-2')
    cube = fits.ImageHDU(data=np.broadcast_to(density, (13, *density.shape)).copy(), header=cube_header)
    cube_source = Source(cube=cube)
    psf = MoffatPSF(alpha=4.765, fwhm=.7, rounded_edges=False)
    rendered = []
    for source in (image_source, cube_source):
        fov = physical_fov()
        fov.extract_from(source, spatial_header=psf.padded_header(fov))
        psf.apply_to(fov)
        rendered.append(fov.hdu.data)
    np.testing.assert_allclose(rendered[0], rendered[1], rtol=2e-6, atol=1e-10)


@pytest.mark.parametrize('width', [.30, .33, .70])
def test_fov_manager_tiles_physical_slit_width(width):
    class SlitVolume:
        def apply_to(self, volumes, **kwargs):
            for volume in volumes:
                volume.update(x_min=-5.5, x_max=5.5, y_min=-width/2, y_max=width/2)
                volume['meta'].update(eta_min=-width/2*u.arcsec, eta_max=width/2*u.arcsec, image_plane_id=0)
            return volumes

    detector = DetectorWindow(pixel_size=.015, x=0, y=0, width=100, height=100, units='pixel', image_plane_id=0, pixel_scale=.1570545, plate_scale=10.)
    manager = FOVManager([SlitVolume(), detector], pixel_scale=.1570545, plate_scale=10., preload_fovs=False)
    manager.is_spectroscope = True
    fov = next(manager.generate_fovs_list())
    assert fov.header['NAXIS2']*fov.header['CDELT2']*3600 == pytest.approx(width)
    assert fov.header['NAXIS1']*fov.header['CDELT1']*3600 == pytest.approx(11.)


@pytest.mark.parametrize('edge', ['left', 'right'])
def test_subpixel_deposition_does_not_wrap_or_index_beyond_slit(edge):
    fov = physical_fov()
    fov.meta['sub_pixel'] = True
    xpixel = -.25 if edge == 'left' else fov.header['NAXIS1'] - .75
    ypixel = (fov.header['NAXIS2'] - 1)/2
    x, y = WCS(fov.header, naxis=2).all_pix2world(xpixel, ypixel, 0)
    source = Source(x=[x*3600], y=[y*3600], ref=[0], weight=[1.], spectra=[SourceSpectrum(ConstFlux1D, amplitude=1*PHOTLAM)])
    fov.extract_from(source)
    cube = fov.make_hdu()
    admitted = cube.data.sum(axis=(1, 2))*fov.pixel_area.value/1e4
    np.testing.assert_allclose(admitted, .75, rtol=1e-6)
    opposite = -1 if edge == 'left' else 0
    np.testing.assert_array_equal(cube.data[:, :, opposite], 0.)
