# tb-07-moffat-kernel-tests

Tests only. No production code changes.

## Why the scope is narrower than expected

The Tier B entry asked to "detach the kernel maths from `_apply_convolution`'s
FOV-header rewriting". Checking against the upstream baseline changed the
picture: `MoffatPSF`, `AOEnhanceablePSF`, `moffat_cell_integrals`,
`fwhm2gamma` and `_legendre_rule` **do not exist upstream at all** —
`scopesim/effects/psfs/analytical.py` ends at `_sigma2gauss` there — and
`psfs/psf_base.py` is byte-identical to upstream.

So there is nothing to detach for upstreaming purposes and no flag to add. The
class is already opt-in by virtue of being named in a YAML. What was actually
missing is that the kernel mathematics had **no direct test anywhere**. It was
exercised only through `MoffatPSF.get_kernel`, so an error in the closed-form
integral could surface only as a slightly wrong PSF — the hardest kind of
error to notice in this codebase.

`_apply_convolution`'s in-place rewriting of the FOV's `NAXIS`/`CDELT`/`CRPIX`/
`CRVAL` and its nulling of `_volume` and `pixel_area` is left untouched. That
is a layering problem, not a maths problem, and belongs with the `observe()`
dispatch rework.

## What is now tested

`moffat_cell_integrals` evaluates the x-marginal of the 2D Moffat in closed
form, integrates the conditional y-distribution exactly as a Student-t with
`2*alpha - 1` degrees of freedom via `scipy.special.stdtr`, and finishes the
remaining x-integral with Gauss-Legendre quadrature. The wavelength averaging
happens *inside* this function (`integral/len(gammas)`), not in `get_kernel`.

- **Against numerical quadrature.** Single cells at three positions compared
  with `scipy.integrate.dblquad` on the analytic profile. They agree to
  machine precision (relative error 0 and 3.3e-16).
- **Normalisation.** Summing the cell integrals over a wide grid gives 1 to
  1e-5.
- **A profile far narrower than the cell** still integrates to unity. This is
  the case the analytic integral exists to handle and the one that point
  sampling of the kernel gets wrong.
- **Wavelength averaging** equals the mean of the single-width results exactly.
- **Symmetry** about both axes.

`fwhm2gamma` implements FWHM = 2*gamma*sqrt(2^(1/alpha) - 1):

- the profile is half its peak at half the FWHM, across four alphas and three
  FWHMs
- gamma scales linearly with FWHM
- the returned quantity keeps the input angular unit (checked with marcsec,
  since the callers mix arcsec and other units)

## Recorded, not fixed

`MoffatPSF._apply_convolution` widens the base signature —
`PSF._apply_convolution(self, obj)` versus
`(self, obj, spatial_shifts=())` — and the `super()` fallback drops
`spatial_shifts` silently. It is currently unreachable with non-empty shifts,
because `defer_render` in `observe()` requires a `FieldOfView3D` and the
fallback only runs when the object is not one. Worth an assertion rather than a
fix, and worth remembering if that dispatch is ever relaxed.

## Verification

`scopesim/tests`: 4 failed, 1397 passed. Same four pre-existing failures as on
`main`; 21 new tests pass.
