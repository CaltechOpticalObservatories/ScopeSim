# tc-01-echelle-hygiene

Dead code and small defects in `scopesim/optics/echelle.py`. **No number
changes**: verified byte-identical on a 216-point ZShooter fingerprint.

`echelle.py` is entirely new in this fork (absent at the upstream baseline
`2d245168`), so there is no upstream behaviour to preserve here — only defects
to remove. 822 lines down to 716.

## Removed, after confirming zero callers in both repos

- **`average_res`** — called `self.detector.n_pixels`, which does not exist on
  the three-field detector dataclass, so **every call raised `AttributeError`**.
  It also transposed an `(nord, 1)` array against an `(nord,)` numerator, so it
  would have broadcast wrongly even with the attribute fixed. No caller in
  ScopeSim, zs-scopesim, tests or notebooks.
- **`implied_design_res`** — no callers.
- **`plot_echellogram`** — no callers; imported matplotlib inside the function
  and called `plt.show()` from library code; raised `NotImplementedError` for
  `cross_disperse=True`, leaving only the non-cross-dispersed branch reachable.
- **Three commented-out blocks**: `pixel_rescale`/`pixel_samples_frac`/
  `pixel_max_npoints` (MKID leftovers referencing `dl_mkid_pixel`, `sampling`
  and `dl_mkid_max`, none of which exist); a `GRATING_CATALOG` loading
  `newport_masters.txt`, which is not in the repo; and an example block written
  positionally against a signature it no longer matches, so it could not run.

`nominal_pixels_per_res_elem` was the only state those two properties consumed.
It is **kept**: `pixels_per_res_elem` remains a documented factory parameter
that callers pass positionally, and silently discarding a supplied value would
be worse than storing it. It is now commented as configuration that no geometry
depends on.

Deliberately **not** touched: `edge_wave` and `_spectrographs` are consumed by
`zs_scopesim_tools/validation.py`; `order_mask`'s only zs-scopesim references
are a `FakeSpectrograph` test double; everything in the pixel/beta mapping and
`__init__` focal-length block is load-bearing geometry.

## Defects fixed

- **`class Detector` renamed to `EchelleDetector`.** It shadowed
  `scopesim.detector.Detector`. Safe: nothing imports the name — both consumers
  do `from ..optics import echelle`, and `scopesim/optics/__init__.py` never
  imports the module. The stale `:param MKIDDetector detector:` docstring is
  corrected too.
- **`blaze` now accepts a scalar angle.** `k[k > 1] = 1` required a writable
  array, so a scalar `beta` produced a 0-d result and raised. Now
  `np.minimum(..., 1.)`, which also reads as the cap it is.
- **`__str__` works for VPH gratings.** `delta` is `None` for VPH (there is no
  blaze facet), so `np.rad2deg(self.delta)` raised. Now prints `delta=n/a`.
- **The `orders` cache no longer depends on `assert`.** It used
  `try: assert self._orders[0] == (m0, m_max) / except (TypeError, AssertionError)`,
  which `python -O` strips — leaving the cached order array permanently stale
  whenever the order range changed. Now an explicit `if`.
- **`pix_size` docstring corrected** from "If float, assumed to be in um" to
  mm. The code does `pix_size * u.mm` and every float caller passes `.015`
  meaning 15 um; honouring the docstring would have made every pixel scale
  wrong by 1000x. The code was right, the docstring was not.
- **`y_pixel_to_wavelength` dropped its unused `m` parameter.** The body
  hardcodes order 1, because the cross-disperser works in first order. Its one
  caller (`test_echelle.py:61`) updated.

## Tests

New `TestEchelleHygiene`: scalar and array `blaze` agree and are capped at 1;
`str()` of a VPH grating; the order cache refreshes when `m_max` changes
(the `-O` bug); `EchelleDetector` is distinct from the core `Detector` and the
old name is gone; and the three deleted diagnostics stay deleted.

## Verification

- `scopesim/tests`: 4 failed, 1382 passed — same four pre-existing failures as
  `main`, six new tests passing.
- ZShooter `SPEC` fingerprint (216 quantities: trace catalog, detector
  geometry, background budget at both slit widths, trace resolution,
  transmission and emissivity, QE accounting, slit-loss, science-truth
  crosscheck, six image planes, six readouts): **byte-identical to `main`**.
