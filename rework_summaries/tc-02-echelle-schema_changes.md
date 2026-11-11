# tc-02-echelle-schema

Removes the duplicated echelle parameter parser, replaces attribute injection
with a real API, and corrects a docstring that documented a schema the code
cannot load. **No number changes**: byte-identical on a 216-point ZShooter
fingerprint, despite being the branch most likely to move something.

## 1. One parser instead of two

`EchelleSpectralTraceList._generate_trace_hdulist` and
`EchelleSpectralEfficiency._generate_efficiency_curve_func` read the same file
and had the same ~16 statements, 14 of them character-for-character identical
including the whole `spectrograph_factory` argument list. They had already
drifted apart in two ways.

Both now call `echelle.spectrograph_from_trace_row(row, meta, colnames)`, which
returns the model plus the parsed intermediates the trace list still needs.

**Divergence 1, `detector_pad`.** The efficiency subtracted `2*detector_pad`
from both pixel counts; the trace list did not. Unified on the trace list's
reading, because that is the one that is load-bearing — the efficiency only
uses `grating.blaze()` and `xdisp_efficiency()`, neither of which depends on a
pixel count.

Confirmed inert before changing it: every production row has
`xbeta_center != 0`, so the Littrow branch that *would* consume the pixel
counts is never reached. The shipped test
(`test_SpectralEfficiency.py`) does use `xbeta_center 0` and so does exercise
that branch — and still passes unchanged, which is the direct evidence that
the efficiency values do not depend on the pad.

**Divergence 2, required columns.** The efficiency indexed
`row['detector_pad']` unconditionally while the trace list guarded with an
`in colnames` check, so a parameter file without that column loaded in one
class and raised `KeyError` in the other. Both now take the guarded path.

A third, quieter difference also disappears: only the trace list recomputed
`pix_per_res_elem` from `nominal_slit_width` and the collimator ratio. Both do
now. That value reaches only the `FWHMPIX` header card and an advisory warning,
so nothing moves.

## 2. Detector placement is an API, not attribute injection

A `SpectrographSetup` was unusable until `EchelleSpectralTraceList` reached
into it and set four private attributes from outside — `_detector_cos`,
`_detector_sin`, `_slit_offset_pix` and `_detector_center` — with the fourth
assigned ~40 lines after the first three, while a method needing only the first
three was already being called in between.

Now:

- `set_slit_rotation(slit_offset_pix, detector_angle_deg)`
- `set_detector_center(center)`
- a read-only `slit_offset_pix` property
- `_require_placement()`, so an unplaced model raises a message naming the
  method to call instead of failing somewhere downstream

The two-step shape is kept deliberately: the centre genuinely cannot be
computed until the model exists, so collapsing this into the constructor would
have been a lie about the dependency order.

## 3. Docstring corrected

The class docstring documented a `slitwidth` column where the code requires
`slitlength`, and omitted `xbeta_center`, which is mandatory — so the worked
example table it offered **could not be loaded**, raising `KeyError` twice. It
also listed `*_unit` keys the code never reads (`echelle_blaze_unit`,
`detector_pad_unit`, `n_disp_unit`, `n_xdisp_unit`; the two angle columns are
always read as degrees) and pointed at a `ZShooter_v1` file carrying the stale
v1 schema.

Replaced with the actual required and optional column lists and a minimal
example that loads.

## 4. Smaller fixes

- **`DETPAD`** keeps its value and its inertness — `test_SpectralTraceList.py`
  explicitly asserts that pad 10 and pad 30 give identical wavelength ranges,
  so that is deliberate and tested. Only the comment changed, from "not
  applied" to saying why it is retained. Kept short: FITS truncates long card
  comments, which the test suite catches.
- **`_warn_if_echelle_design_res_inconsistent`** no longer wraps its unit
  coercions in `except (TypeError, ValueError, u.UnitConversionError): return`.
  These values come from the configured model, so a coercion failure is a
  broken parameter file, not a reason to skip the consistency check in silence.

## Note for reviewers: `validation/zshooter_xdisp/geometry.py`

That module loads `echelle.py` from **pinned git revisions** rather than the
working tree, so it must keep using `_slit_offset_pix`. It is deliberately not
migrated to the new property; doing so breaks it, which is how this was found.

## Verification

- `scopesim/tests`: 4 failed, 1376 passed — the same four pre-existing failures
  as `main`.
- `zs-scopesim/tests/test_xdisp_geometry.py`: 2 failed, 11 passed — the same
  two pre-existing failures as `main` (a float `samples` passed to
  `np.linspace`), plus one new test.
- ZShooter `SPEC` fingerprint (216 quantities): **byte-identical to `main`**.
