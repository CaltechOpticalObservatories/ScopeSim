# tb-05-integrate-spectral-bins

Puts the fork's spectral-sampling rework behind
`!SIM.spectral.integrate_spectral_bins`, defaulting to upstream behaviour.

This is the largest and riskiest of the Tier B branches, because the fork
**deleted** the upstream code paths rather than branching around them.
Restoring them is most of the diff.

## Why

The fork replaced upstream's point sampling of spectra (`spec(fov_waveset)`,
`interp1d` onto the FOV grid) with flux-conserving bin integration. The physics
is better — a line narrower than the grid keeps its photons instead of being
sampled at whatever height it happens to have at a bin centre — but it moves
the flux of every instrument in the IRDB. Upstream cannot take that as a
default. `trace_flux_jacobian : False` in `defaults.yaml` is this fork's own
precedent for how such a change should ship.

## One flag, not several

The parts are not independently selectable. Every mixed setting is physically
wrong rather than merely different:

- **`prepare_source` resampling x cube rebinning.** With upstream resampling on
  *and* rebinning on, the cube is first point-sampled onto the global grid,
  destroying narrow lines, and only then integrated — strictly worse than
  either pure mode.
- **Edge-plane `f0`/`f1` scaling x rebinning.** Pre-scaling the first and last
  planes turns them into partial-bin *counts*, but `rebin_spectral_density`
  consumes a *density* and integrates each plane over its own width, so the
  partial bin is counted twice. The opposite mix leaves the edge planes at full
  height and over-counts band-edge flux.
- **`cube_waveset` padding x `extract_from` padding.** A padded output grid over
  an unpadded extraction fills the pad with zeros — a fake flux cliff at both
  band edges. Both are driven by the same `spectral_padding` value.
- **`average_spectrum` x cube rebinning** are mechanically separable, but
  `test_photon_integrals_agree_between_representations` asserts that a table, an
  image and a cube deliver the same photon rate. Splitting the flag would make
  that invariant configuration-dependent, which is exactly what a config switch
  should not be able to express.

## Sites

Declared once in `defaults.yaml`, read via a lazy property on the **base**
`FieldOfView` (not `FieldOfView3D` — `_extract_volume_from_cube` lives on the
base), following the `spline_order` idiom: bang-string read directly, no entry
in the `__init__` meta dict and none in `FOVManager.meta`. `sub_pixel`'s meta
entry exists because it is overridable per FOV; this is a global physics
switch.

**`optical_train.py:608` first — it removes work elsewhere.** `spectral_padding`
is written in exactly one place, and `slit_spectral_width` only as a side
effect of the same call. Gating that single assignment leaves
`fov.meta['spectral_padding']` at `0*u.um`, which reduces five further sites to
upstream *by construction* with no edit at all, since `x + (-0.0) == x`
exactly.

Then:

- **`cube_waveset`** — early return giving upstream's grid.
- **`_extract_volume_from_cube`** — one named `density_input` intermediate,
  read twice (mask margin, and the `f0`/`f1` edge scaling).
- **`_make_cubefields`** — `if/else` on the one expression that differs,
  restoring `interp1d`.
- **The four `average_spectrum` call sites** — via a small
  `_sampled_spectrum` helper. Note there are **four**, not three: the fourth is
  in `make_hdu`'s deferred-PSF table branch, not in a `_make_*` method, and is
  the one that gets missed.
- **`prepare_source`** — restores the per-spectrum resampling loop (deleted
  outright by the fork, no counterpart) and the layer-by-layer cube
  interpolation with its `CTYPE3`/`CRPIX3`/`CRVAL3`/`CDELT3`/`CUNIT3` rewrite
  and the "wavesets disjoint" warning. Re-adds three imports: `interp1d`,
  `SourceSpectrum, Empirical1D`, `SpectrumSourceField`. A shared
  `_global_fov_waveset` helper builds the grid both places need.

## The off path must be upstream's literal `arange`

`start + np.arange(n)*step` is **not** `np.arange(start, stop, step)[:n]`.
Measured:

| range | step | len(arange) | n_wave | max difference |
|---|---:|---:|---:|---:|
| 1.9 to 2.1 | 0.01 | 21 | 20 | 4.4e-16 um |
| 0.3 to 2.5 | 1e-5 | 220000 | 220000 | 2.2e-12 um |

The first case shows the off-by-one guard dropping a whole bin, so getting this
wrong changes `NAXIS3`, not just the last ulp. Upstream's comment about "some
floating point madness" is preserved verbatim, and
`test_cube_waveset_matches_upstream_arange_grid` asserts with
`assert_array_equal`, not `allclose`, specifically to catch a future
"simplification".

The early return also happens *before* the padding and step-refinement lines,
so stale meta cannot widen the grid even if something sets it.

## Left unconditional on purpose

These are correct on both paths and should not be gated:

- The `.copy()` in `_extract_volume_from_cube`. Upstream mutated the caller's
  source array in place via `data[0] *= f0`. That is a bug on any path;
  `test_make_spectrum_from_cube` already asserts the source is unchanged. This
  is the one place where "restore upstream" would mean "restore a bug".
- `abs(wdel)` in the mask. Bit-identical for an ascending cube; upstream's raw
  `0.5 * wdel` is simply broken for a descending one.
- float64 for cube data. A 1e-8 contrast cannot survive float32, which
  `test_prepare_cube_retains_faint_contrast_in_double_precision` asserts. This
  does change METIS/MICADO cube results at roughly the 1e-7 relative level and
  should be called out as a deliberate exception in any upstream PR.
- The `CDELT3`/`CRVAL3` header written by `_make_cubefields`. In both branches
  the data lies on `fov_waveset`, so the WCS is correct either way; upstream
  simply left a stale third axis on the yielded HDU.

## Tests

Opt-ins, three edits covering everything:

- `physical_fov()` in `test_slit_spatial_integration.py` gains the key. Because
  `test_adc_slit_illumination.py` imports that helper, one line covers both
  files.
- `test_FieldOfView.py`'s cross-representation test gains the key in its
  `UserCommands`, plus the new `integrate_spectral_bins` fixture — it calls
  `OpticalTrain().prepare_source(...)` *before* building those commands, so it
  straddles both modes without the fixture.
- A function-scoped fixture in `conftest.py` sets the key in
  `rc.__currsys__` and restores it. Needed because `prepare_source` takes no
  `cmds` and `OpticalTrain(cmds)` would require a real instrument package.

New tests: the bit-exact grid above; that padding is ignored when the flag is
off (guards the early-return ordering) and applied when on; that a narrow line
*is lost* under point sampling, so a regression in the restored path is caught
rather than silently "improved"; and that `prepare_source` resamples onto the
global grid by default.

**Deleted:** `test_reuses_cached_spectrum_waveset`, which referenced
`fov_mod._SPECTRUM_WAVESET_AA_CACHE` — a module attribute that exists nowhere
in the tree, so the test *errored* rather than failed. The cache is
deliberately gone: a `SourceSpectrum` can re-sample itself in place through
`z`, which `test_mutating_spectrum_redshift_updates_native_sampling` asserts.

## What this does not fix

The two `test_basic_instrument` flux regressions are **unchanged** by this flag
(still 11 vs 7 and 78 vs 75). They come from the unflagged *spatial* projection
rework — `project_spatial_density` replacing `add_imagehdu_to_imagehdu` in
`_make_imagefields`, and the PSF-deferred rendering path. Full parity with
upstream needs that gated too, which is separate work.

Three further unflagged divergences sit inside functions this branch touches
and will be visible to any reviewer diffing against upstream:
`int(np.floor(xpix + .5))` vs `int(xpix)`, the `project_spatial_density`
rewiring, and `FieldOfView1D._make_cubefields`'s surface-brightness unit fix.
Each changes default results and each needs its own story before this goes
upstream.

## Verification

`scopesim/tests`: 3 failed, 1382 passed — one *below* the `main` baseline of 4,
because the stale cache test is gone.

End to end, with the companion irdb branch applied, a ZShooter SPEC observation
is **bit-identical** to `main`: all six image planes match on SHA-256. Without
the irdb opt-in the same run shifts +3.1% in total flux and +24% in peak, so
the two must merge together.
