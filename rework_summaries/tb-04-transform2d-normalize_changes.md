# tb-04-transform2d-normalize

Makes `Transform2D` input normalization opt-in instead of unconditional, and
replaces two tests that claimed to cover it but did not.

## Why

`Transform2D.fit` rescaled `xin` and `yin` to `[-1, 1]` before fitting, stored
the rescaling as a `pretransform`, and carried `1/scale` into `gradient()` as a
chain-rule factor. Upstream's version is five lines with none of that.

Enabling it by default is not upstreamable for two reasons:

1. **It changes the public meaning of `.matrix`.** The coefficients are in
   rescaled coordinates, not input coordinates. The existing upstream test
   `test_fit_gives_correct_matrix` had to be given `normalize=False` to keep
   passing — that edit is the clearest evidence the change is observable.
2. It silently alters the fitted transforms of every instrument in the IRDB.

So it is now off by default and selected per trace.

## What the rescaling is actually worth

Measured, because the original justification did not survive checking.

| Input coordinates | degree | unnormalized | normalized |
|---|---:|---:|---:|
| focal-plane mm, centred (-50..50, -25..25) | 4 | 1.2e-14 | 1.6e-14 |
| focal-plane mm, centred | 6 | 2.8e-14 | 1.9e-13 |
| pixels, uncentred (0..2048) | 4 | 5.8e-11 | 8.7e-14 |
| pixels, uncentred | 6 | 5.8e-11 | 8.9e-14 |

(worst absolute residual of the fitted surface)

The rescaling is a **safeguard for poorly scaled inputs**, worth ~650x when
coordinates run far from the origin, and **immaterial — if anything marginally
worse — on the centred focal-plane mm that ScopeSim actually fits.** Both
routes reach machine precision there, which is also why turning it on does not
move an existing trace.

This corrects the rationale I started with, which was that `fit_degree: 6` in
`ZShooter_SPEC.yaml` made conditioning important. At degree 6 in centred mm the
unnormalized fit is the better of the two.

## Where the switch lives, and why it is not a `!SIM` flag

`fit_normalize` is a plain entry in `SpectralTrace._class_params`, defaulting
to `False`, exactly like the `fit_degree` that sits beside it:

```python
"trace_flux_jacobian": "!SIM.spectral.trace_flux_jacobian",
"fit_degree": 4,
"fit_normalize": False,
```

It is resolved in `compute_interpolation_functions` and passed to the six
`Transform2D.fit` calls.

A global `!SIM.spectral.*` bang-string was tried first and rejected. The fit
happens during `SpectralTrace.__init__`, and several tests construct trace
lists with `cmds=None` while `scopesim.rc.__currsys__` is patched to a bare
`MagicMock` (`test_OpticalTrain.py`'s `patch_globals`). Any bang-key resolved
at construction time fails there, which is why `trace_flux_jacobian` gets away
with it — it is only read later, inside `_trace_flux_scale`.

Per-trace is also the right scope on the merits: this is a property of one
fit, not of the simulation. `fit_degree` is the precedent, already overridden
for exactly one instrument in `ZShooter_SPEC.yaml`.

**Config impact:** ZShooter and X-shooter must set `fit_normalize: true` on
their trace effects to keep today's numbers. See the companion irdb branch.

## Tests

Two existing tests were named for normalization but passed with it on *or* off,
so they were not testing it:

- `test_fit_preserves_high_dynamic_range_coefficients` — now explicitly
  `normalize=False`; it is really a float64-dtype and accuracy test.
- `test_gradient_with_normalized_fit_uses_input_units` — now explicitly
  `normalize=True`, so the `dpretransform` chain-rule factor is actually
  exercised. With `normalize=False` that factor is 1 and the test is vacuous.

New:

- `test_normalization_conditions_poorly_scaled_inputs` — the uncentred case,
  asserting >100x improvement.
- `test_normalization_is_immaterial_for_centred_focal_plane_mm` — both routes
  below 1e-12, documenting that enabling it is safe.
- `TestSpectralTraceFitNormalize` — the parameter defaults off, opting in
  populates `pretransform_x`/`dpretransform_x`, both routes agree on the mapped
  wavelength and slit coordinates to 1e-9, and gradients agree to 1e-7, i.e.
  stay in per-mm units either way.

## Deliberately not included

`fit2matrix` builds the coefficient matrix as float64 where upstream used
float32. That is left unconditional and unchanged by this branch: float32 for a
degree-4-or-higher 2D polynomial coefficient matrix is indefensible regardless
of normalization, and it deserves its own one-line PR with residual numbers
rather than being bundled into a behaviour toggle.

`xilam2xy_fit`, `xy2xilam_fit` and `_xiy2xlam_fit` in the same module are dead
— zero call sites anywhere including tests — but they are upstream code
(oczoske, 2021) and untouched by this fork, so removing them is upstream's
call, not ours.

## Verification

`scopesim/tests`: 4 failed, 1382 passed. Same four pre-existing failures as on
`main`.
