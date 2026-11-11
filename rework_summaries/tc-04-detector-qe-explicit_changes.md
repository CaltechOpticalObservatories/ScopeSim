# tc-04-detector-qe-explicit

Makes the detector-QE hand-off explicit and narrows two over-broad fallbacks.
**No number changes**: byte-identical on a 216-point ZShooter fingerprint.

## 1. The side-channel becomes a declared attribute

`TaperedQuantumEfficiency.apply_to` does not attenuate anything. It tags the
field of view, and the spectral trace mapping picks the tag up later:

```python
obj.meta["detector_qe"] = self          # ter_curves.py, the only writer
...
detector_qe = fov.meta.get("detector_qe")   # the only reader
```

One writer, one reader, communicating through a magic string in a dict that
nothing declares or validates. `FieldOfView` now carries
`self.detector_qe = None` as a documented attribute, and both sides use it.
`.get()` with its implicit `None` default is gone, so a typo is an
`AttributeError` rather than silently un-attenuated traces.

The docstring on the attribute records the ordering requirement, which is the
genuinely fragile part: the QE effect is z_order 610 and the trace list 670,
but `OpticsManager.get_z_order_effects` has its sort commented out, so the
ordering holds only because of the order effects are listed in the instrument
YAML.

**Not done, deliberately:** QE is not moved to a detector-stage z_order. It
would be easy to read the original review as asking for that, but upstream's
own `QuantumEfficiencyCurve` is z_order 513 and is applied to the `Source` —
*earlier* than ours, not later. Neither upstream nor this fork has ever applied
QE at an 8xx stage, so moving it would be a divergence, not a convergence.

## 2. `detector_qe_at` dispatches on the signature, not on a caught error

```python
try:
    return detector_qe.throughput(wave, detector_x=..., detector_y=...)
except TypeError:
    pass
return detector_qe.throughput(wave)
```

The intent was "fall back if this implementation does not accept positions".
What it actually did was swallow **any** `TypeError`, including one raised
inside a position-aware `throughput` that was simply broken — silently
degrading to position-independent QE and quietly changing the physics.

Now it inspects the signature for `detector_x` or `**kwargs` and calls
accordingly. A genuine failure inside the callee propagates.

## 3. `_diffuse_positions` stops advertising an argument it ignores

It accepted `footprint` and never referenced it. I looked into why before
changing anything, since the obvious "fix" would have moved numbers.

`TaperedQuantumEfficiency` models a graded coating whose bandpass centre shifts
along the cross-dispersion axis, so each echelle order lands on coating tuned
near its own wavelength. Non-dispersed diffuse light reaches the whole
detector, so `effective_diffuse_throughput` averages the coating over detector
positions — and `PostDisperserDiffuseBackground` collapses to a single scalar
rate per image plane, so that average is inherently detector-wide.

The configured span **is** the detector span in the shipped configuration: the
class is used only for b/g/r, which are image planes 0/1/2, all `y_size: 4096`,
with `position_min: 0, position_max: 4096`. (The 2048-tall K arm uses
`SpectralQuantumEfficiency`, which has no position dependence at all.) So
ignoring the footprint is numerically identical to honouring it today.

There is no rationale on record — `b928d4e9 "Add tapered detector quantum
efficiency"` is the entire commit message and there is no design note.

Accordingly: **behaviour unchanged.** The parameter is removed from the private
helper and the invariant that makes this correct is written into the docstring.
`effective_diffuse_throughput(footprint=...)` keeps the parameter, because that
is the shared protocol. `uses_detector_footprint = True` stays: it is read only
at `illumination.py:736`, where it widens a cache key, which is conservative
and correct.

The latent gap is now documented rather than fixed: `apply_to` warns when the
taper span differs from the detector span, but that warning never reaches the
diffuse path, so a future configuration with a mismatched span would average
over detector rows that do not exist.

## Tests

- The existing side-channel test now asserts the attribute.
- New `TestDetectorQePositionDispatch`: a position-aware `throughput` receives
  the coordinates; a spectral-only one is called without them; and a `TypeError`
  raised *inside* a position-aware implementation now propagates instead of
  being mistaken for "does not support positions".
- `test_calibrated_spectral_trace.py`'s `SimpleNamespace` fake field of view
  gained `detector_qe=None`, so the fake matches the real contract rather than
  relying on `.get()` returning `None` for a missing key.

## Verification

- `scopesim/tests`: 4 failed, 1379 passed — the same four pre-existing failures
  as `main`.
- ZShooter `SPEC` fingerprint (216 quantities): **byte-identical to `main`**.
