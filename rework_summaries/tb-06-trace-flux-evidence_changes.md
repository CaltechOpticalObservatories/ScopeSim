# tb-06-trace-flux-evidence

Tests only. No production code changes.

## Why there is no code change

`SpectralTrace._trace_flux_scale` is already structured the way the rest of
this Tier B series is trying to reach:

```python
use_jacobian = from_currsys(self.meta["trace_flux_jacobian"], self.cmds)

if not use_jacobian:
    dlam_by_dx, dlam_by_dy = self.xy2lam.gradient()
    dlam_per_pix = pixsize * np.sqrt(dlam_by_dx(x_mm, y_mm)**2 + dlam_by_dy(x_mm, y_mm)**2)
    return pixscale * dlam_per_pix          # upstream, byte for byte

dxi_by_dx, dxi_by_dy = self.xy2xi.gradient()
dlam_by_dx, dlam_by_dy = self.xy2lam.gradient()
jacobian = (dxi_by_dx(x_mm, y_mm)*dlam_by_dy(x_mm, y_mm)
            - dxi_by_dy(x_mm, y_mm)*dlam_by_dx(x_mm, y_mm))
return np.abs(jacobian) * self._detector_pixel_area(det_header)
```

`defaults.yaml` already ships `trace_flux_jacobian : False`, so upstream
behaviour is already the default, and both ZShooter and X-shooter already opt
in. The flag is also overridable per instance as a constructor kwarg.

What was missing was the argument for making it the default upstream.

## What the existing test proved, and what it did not

`test_trace_flux_jacobian_handles_tilted_trace` asserted only that the
projected scale is *larger* than the Jacobian on one tilted trace, plus that
the Jacobian equals 1e-3 there. A sign, not a magnitude — and nothing showing
how the error behaves.

## The analytic result

For the `_tilted_linear_trace_table` family, with

    x = lam + tilt*xi,    y = xi/10

the inverse transforms are exactly

    xi  = 10y
    lam = x - 10*tilt*y

so the area Jacobian is

    |d(xi,lam)/d(x,y)| = |0*(-10*tilt) - 10*1| = 10

**independent of tilt**, while the projected scale picks up

    sqrt((dlam/dx)^2 + (dlam/dy)^2) = sqrt(1 + (10*tilt)^2)

The ratio is therefore `sqrt(1 + (10*tilt)^2)`, which grows without bound as
the slit shears, and equals 1 only for an unsheared trace:

| tilt | over-count |
|---:|---:|
| 0.0 | 1.000 |
| 0.05 | 1.118 |
| 0.1 | 1.414 |
| 0.2 | 2.236 |
| 0.3 | 3.162 |

The upstream formula measures the length of the wavelength gradient and
multiplies by the slit plate scale, which is the area of a *rectangle*. The
true cell is a parallelogram, and the two agree only when it happens to be
rectangular. That is the case for an unsheared trace and for nothing else.

## Changes

- `_tilted_linear_trace_table` takes a `tilt` argument, defaulting to `0.1` so
  the existing caller is unaffected.
- `test_projected_scale_overcounts_by_a_known_factor`, parametrised over five
  tilts, asserting the Jacobian is 1e-3 at every tilt and that the ratio
  matches `sqrt(1 + (10*tilt)^2)` to 1e-5.
- `test_the_two_agree_only_for_an_unsheared_trace`.

## Still outstanding for an upstream proposal

Before asking AstarVienna to flip the default, this wants before/after numbers
from a real METIS LSS and a MICADO spectroscopy run, not just the analytic
family above. The issue thread on METIS LMS rectification systematics suggests
that is the kind of evidence the maintainers respond to. Not done here.

`_detector_pixel_area` is covered separately by
`test_detector_jacobian_uses_physical_pixel_area`, including asymmetric pixels
given in metres.

## Verification

`scopesim/tests`: 4 failed, 1382 passed. Same four pre-existing failures as on
`main`.
