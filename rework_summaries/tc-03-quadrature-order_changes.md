# tc-03-quadrature-order

Makes the per-pixel quadrature order configurable, defaulting to upstream's
sampling. Pairs with irdb branch `rework/tc-03-quadrature-opt-in`, which must
merge with it.

## What this does and, importantly, what it does not

The fork replaced upstream's single sample at each pixel centre with a
Gauss-Legendre integration of the spectral density across the pixel. That is
better physics, but it silently changes results for every instrument that uses
`SpectralTrace`, including METIS, MICADO and Mosaic.

`_pixel_quadrature_order` was a plain class attribute equal to 4 — not
configurable at all. It is now `!SIM.spectral.pixel_quadrature_order`,
**default 1**, resolved through a new `SpectralTrace.quadrature_order()`.

**Order 1 is exactly upstream's scheme.** `leggauss(1)` returns one node at 0
with weight 2, and the loop uses `nodes/2, weights/2`, so it degenerates to a
single sample at the pixel centre with weight 1. Verified numerically, and the
weights sum to 1 at every order, so flux is conserved either way.

**This is not upstream bit-parity, and the branch should not be read as
claiming it.** With the order at 1, METIS/MICADO/Mosaic get upstream's
*sampling*, but they still get the fork's `XiLamImage`, `_set_dispersion` and
`Transform2D` normalisation. Full parity would need both `XiLamImage` variants
coexisting — the fork's requires `fov.meta["eta_min"]/["eta_max"]`, which do not
exist anywhere upstream, and it changed the wavelength origin
(`all_pix2world(..., 1)` → `..., 0`), the `xi` origin and the whole image
algorithm. That is roughly 200 lines of parallel code in one module, and it was
judged not worth it. Recording the cost here so the upstream conversation can
start from facts rather than being rediscovered.

A second sleeper, out of scope but worth knowing: upstream's `_set_dispersion`
accepted a `pixsize` argument and then unconditionally overwrote it; the fork
honours it. That changes `dlam_per_pix` — the entire slit-shear amplitude —
whenever `CDELT1D != pixel_scale/plate_scale`, independently of this flag.

## Why a config key and not a subclass

An echelle-only `SpectralTrace` subclass looked attractive, but
`XSHOOTER_SPEC.yaml:81` uses the plain `SpectralTraceList`, not the echelle
class — so a subclass would have silently reverted XSHOOTER to upstream
mapping. A key lets both packages opt in explicitly.

## Measured effect

On ZShooter `SPEC`, order 4 against order 1:

| | image plane sum | readout sum |
|---|---:|---:|
| B / G / R | −0.002% / −0.024% / +0.014% | ~0% / −0.0002% / +0.0002% |
| YJ / H / K | +0.026% / +0.030% / +0.028% | +0.022% / +0.029% / +0.030% |

Small, but not nothing, and it is a real integration-vs-point-sampling
difference rather than noise — which is why both packages opt in.

On upstream's own `test_basic_instrument`, the quadrature accounts for about
0.5% of the spectroscopy flux discrepancy (317583 → 319093 at order 1). The
remaining ~57% comes from the unflagged spatial projection rework, not from
anything here.

## Placement detail

The key is deliberately the **first** entry in `SpectralTrace._class_params`.
The open PR `rework/tb-04-transform2d-normalize` inserts `fit_normalize`
immediately after `"fit_degree": 4,`, so adding there too would have produced a
pointless merge conflict. For the same reason the tests live in a new file
rather than being a third appender to `test_SpectralTraceListUtils.py`, which
both tb-04 and tb-06 already extend.

## Tests

New `scopesim/tests/tests_effects/test_pixel_quadrature.py`: the default
resolves to 1; order 1 is a single centre sample of unit weight; order 4 samples
away from the centre but stays inside the pixel; weights sum to 1 at every
order; orders below 1 raise; and a per-trace override is honoured.

## Verification

- `scopesim/tests`: 4 failed, 1385 passed — same four pre-existing failures as
  `main`, nine new tests passing.
- ZShooter `SPEC` fingerprint (216 quantities) with the irdb opt-in applied:
  **byte-identical to `main`**.
- Without the opt-in: 24 differing lines, confirming the key is load-bearing
  and that the two branches must merge together.
