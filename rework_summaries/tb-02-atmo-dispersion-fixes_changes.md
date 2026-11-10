# tb-02-atmo-dispersion-fixes

Bug fixes to `scopesim/effects/atmo_dispersion.py`. No new features, no flags,
no behaviour that was previously correct is changed.

**Scope note.** `atmo_dispersion.py` stays a parallel alternative to
`scopesim/effects/shifts.py`, which is byte-identical to upstream. This branch
deliberately does **not** merge the two or offer one as a replacement for the
other. The whole file is new in this fork, so there is no upstream behaviour to
preserve here, only defects to remove.

## 1. The interpolation method is now required (the substantive fix)

`ShiftFoV3D.apply_to` chose between `"roll"` and `"map_coords"` from the median
shift magnitude of whichever FOV reached it first, then cached that choice on
`self.meta["method"]`:

```python
if self.meta.get("method", None) is None:
    if np.sqrt(np.median(dxs)**2 + np.median(dys)**2) >= 10:
        self.meta["method"] = "roll"
    else:
        self.meta["method"] = "map_coords"
```

One effect instance is shared across every FOV of an `OpticalTrain`, so the
first FOV permanently decided the algorithm for all later ones, including
wavelength ranges where the shift is sub-pixel (`roll` quantises it to zero) or
very large (`map_coords` does an unnecessary spline).

The same stickiness disagreed with `_padded_header`, which `observe()` calls
**before** the first `apply_to`. At that point `method` is still `None`, so
padding was sized under the `map_coords` assumption (`guard = 4`, no integer
rounding) regardless of what would later be applied. That particular
disagreement over-pads rather than under-pads, so it wastes memory rather than
corrupting data — but the two code paths were reasoning about different
effects.

How the shift is applied is a physical choice about the data, so it is now
configured rather than inferred. `check_keys` raises if it is absent, and the
existing value check stays.

**Config impact: none.** Both production configurations already pin it —
`irdb/MaunaKea/MaunaKea.yaml` sets `method: "roll"` and
`irdb/XSHOOTER/XSHOOTER_SPEC.yaml` sets `method: map_coords`. Neither ever
exercised the auto-selection branch. Only callers that omitted `method`
entirely are affected.

This also removes a `getattr(self, self.meta["method"] + "_shift")` dispatch on
a string that was validated at `__init__` and then mutated afterwards by the
block above. It is now an explicit two-way choice.

### A runtime break this surfaced

`ADCShift.get_shifts` builds an inner `ADShift` purely to obtain the
*uncorrected* dispersion, which it then differences against a perturbed zenith
angle. The `ad_kwargs` dict it passes contains only `temperature`, `pressure`,
`humidity`, `x_co2` and `wave_ref_um` — no `method`. Making the key required
would therefore have raised on every ZShooter run through `MaunaKea.yaml`, and
**no existing test covered it**.

The inner effect now inherits the outer one's method. It never applies a shift,
so the value is immaterial to it; inheriting keeps the setting in one
configured place. `test_adc_residual_shifts_inherit_the_configured_method`
covers the path, and was confirmed to fail without the fix.

## 2. ASCII source

`AGENTS.md` asks for ASCII-only source. `refractive_index` used literal Unicode
Greek letters as variable names and the unit comments carried middle dot, micro
and degree signs — 34 lines in total. All renamed to spelled-out ASCII
(`sigma`, `alpha`, `beta`, `gamma`, `rho_a`, `rho_w`, `rho_axs`, `rho_ws`), and
comments rewritten as `K Pa^-1`, `um^-2`, `deg C`.

Renames only. Verified bit-identical output at Ciddor's reference condition:
`n = 1.0002765302104355` and `Z_a = 0.9995922115359966` before and after.

## 3. Smaller defects

- **Duplicate import.** `SkyCoord` was imported twice, on lines 5 and 9, with
  unrelated imports in between. `Any` was imported from `typing` and never
  used. Both removed.
- **The 0.1 pixel quantisation is now named.** `np.round(..., 1)` appears in
  both `apply_to` and `_padded_header`, and exists so that many wavelength
  planes share one cached sparse interpolation operator. It is also a hard
  floor on the accuracy of the applied dispersion, which was not stated
  anywhere. The value is unchanged; it is now `SHIFT_QUANTIZATION_DECIMALS`
  with a comment saying what it costs, so it is greppable.
- **The cached shift operator is returned read-only.** `_spatial_shift_operator`
  is an `lru_cache`'d function handing out a mutable `csr_array` by reference.
  Its one caller only reads it, but nothing enforced that. Now mirrors the
  precedent in `psfs/analytical.py`, where `_legendre_rule` already sets
  `writeable = False` on its cached arrays.
- **`ADShift.apply_to` deleted.** It was a pure pass-through to `super()`.
- **`ADCShift.z_order` annotated as `ClassVar`,** consistent with `ShiftFoV3D`
  and `ADShift` in the same file and with every class in `shifts.py`.

## 4. Tests

`refractive_index` and `compressibility` had **no direct test anywhere**. They
now check against published Ciddor 1996 values:

- `n = 1.00027653` at 633 nm, 15 deg C, 101325 Pa, dry air, 450 ppm CO2 —
  reproduced to 8 significant figures
- `Z_a = 0.9995922115` for dry air at 288.15 K, 101325 Pa — Ciddor's own
  Appendix value, reproduced exactly
- normal dispersion across 0.35 / 0.633 / 1.5 um
- moist air less refractive than dry at the same temperature and pressure
- refractivity nearly proportional to pressure
- compressibility approaching the ideal-gas limit as pressure falls

Plus coverage that `method` is required, that an unknown value is rejected, and
the `ADCShift` inheritance path above.

## Review notes

- The `np.round(..., 1)` floor is documented, not changed. Whether 0.1 pixel is
  an acceptable systematic for the ADC residuals is a separate question.
- `is_night` still returns `bool | Time` and `resolve_time` disambiguates with
  `isinstance`. Left alone; noted as a wart.
- `ADCShift._adc_nir_resid` / `_adc_ub_resid` / `_adc_gri_resid` contain
  hard-coded band-remapping constants with no test. Out of scope here.

## Verification

`scopesim/tests`: 4 failed, 1389 passed — the same four pre-existing failures
as on `main` (two `test_basic_instrument` flux regressions, `test_import_all`
from an untracked local file, and the stale waveset-cache test removed on
`tb-05`).
