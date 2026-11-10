# tb-01-utils-observation-helpers

Two unrelated cleanups in `scopesim/utils.py`, in the observation-info block
added to support the sky, PSF and atmospheric-dispersion effects. No behaviour
change on any path that currently works.

## 1. Narrow the bare `except`

`get_observation_info_from_cmds` resolves the optional `!OBS` pointing keys and
treats a missing one as "not configured":

```python
for k, v in target_kwargs.items():
    try:
        target_kwargs[k] = from_currsys(v, cmds)
    except:                                  # <- bare
        logger.debug(f'Target coord {v} not set in !OBS config.')
        target_kwargs[k] = None
```

The `try` wraps a single call where `v` is always a literal `"!OBS.*"` string.
The only failure it is meant to absorb is `from_currsys`'s own
`raise ValueError(f"{item} was not found in rc.__currsys__")`.

As written it also swallowed `KeyboardInterrupt`, `SystemExit`, `MemoryError`,
`RecursionError`, and any `AttributeError` or `TypeError` from a malformed
`cmds` — converting each into `target_kwargs[k] = None`. Those `None`s then
flow into `get_target`, which raises
`ValueError("No valid target coordinates provided...")`. So a real failure
anywhere underneath surfaced as a confident, wrong diagnosis about pointing
configuration.

Now `except ValueError:`. A missing key still behaves exactly as before.

This matters because all three consumers call it during `__init__`:
`PalaceAirglowEmission`, `SkyBackgroundTERCurve` and `TAPASTERCurve` in
`sky_ter_curves.py`, plus `ADShift`/`ADCShift`. A swallowed error at
construction is hard to trace back.

## 2. Delete `get_next_moon` and `_get_next_moon_cached`

Both are dead. The only call site has been commented out since it was written,
in `resolve_time`:

```python
# if t is None:
#     kw = {"bright":"full", "gray":"half", "dark":"new"}
#     t = Time(get_next_moon(kw[time_str], location), format="isot", location=location)
```

They are also wrong in two ways that would bite if revived:

- `_get_next_moon_cached` is decorated `@functools.lru_cache(maxsize=32)` but
  calls `Time.now()` **inside** the cached body, against a cache key whose time
  component has only calendar-day resolution. Within one day the result is
  frozen at the first call's wall clock, so it can be up to 24 hours stale.
- It raises `ValueError("location_key must be provided.")` when `location_key`
  is `None`, which is its own default.

Removed along with the commented call site. Recoverable from history if the
brightness-word moon-phase path is ever finished.

No imports became unused: `functools` is still used by `@functools.wraps`,
`get_body` by `get_moon_phase`, and `get_sun`/`AltAz`/`EarthLocation` by
`is_night`, `get_zenith_angle` and `get_location`.

## Tests

- `test_absent_coordinate_keys_fall_back_to_alt_az` — with `!OBS.ra`,
  `!OBS.dec` and `!OBS.airmass` unconfigured, the target still resolves from
  alt/az, i.e. the optional-key behaviour is preserved.
- `test_unrelated_failure_propagates_instead_of_becoming_none` — monkeypatches
  `from_currsys` to raise `RuntimeError` for one key and asserts it propagates
  rather than being flattened to `None`.

## Review notes

- `is_night` returns `bool | Time` depending on `return_midnight`, and
  `resolve_time` disambiguates with `isinstance(nighttime, Time)`. Latent type
  confusion if that default ever flips. Not touched here.
- When `!OBS.brightness` is a word rather than an MJD, `resolve_time` falls back
  to `Time.now()` and the brightness is not used to pick a moon phase. That is
  the feature the deleted helpers were written for; it remains unimplemented.

## Verification

`scopesim/tests`: 4 failed, 1378 passed. Same four pre-existing failures as on
`main`; the two added tests pass.
