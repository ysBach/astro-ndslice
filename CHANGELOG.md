# Changelog

## Unreleased

### Fixes

- **Python 3.9:** installation required Python 3.10+, and imports used
  incompatible type annotations. Now supports Python 3.9+, including CI tests
  of the installed wheel.
- **Image placement:** slice tuples work directly as NumPy indices. Canvas and
  overlap sizes use the same rounded offsets, including half-pixel shifts.
  - `offsets2slice(..., fits_convention=False)` returned lists of slices, causing
    `array[indices[i]]` to raise `IndexError`. Now returns usable tuples.
  - `offseted_shape(shapes=[(3,), (3,)], offsets=[(0,), (1.5,)])` returned `(4,)`.
    Now returns `(5,)`, matching pixel placement.
  - `offseted_shape(..., method="inner")` returned `(0,)` for touching images.
    Now raises `ValueError` when there are no shared pixels.
  - `offseted_shape(shapes=[(2.5,)], offsets=[(0,)])` silently rounded to `(2,)`.
    Now raises `ValueError`.
- **FITS conversion:** reverse slices retain their selected pixels;
  unrepresentable slices raise `ValueError`.
  - `slice(None, None, -1)` became `[:1]`, selecting only the first pixel.
    Now becomes `[:1:-1]`, preserving the full reversal.
  - `slice(2, 2)` became `[3:2]`, selecting two pixels instead of none.
    Now raises `ValueError` when converting to FITS.
- **WCS and headers:** coordinate arrays work. The default `ignore_ltm=False` applies
  LTM scaling with a matrix solve; singular or invalid transforms raise
  `ValueError`, including under `python -O`.
  - `loc_target=np.array([2, 3])` raised an ambiguous-truth `ValueError`.
    Now accepts coordinate arrays.
  - `loc_target="center"` with unknown `WCS.pixel_shape` used zero dimensions.
    Now raises `ValueError`.
  - `LTV1=6, LTM1_1=2` returned `[6]` by default, ignoring scaling. Now returns
    `[3.]`; `ignore_ltm=True` retains the raw value. Nonidentity LTM matrices
    rejected by the earlier validation-only fix are now supported.
- **Input helpers:** `listify()` broadcasts multiple inputs consistently;
  `slicefy()` accepts generators; `bezel2slice(None)` returns full slices.
- **Development:** explicit exports preserve public names. Development
  dependencies include Astropy and isort.
- **Releases:** Hatchling builds explicit package contents. CI tests installed
  distributions; GitHub releases publish the tested files to PyPI after version,
  tag, changelog, lint, and test checks pass.

### Migration checklist

1. **LTM:** add `ignore_ltm=True` to retain raw LTV differences. The new default
   applies LTM correction and rejects invalid transforms.
2. **Image indices:** replace whole tuples from `offsets2slice()`; their elements
   are no longer mutable.
3. **Geometry:** use matching shapes/offsets, nonnegative integer sizes, and finite
   offsets. Empty overlap raises `ValueError`; half-pixel canvas sizes may change.
4. **FITS slices:** keep negative bounds and empty slices in Python form with
   `fits_convention=False`.
5. **WCS centers:** set `WCS.pixel_shape`, pass coordinates, or use `"origin"`.

Also: `listify(..., scalar2list=False)` affects only single-input calls;
`slicefy()` rejects empty iterable rules.
Details: [offsets](astro_ndslice/offset.py), [slices](astro_ndslice/slices.py).
