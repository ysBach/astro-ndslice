# astro-ndslice

[![DOI](https://zenodo.org/badge/601666411.svg)](https://zenodo.org/badge/latestdoi/601666411)

Slice NumPy arrays, convert FITS/IRAF sections, and align images by pixel offsets.

## Install

```bash
pip install astro-ndslice
```

Python >= 3.10.

For astronomers who need WCS/header offset helpers:

```bash
pip install "astro-ndslice[full]"
```

## Slice an image

```python
import numpy as np
from astro_ndslice import slicefy

data = np.arange(100).reshape(10, 10)
cutout = data[slicefy("[2:5,3:7]")]  # Same as data[2:7, 1:5].
trimmed = data[slicefy(1)]          # Trim one pixel from each edge - bezeling.
```

* **FITS sections** use **1-based, inclusive bounds in x, y order**.
* **NumPy** uses **0-based, exclusive stops in row, column order**.
* `slicefy()` assumes FITS strings
* `slice_from_string()` assumes Python strings unless you pass `fits_convention=True`.

## Align images

```python
import numpy as np
from astro_ndslice import offseted_shape, offsets2slice

images = [np.ones((3, 4)), np.full((2, 3), 2)]
shapes = [image.shape for image in images]  # NumPy axis order.
offsets = [(0, 0), (2, 1)]                # x, y offsets.
_, shape = offseted_shape(shapes, offsets)
indices = offsets2slice(shapes, offsets)
stack = np.full((len(images), *shape), np.nan)
for image, index in zip(images, indices):
    stack[index] = image
```

Both helpers round relative offsets to pixel positions. Use `method="inner"`
to find the region shared by every image; no shared pixels raises `ValueError`.

## Function guide

| Task | Functions |
| --- | --- |
| Normalize scalars and iterables | `is_list_like`, `listify`, `ndfy` |
| Parse sections and trim edges | `slicefy`, `slice_from_string`, `slice_to_string`, `bezel2slice` |
| Place images and find overlap | `regularize_offsets`, `offseted_shape`, `offsets2slice` |
| Read WCS or physical offsets | `calc_offset_wcs`, `calc_offset_physical` (requires Astropy) |

`calc_offset_physical()` corrects LTV translation terms
for LTM scaling in shared physical-coordinate units. For example,
`LTV1=6, LTM1_1=2` gives `3` along x; `ignore_ltm=True` gives the raw value `6`.
This preserves the LTV sign; differing LTM matrices require resampling for
image alignment.

See [CHANGELOG.md](CHANGELOG.md) for return-type changes and stricter validation.
For version bumps and PyPI publishing, see [RELEASING.md](RELEASING.md).
