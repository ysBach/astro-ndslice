# astro-ndslice

[![DOI](https://zenodo.org/badge/601666411.svg)](https://zenodo.org/badge/latestdoi/601666411)

A collection of simple tools to slicing astronomical images (many are applicable to general ndarray).

* Module `lists`: Contains some universally usable convenience functions.
  * `is_list_like` : Checks if anything is list-like (list, tuple, ndarray).
  * `listify`: Convert scalar or iterable into list (``scalar2list=True, none2list=False``)
  * `ndfy`: Convert scalar or iterable into ndarray (see docstring for details)
* Module `slices`: Contains tools to convert FITS convention (e.g., "[10:20, 1:2]") or bezel (e.g., `bezel=[10, 10]`) into python `slice`.
  * Most users will only need `slicefy` (e.g., ``data[slicefy("[10:110, 50:150]")]``).
* Module `offset`: Contains tools to find the offsets between two images/FITS files (essential before image combination)
  * To use `calc_offset_wcs` or `calc_offset_physical`, `astropy` is required.