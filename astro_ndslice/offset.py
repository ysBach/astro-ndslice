import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "regularize_offsets",
    "offseted_shape",
    "offsets2slice",
    "calc_offset_wcs",
    "calc_offset_physical",
]


def _normalize_shapes(shapes: ArrayLike, offsets: np.ndarray) -> np.ndarray:
    """Validate and normalize image shapes against regularized offsets."""
    try:
        _shapes = np.atleast_2d(np.asarray(shapes))
    except (TypeError, ValueError) as exc:
        raise ValueError("shapes must be a rectangular numeric array.") from exc

    if _shapes.ndim != 2 or offsets.ndim != 2:
        raise ValueError("Shapes and offsets must be at most 2-D.")

    if _shapes.shape != offsets.shape:
        raise ValueError("shapes and offsets must have the identical shape.")

    try:
        valid = (
            np.all(np.isfinite(_shapes))
            and np.all(_shapes >= 0)
            and np.all(_shapes == np.floor(_shapes))
        )
    except TypeError as exc:
        raise ValueError("shapes must contain finite non-negative integers.") from exc
    if not valid:
        raise ValueError("shapes must contain finite non-negative integers.")

    # Use the exclusive limit for floating inputs: float(intp.max) can round
    # up to the first unrepresentable integer.
    limit = np.iinfo(np.intp).max
    out_of_range = (
        np.any(_shapes >= limit + 1)
        if np.issubdtype(_shapes.dtype, np.floating)
        else np.any(_shapes > limit)
    )
    if out_of_range:
        raise ValueError("shapes exceed the platform index range.")

    return _shapes.astype(np.intp, copy=False)


def regularize_offsets(
    offsets: np.ndarray, offset_order_xyz: bool = True, intify_offsets: bool = False
) -> np.ndarray:
    """Shift each offset axis so its minimum is zero.

    Parameters
    ----------
    offsets : array-like
        Finite offsets shaped ``(nimage, ndim)``. A 1D input represents one
        image and is promoted to 2D.
    offset_order_xyz : bool, optional
        Read offsets in xyz order and reverse to NumPy axis order.
        Default: `True`.
    intify_offsets : bool, optional
        Round relative offsets to integers using nearest-even rounding.
        Default: `False`.

    Returns
    -------
    ndarray
        Relative offsets in NumPy axis order, with zero minimum per axis.

    Notes
    -----
    Setup: ``o = [(0, 0), (2.5, -1)]`` (two 2D images).

    Timing on MBP 14" [2024, macOS 26.6, M4Pro(8P+4E/G20c/N16c/48G)]
    (2026-09-07; CPython 3.13.11, NumPy 2.4.6)::

        regularize_offsets(o)  3.255 +/- 0.033 us

    Mean +/- std. dev. per call (`timeit`, 7 runs; 100,000 loops each).
    """
    _offsets = np.atleast_2d(offsets)
    if _offsets.ndim != 2:
        raise ValueError("offsets must be at most 2-D.")
    if _offsets.shape[0] == 0:
        raise ValueError("offsets must contain at least one image.")
    if offset_order_xyz:
        _offsets = _offsets[..., ::-1]
    try:
        valid = np.all(np.isfinite(_offsets))
    except TypeError as exc:
        raise ValueError("offsets must contain finite numeric values.") from exc
    if not valid:
        raise ValueError("offsets must contain finite numeric values.")

    _offsets = _offsets - np.min(_offsets, axis=0)
    if intify_offsets:
        _offsets = np.rint(_offsets).astype(int)

    return _offsets


def offseted_shape(
    shapes: np.ndarray,
    offsets: np.ndarray,
    method: str = "outer",
    offset_order_xyz: bool = True,
    intify_offsets: bool = False,
    pythonize_offsets: bool = True,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Calculate relative offsets and a bounding or overlapping pixel shape.

    Parameters
    ----------
    shapes : array-like
        Image shapes in NumPy axis order, shaped ``(nimage, ndim)``.
        Values must be finite nonnegative integers within the index range.
    offsets : array-like
        Finite image positions, with the same shape as `shapes`.
        A common translation is removed before computing pixel bounds.
    method : {"outer", "inner"}, optional
        `"outer"` bounds all images, including gaps; `"inner"` gives their overlap.
        Default: `"outer"`.
    offset_order_xyz : bool, optional
        Read offsets in xyz order. Default: `True`.
    intify_offsets : bool, optional
        Round returned offsets. Pixel bounds always use nearest-even rounding,
        matching ``offsets2slice()``.
        Default: `False`.
    pythonize_offsets : bool, optional
        Return offsets in NumPy axis order. If `False`, retain the input
        axis order. Default: `True`.

    Returns
    -------
    offsets : ndarray
        Relative offsets, with zero minimum per axis.
    shape_out : tuple of int
        Pixel shape in NumPy axis order; no stack axis is included.

    Raises
    ------
    ValueError
        If inputs are invalid or `"inner"` has no shared pixels.

    Notes
    -----
    Setup: ``s = [(100, 120), (80, 100)]``;
    ``o = [(0, 0), (2.5, -1)]`` (two 2D images).

    Timing on MBP 14" [2024, macOS 26.6, M4Pro(8P+4E/G20c/N16c/48G)]
    (2026-09-07; CPython 3.13.11, NumPy 2.4.6)::

        offseted_shape(s, o)  11.408 +/- 0.348 us
        offseted_shape(s, o, method="inner")  13.971 +/- 0.373 us

    Mean +/- std. dev. per call (`timeit`, 7 runs; 20,000 loops each).
    """

    _offsets = regularize_offsets(
        offsets, offset_order_xyz=offset_order_xyz, intify_offsets=intify_offsets
    )
    _shapes = _normalize_shapes(shapes, _offsets)
    # ``offsets2slice`` rounds offsets because NumPy slices require integer
    # bounds.  Use the same placement here so that its slices always fit the
    # shape returned by this function, including half-integer offsets.
    _placement_offsets = np.rint(_offsets).astype(int)

    if method == "outer":
        shape_out = np.max(_shapes + _placement_offsets, axis=0)
    elif method == "inner":
        lower_bound = np.max(_placement_offsets, axis=0)
        upper_bound = np.min(_placement_offsets + _shapes, axis=0)
        shape_out = upper_bound - lower_bound
        if np.any(shape_out <= 0):
            raise ValueError(
                "There doesn't exist fully-overlapping pixel! "
                + f"Naïve output shape={shape_out}."
            )
    else:
        raise ValueError("method unacceptable (use one of 'inner', 'outer').")

    if offset_order_xyz and not pythonize_offsets:
        # reverse _offsets to original xyz order
        _offsets = _offsets[..., ::-1]

    return _offsets, tuple(shape_out)


def offsets2slice(
    shapes: np.ndarray,
    offsets: np.ndarray,
    method: str = "outer",
    shape_order_xyz: bool = False,
    offset_order_xyz: bool = True,
    outer_for_stack: bool = True,
    fits_convention: bool = False,
) -> list:
    """Build indices to place images in a canvas or extract their overlap.

    Parameters
    ----------
    shapes, offsets : array-like
        Matching arrays shaped ``(nimage, ndim)``. Shapes must be finite
        nonnegative integers within the index range; offsets must be finite.
        Relative offsets use nearest-even rounding to integer pixels.
    method : {"outer", "inner"}, optional
        `"outer"` places images in the output; `"inner"` extracts input overlap.
        Default: `"outer"`.
    shape_order_xyz : bool, optional
        Read shapes in xyz order. Default: `False` (NumPy axis order).
    offset_order_xyz : bool, optional
        Read offsets in xyz order. Default: `True`.
    outer_for_stack : bool, optional
        Include the leading stack axis for `"outer"`. Inner indices always
        address N-D input images. Default: `True`.
    fits_convention : bool, optional
        Return FITS strings with 1-based, inclusive bounds in xyz order.
        Otherwise, return tuples of Python slices. Default: `False`.

    Returns
    -------
    list of tuple of slice or list of str
        One index per image. Python output works as ``array[indices[i]]``.

    Raises
    ------
    ValueError
        If inputs are invalid, `"inner"` has no shared pixels, or FITS output
        would require an empty image section.

    Notes
    -----
    Setup: ``s = [(100, 120), (80, 100)]``;
    ``o = [(0, 0), (2.5, -1)]`` (two 2D images).

    Timing on MBP 14" [2024, macOS 26.6, M4Pro(8P+4E/G20c/N16c/48G)]
    (2026-09-07; CPython 3.13.11, NumPy 2.4.6)::

        offsets2slice(s, o)  12.171 +/- 0.488 us
        offsets2slice(s, o, method="inner")  16.992 +/- 0.283 us

    Mean +/- std. dev. per call (`timeit`, 7 runs; 20,000 loops each).

    Examples
    --------
    >>> import numpy as np
    >>> images = [np.ones((3, 4)), np.full((2, 3), 2)]
    >>> shapes = [image.shape for image in images]
    >>> offsets = [(0, 0), (2, 1)]
    >>> _, shape = offseted_shape(shapes, offsets)
    >>> indices = offsets2slice(shapes, offsets)
    >>> stack = np.full((len(images), *shape), np.nan)
    >>> for image, index in zip(images, indices):
    ...     stack[index] = image
    """
    _shapes = np.atleast_2d(shapes)
    if shape_order_xyz:
        _shapes = _shapes[..., ::-1]

    _offsets = regularize_offsets(
        offsets, offset_order_xyz=offset_order_xyz, intify_offsets=True
    )
    _shapes = _normalize_shapes(_shapes, _offsets)
    if fits_convention and np.any(_shapes == 0):
        raise ValueError("FITS sections cannot represent empty image dimensions.")

    if method == "outer":
        starts = _offsets
        stops = _offsets + _shapes
        include_stack_axis = outer_for_stack
    elif method == "inner":
        offmax = np.max(_offsets, axis=0)
        if np.any(np.min(_shapes + _offsets, axis=0) <= offmax):
            raise ValueError(
                "At least 1 frame has no overlapping pixel with all others. "
                + "Check if there's any overlapping pixel for images for the "
                + "given offsets."
            )

        # 1-D array +/- 2-D array:
        #   the former 1-D array is broadcast s.t. it is "tile"d along axis=-1.
        starts = offmax - _offsets
        stops = np.min(_offsets + _shapes, axis=0) - _offsets
        include_stack_axis = False
    else:
        raise ValueError("method unacceptable (use one of 'inner', 'outer').")

    slices = []
    for image_i, (start, stop) in enumerate(zip(starts, stops)):
        # NOTE: starts/stops are all in pythonic index
        if fits_convention:
            tmp = [
                f"{start_i + 1:d}:{stop_i:d}" for start_i, stop_i in zip(start, stop)
            ]
            if include_stack_axis:
                tmp.insert(0, f"{image_i + 1}:{image_i + 1}")
            slices.append("[" + ",".join(tmp[::-1]) + "]")  # order is opposite!
        else:
            tmp = [slice(start_i, stop_i, None) for start_i, stop_i in zip(start, stop)]
            if include_stack_axis:
                tmp.insert(0, slice(image_i, image_i + 1, None))
            slices.append(tuple(tmp))

    return slices


def calc_offset_wcs(
    target,
    reference,
    loc_target: str | ArrayLike = "center",
    loc_reference: str | ArrayLike = "center",
    order_xyz: bool = True,
    intify_offset: bool = False,
) -> np.ndarray:
    """Map a target pixel through WCS and subtract a reference pixel.

    Parameters
    ----------
    target, reference : astropy.wcs.WCS
        Target and reference coordinate systems.
    loc_target, loc_reference : {"center", "origin"} or array-like, optional
        Zero-based pixel locations in xyz order, with one value per axis.
        `"center"` uses half of each axis length and requires a known
        ``WCS.pixel_shape``. `"origin"` uses zeros. Default: `"center"`.
    order_xyz : bool, optional
        Return offsets in xyz order; use `False` for NumPy axis order.
        Default: `True`.
    intify_offset : bool, optional
        Round the result to integers using nearest-even rounding.
        Default: `False`.

    Returns
    -------
    ndarray
        Target location in reference pixels, minus `loc_reference`.

    Raises
    ------
    TypeError
        If either input is not an Astropy WCS.
    ValueError
        If a location name, dimension, or value is invalid, or a center
        calculation lacks an image shape.

    Notes
    -----
    Setup: two 100x100 TAN WCSs, ``w`` and ``r``, with CRVAL=(0, 0),
    CRPIX=(50, 50)/(52, 49), and CDELT=(-1/3600, 1/3600) deg/pixel.
    WCS construction is excluded; locations use the default center.

    Timing on MBP 14" [2024, macOS 26.6, M4Pro(8P+4E/G20c/N16c/48G)]
    (2026-09-07; CPython 3.13.11, NumPy 2.4.6, Astropy 7.2.0)::

        calc_offset_wcs(w, r)  13.374 +/- 0.097 us

    Mean +/- std. dev. per call (`timeit`, 7 runs; 20,000 loops each).
    """
    from astropy.wcs import WCS

    def _parse_loc(loc, obj):
        if isinstance(obj, WCS):
            w = obj
        else:
            raise TypeError("input must be an instance of astropy.wcs.WCS.")

        if isinstance(loc, str):
            if loc == "center":
                pixel_shape = w.pixel_shape
                if pixel_shape is None:
                    raise ValueError(
                        "loc='center' requires WCS.pixel_shape to be known."
                    )
                try:
                    _loc = np.asarray(pixel_shape, dtype=float) / 2
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "loc='center' requires a finite WCS.pixel_shape."
                    ) from exc
            elif loc == "origin":
                _loc = np.zeros(w.naxis, dtype=float)
            else:
                raise ValueError("loc must be 'center', 'origin', or a coordinate.")
        else:
            try:
                _loc = np.atleast_1d(np.asarray(loc, dtype=float))
            except (TypeError, ValueError) as exc:
                raise ValueError("loc must be a finite numeric coordinate.") from exc

        if _loc.ndim != 1 or _loc.size != w.naxis:
            raise ValueError(f"loc must contain exactly {w.naxis} coordinates.")
        if not np.all(np.isfinite(_loc)):
            raise ValueError("loc must be a finite numeric coordinate.")

        return w, _loc

    w_targ, _loc_targ = _parse_loc(loc_target, target)
    w_ref, _loc_ref = _parse_loc(loc_reference, reference)

    _loc_targ_coo = w_targ.all_pix2world(*_loc_targ, 0)
    _loc_targ_pix_ref = w_ref.all_world2pix(*_loc_targ_coo, 0)

    offset = _loc_targ_pix_ref - _loc_ref

    if intify_offset:
        offset = np.around(offset).astype(int)

    if order_xyz:
        return offset
    else:
        return offset[::-1]


def _check_ltm(hdr):
    ndim = hdr["NAXIS"]
    for i in range(ndim):
        for j in range(ndim):
            key = f"LTM{i + 1}_{j + 1}"
            try:
                value = float(hdr[key])
            except (KeyError, IndexError):
                continue
            except (TypeError, ValueError) as exc:
                raise NotImplementedError(
                    "Only an identity LTM matrix is supported."
                ) from exc
            if not np.isfinite(value) or value != float(i == j):
                raise NotImplementedError("Only an identity LTM matrix is supported.")

        try:  # Sometimes LTM matrix is saved as ``LTMi``, not ``LTMi_j``.
            value = float(hdr[f"LTM{i + 1}"])
        except (KeyError, IndexError):
            continue
        except (TypeError, ValueError) as exc:
            raise NotImplementedError(
                "Only an identity LTM matrix is supported."
            ) from exc
        if not np.isfinite(value) or value != 1.0:
            raise NotImplementedError("Only an identity LTM matrix is supported.")


def calc_offset_physical(
    target,
    reference=None,
    order_xyz: bool = True,
    ignore_ltm: bool = True,
    intify_offset: bool = False,
) -> np.ndarray:
    """Subtract FITS header LTV values to obtain a pixel offset.

    Parameters
    ----------
    target : astropy.io.fits.Header
        Header containing `NAXIS` and optional ``LTVi`` keywords.
        Missing ``LTVi`` values default to zero.
    reference : astropy.io.fits.Header, optional
        Subtract this header's LTV values. `None` returns the target's values.
        Default: `None`.
    order_xyz : bool, optional
        Return offsets in xyz order; use `False` for NumPy axis order.
        Default: `True`.
    ignore_ltm : bool, optional
        Skip LTM validation. If `False`, require an identity matrix;
        this helper does not apply scaling or other transforms.
        Default: `True`.
    intify_offset : bool, optional
        Round the result to integers using nearest-even rounding.
        Default: `False`.

    Returns
    -------
    ndarray
        Target LTV values minus reference LTV values, in the requested order.

    Raises
    ------
    TypeError
        If an input is not an Astropy FITS Header.
    NotImplementedError
        If LTM validation is enabled and the matrix is not the identity.

    Notes
    -----
    Reads LTV/LTM directly from FITS headers; WCS inputs are not accepted.

    Setup: ``h = Header(dict(NAXIS=2, LTV1=-9.5, LTV2=-19,
    LTM1_1=1, LTM2_2=1))``. Header construction is excluded.

    Timing on MBP 14" [2024, macOS 26.6, M4Pro(8P+4E/G20c/N16c/48G)]
    (2026-09-07; CPython 3.13.11, NumPy 2.4.6, Astropy 7.2.0)::

        calc_offset_physical(h)  4.644 +/- 0.066 us
        calc_offset_physical(h, ignore_ltm=False)  13.234 +/- 0.163 us

    Mean +/- std. dev. per call (`timeit`, 7 runs; 50,000/20,000 loops in row order).
    """
    from astropy.io.fits import Header

    do_ref = reference is not None
    if not isinstance(target, Header):
        raise TypeError("target must be an instance of astropy.io.fits.Header.")
    if do_ref:
        if not isinstance(reference, Header):
            raise TypeError("reference must be an instance of astropy.io.fits.Header.")

    if not ignore_ltm:
        _check_ltm(target)
        if do_ref:
            _check_ltm(reference)

    ndim = target["NAXIS"]
    ltvs_obj = []
    for i in range(ndim):
        try:
            ltvs_obj.append(target[f"LTV{i + 1}"])
        except (KeyError, IndexError):
            ltvs_obj.append(0)

    if do_ref:
        ltvs_ref = []
        for i in range(ndim):
            try:
                ltvs_ref.append(reference[f"LTV{i + 1}"])
            except (KeyError, IndexError):
                ltvs_ref.append(0)
        offset = np.array(ltvs_obj) - np.array(ltvs_ref)
    else:
        offset = np.array(ltvs_obj)

    if intify_offset:
        offset = np.around(offset).astype(int)

    if order_xyz:
        return offset  # This is already xyz order!
    else:
        return offset[::-1]
