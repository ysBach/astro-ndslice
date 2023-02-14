import numpy as np

__all__ = [
    "regularize_offsets", "offseted_shape", "offsets2slice",
    "calc_offset_wcs", "calc_offset_physical",
]


def regularize_offsets(
    offsets: np.ndarray,
    offset_order_xyz: bool = True,
    intify_offsets: bool = False
) -> np.ndarray:
    """ Makes offsets all non-negative and relative to each other.

    Parameters
    ----------
    offsets : ndarray
        The offsets to be regularized. Must be in the order of python/numpy
        (i.e., z, y, x order). First, converted to 2d array, then, the offsets
        are made relative to each other (i.e., the minimum offset is set to 0
        by ``_offsets - np.min(_offsets, axis=0)``).

    offset_order_xyz : bool, optional
        Whether the order of offsets is in xyz order. Default: `True`.

    intify_offsets : bool, optional
        Whether to convert the offsets to integers. Default: `False`.

    """
    _offsets = np.atleast_2d(offsets)
    if offset_order_xyz:
        _offsets = np.flip(_offsets, -1)

    # _offsets = np.max(_offsets, axis=0) - _offsets
    _offsets = _offsets - np.min(_offsets, axis=0)
    # This is the convention to follow IRAF (i.e., all of offsets > 0.)
    if intify_offsets:
        _offsets = np.around(_offsets).astype(int)

    return _offsets


def offseted_shape(
    shapes: np.ndarray,
    offsets: np.ndarray,
    method: str = 'outer',
    offset_order_xyz: bool = True,
    intify_offsets: bool = False,
    pythonize_offsets: bool = True
) -> tuple(np.ndarray, tuple):
    '''shapes and offsets must be in the order of python/numpy (i.e., z, y, x order).

    Paramters
    ---------
    shapes : ndarray
        The shapes of the arrays to be processed. It must have the shape of
        ``nimage`` by ``ndim``. The order of shape must be pythonic (i.e.,
        ``shapes[i] = image[i].shape``, not in the xyz order).

    offsets : ndarray
        The offsets must be ``(cen_i - cen_ref) + const`` format, i.e., an
        offset is the position of the target frame (position can be, e.g.,
        origin or center) in the coordinate of the reference frame, with a
        possible non-zero constant offset applied. It must have the shape of
        ``nimage`` by ``ndim``.

    method : str, optional
        The method to calculate the `shape_out`::

          * ``'outer'``: To combine images, so every pixel in `shape_out` has
            at least 1 image pixel.
          * ``'inner'``: To process only where all the images have certain
            pixel (fully-overlap).

    offset_order_xyz : bool, optional
        Whether `offsets` are in xyz order. If so, those will be flipped to
        pythonic order. Default: `True`

    Returns
    -------
    _offsets : ndarray
        The *relative* offsets calculated such that at least one image per each
        dimension must have offset of 0.

    shape_out : tuple
        The shape of the array depending on the `method`.
    '''
    _offsets = regularize_offsets(
        offsets,
        offset_order_xyz=offset_order_xyz,
        intify_offsets=intify_offsets
    )

    if method == 'outer':
        shape_out = np.around(np.max(np.array(shapes) + _offsets, axis=0)).astype(int)
        # print(_offsets, shapes, shape_out)
    # elif method == 'stack':
    #     shape_out_comb = np.around(
    #         np.max(np.array(shapes) + _offsets, axis=0)
    #     ).astype(int)
    #     shape_out = (len(shapes), *shape_out_comb)
    elif method == 'inner':
        lower_bound = np.max(_offsets, axis=0)
        upper_bound = np.min(_offsets + shapes, axis=0)
        npix = upper_bound - lower_bound
        shape_out = np.around(npix).astype(int)
        if np.any(npix < 0):
            raise ValueError("There doesn't exist fully-overlapping pixel! "
                             + f"Naïve output shape={shape_out}.")
        # print(lower_bound, upper_bound, shape_out)
    else:
        raise ValueError("method unacceptable (use one of 'inner', 'outer').")

    if offset_order_xyz and not pythonize_offsets:
        # reverse _offsets to original xyz order
        _offsets = np.flip(_offsets, -1)

    return _offsets, tuple(shape_out)


def offsets2slice(
    shapes: np.ndarray,
    offsets: np.ndarray,
    method: str = 'outer',
    shape_order_xyz: bool = False,
    offset_order_xyz: bool = True,
    outer_for_stack: bool = True,
    fits_convention: bool = False
) -> list:
    """ Calculates the slices for each image to extract overlapping parts.

    Parameters
    ----------
    shapes, offsets : ndarray
        The shape and offset of each image. If multiple images are used, it
        must have shape of ``nimage`` by ``ndim``.

    method : str, optional
        The method to calculate the `shape_out`::

          * ``'outer'``: To combine images, so every pixel in `shape_out` has
            at least 1 image pixel.
          * ``'inner'``: To process only where all the images have certain
            pixel (fully-overlap).

    shape_order_xyz, offset_order_xyz : bool, optional.
        Whether the order of the shapes or offsets are in xyz or pythonic.
        Shapes are usually in pythonic as it is obtained by
        ``image_data.shape``, but offsets are often in xyz order (e.g., if
        header ``LTVi`` keywords are loaded in their alpha-numeric order; or
        you have used `~calc_offset_wcs` or `~calc_offset_physical` with
        default ``order_xyz=True``). Default is `False` and `True`,
        respectively.

    outer_for_stack : bool, optional.
        If `True`(default), the output slice is the slice in tne ``N+1``-D
        array, which will be constructed before combining them along
        ``axis=0``. That is, ``comb = np.nan*np.ones(_offseted_shape(shapes,
        offsets, method='outer'))`` and ``comb[slices[i]] = images[i]``. Then a
        median combine, for example, is done by ``np.nanmedian(comb, axis=0)``.
        If ``stack_outer=False``, ``slices[i]`` will be
        ``slices_with_stack_outer_True[i][1:]``.

    fits_convention : bool, optional.
        Whether to return the slices in FITS convention (xyz order, 1-indexing,
        end index included). If `True` (default), returned list contains str;
        otherwise, slice objects will be contained.

    Returns
    -------
    slices : list of str or list of slice
        The meaning of it differs depending on `method`::

          * ``'outer'``: the slice of the **output** array where the i-th image
            should fit in.
          * ``'inner'``: the slice of the **input** array (image) where the
            overlapping region resides.

    Example
    -------
    >>>
    """
    _shapes = np.atleast_2d(shapes)
    if shape_order_xyz:
        _shapes = np.flip(_shapes, -1)

    _offsets = regularize_offsets(
        offsets,
        offset_order_xyz=offset_order_xyz,
        intify_offsets=True
    )

    if _shapes.ndim != 2 or _offsets.ndim != 2:
        raise ValueError("Shapes and offsets must be at most 2-D.")

    if _shapes.shape != _offsets.shape:
        raise ValueError("shapes and offsets must have the identical shape.")

    if method == 'outer':
        starts = _offsets
        stops = _offsets + _shapes
        if outer_for_stack:
            def _initial_tmp(i):
                return [f"{i + 1}:{i + 1}"] if fits_convention else [slice(i, i + 1, None)]
        else:
            _initial_tmp = lambda i: []  # initialized empty list regardless of argument
    elif method == 'inner':
        offmax = np.max(_offsets, axis=0)
        if np.any(np.min(_shapes + _offsets, axis=0) <= offmax):
            raise ValueError(
                "At least 1 frame has no overlapping pixel with all others. "
                + "Check if there's any overlapping pixel for images for the given offsets."
            )

        # 1-D array +/- 2-D array:
        #   the former 1-D array is broadcast s.t. it is "tile"d along axis=-1.
        starts = offmax - _offsets
        stops = np.min(_offsets + _shapes, axis=0) - _offsets
        _initial_tmp = lambda i: []  # initialized empty list regardless of argument
    else:
        raise ValueError("method unacceptable (use one of 'inner', 'outer').")

    slices = []
    for image_i, (start, stop) in enumerate(zip(starts, stops)):
        # NOTE: starts/stops are all in pythonic index
        tmp = _initial_tmp(image_i)
        # print(tmp)
        for start_i, stop_i in zip(start, stop):  # i = coordinate, (z y x) order
            if fits_convention:
                tmp.append(f"{start_i + 1:d}:{stop_i:d}")
            else:
                tmp.append(slice(start_i, stop_i, None))
            # print(tmp)

        if fits_convention:
            slices.append('[' + ','.join(tmp[::-1]) + ']')  # order is opposite!
        else:
            slices.append(tmp)

    return slices


def calc_offset_wcs(
        target,
        reference,
        loc_target: str = "center",
        loc_reference: str = "center",
        order_xyz: bool = True,
        intify_offset: bool = False
) -> np.ndarray:
    """ The pixel offset of target's location when using WCS in referene.

    Parameters
    ----------
    target : WCS
        The WCS object to calculate the position (see `loc_target`)

    reference : WCS
        The reference WCS to calculate the position *from*.

    loc_target, loc_reference : {"center", "origin"} or ndarray, optional.
        The location to calculate the position (in pixels and in xyz order)::

         * ``'center'``: The center of the image (half of ``NAXISi`` keys).
         * ``'origin'``: The origin of the image (``0``).
         * ndarray: The location in the image coordinate (same x, y position of
           two images).

        Default is ``'center'`` (half of ``NAXISi`` keys in `target`).

    order_xyz : bool, optional.
        Whether to return the position in xyz order or not (python order:
        ``[::-1]`` of the former). Default is `True`.

    intify_offset : bool, optional.
        Whether to convert the offset to integer or not. Default is `False`.
    """
    from astropy.wcs import WCS

    def _parse_loc(loc, obj):
        if isinstance(obj, WCS):
            w = obj
        else:
            raise TypeError("input must be an instance of astropy.wcs.WCS.")

        if loc == "center":
            _loc = np.atleast_1d(w._naxis)/2
        elif loc == "origin":
            _loc = [0.]*w.naxis
        else:
            _loc = np.atleast_1d(loc)

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


def calc_offset_physical(
        target,
        reference=None,
        order_xyz: bool=True,
        ignore_nondiag_ltm: bool=True,
        intify_offset: bool=False
) -> np.ndarray:
    """ The pixel offset by physical-coordinate information in referene.

    Parameters
    ----------
    target : Header
        The object to extract header to calculate the position

    reference : Header
        The reference to extract header to calculate the position *from*. If
        `None`, it is basically identical to extract the LTV values from
        `target`.
        Default is `None`.

    order_xyz : bool, optional.
        Whether to return the position in xyz order or not (python order:
        ``[::-1]`` of the former).
        Default is `True`.

    ignore_nondiag_ltm : bool, optional.
        Whether to assuem the LTM matrix is diagonal. If it is not and
        ``ignore_nondiag_ltm=False``, a `NotImplementedError` will be raised,
        i.e., non-diagonal LTM matrices are not supported.

    Notes
    -----
    Similar to `calc_offset_wcs`, but with locations fixed to origin (as
    non-identity LTM matrix is not supported). Also, input of WCS is not
    accepted because astropy's wcs module does not parse LTV/LTM from header.
    """
    from astropy.io.fits import Header

    def _check_ltm(hdr):
        ndim = hdr["NAXIS"]
        for i in range(ndim):
            for j in range(ndim):
                try:
                    assert float(hdr["LTM{i}_{j}"]) == 1.0*(i == j)
                except (KeyError, IndexError):
                    continue
                except (AssertionError):
                    raise NotImplementedError("Non-diagonal LTM matrix is not supported.")

            try:  # Sometimes LTM matrix is saved as ``LTMi``, not ``LTMi_j``.
                assert float(target["LTM{i}"]) == 1.0
            except (KeyError, IndexError):
                continue
            except (AssertionError):
                raise NotImplementedError("Non-diagonal LTM matrix is not supported.")

    do_ref = reference is not None
    if not isinstance(target, Header):
        raise TypeError("target must be an instance of astropy.io.fits.Header.")
    if do_ref:
        if not isinstance(reference, Header):
            raise TypeError("reference must be an instance of astropy.io.fits.Header.")

    if not ignore_nondiag_ltm:
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
