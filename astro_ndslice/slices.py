import numpy as np

from .list import is_list_like, listify, ndfy

__all__ = [
    "slice_from_string", "slicefy", "bezel2slice",
]


# Directly imported from ccdproc.utils.slices
def slice_from_string(
    string: str,
    fits_convention: bool = False
) -> tuple:
    """Convert a string to a tuple of slices.

    Parameters
    ----------
    string : str
        A string that can be converted to a slice.

    fits_convention : bool, optional
        If True, assume the input string follows the FITS convention for
        indexing: the indexing is one-based (not zero-based) and the first
        axis is that which changes most rapidly as the index increases.

    Returns
    -------
    slice_tuple : tuple of slice objects
        A tuple able to be used to index a numpy.array

    Notes
    -----
    The ``string`` argument can be anything that would work as a valid way to
    slice an array in Numpy. It must be enclosed in matching brackets; all
    spaces are stripped from the string before processing.
    Directly imported from ccdproc.utils.slices

    Examples
    --------
    >>> import numpy as np
    >>> arr1d = np.arange(5)
    >>> a_slice = slice_from_string('[2:5]')
    >>> arr1d[a_slice]
    array([2, 3, 4])
    >>> a_slice = slice_from_string('[ : : -2] ')
    >>> arr1d[a_slice]
    array([4, 2, 0])
    >>> arr2d = np.array([arr1d, arr1d + 5, arr1d + 10])
    >>> arr2d
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    >>> a_slice = slice_from_string('[1:-1, 0:4:2]')
    >>> arr2d[a_slice]
    array([[5, 7]])
    >>> a_slice = slice_from_string('[0:2,0:3]')
    >>> arr2d[a_slice]
    array([[0, 1, 2],
           [5, 6, 7]])
    """
    no_space = string.replace(' ', '')

    if not no_space:
        return ()

    if not (no_space.startswith('[') and no_space.endswith(']')):
        raise ValueError('Slice string must be enclosed in square brackets.')

    no_space = no_space.strip('[]')
    if fits_convention:
        # Special cases first
        # Flip dimension, with step
        no_space = no_space.replace('-*:', '::-')
        # Flip dimension
        no_space = no_space.replace('-*', '::-1')
        # Normal wildcard
        no_space = no_space.replace('*', ':')
    string_slices = no_space.split(',')
    slices = []
    for string_slice in string_slices:
        slice_args = [int(arg) if arg else None
                      for arg in string_slice.split(':')]
        a_slice = slice(*slice_args)
        slices.append(a_slice)

    if fits_convention:
        slices = _defitsify_slice(slices)

    return tuple(slices)


# Directly imported from ccdproc.utils.slices
def _defitsify_slice(slices: list) -> list:
    """
    Convert a FITS-style slice specification into a python slice.
    This means two things:
    + Subtract 1 from starting index because in the FITS
      specification arrays are one-based.
    + Do **not** subtract 1 from the ending index because the python
      convention for a slice is for the last value to be one less than the
      stop value. In other words, this subtraction is already built into
      python.
    + Reverse the order of the slices, because the FITS specification dictates
      that the first axis is the one along which the index varies most rapidly
      (aka FORTRAN order).

    Directly imported from ccdproc.utils.slices
    """

    python_slice = []
    for a_slice in slices[::-1]:
        new_start = a_slice.start - 1 if a_slice.start is not None else None
        if new_start is not None and new_start < 0:
            raise ValueError("Smallest permissible FITS index is 1")
        if a_slice.stop is not None and a_slice.stop < 0:
            raise ValueError("Negative final index not allowed for FITS slice")
        new_slice = slice(new_start, a_slice.stop, a_slice.step)
        if (a_slice.start is not None and a_slice.stop is not None and
                a_slice.start > a_slice.stop):
            # FITS use a positive step index when dimension are inverted
            new_step = -1 if a_slice.step is None else -a_slice.step
            # Special case to prevent -1 as slice stop value
            new_stop = None if a_slice.stop == 1 else a_slice.stop-2
            new_slice = slice(new_start, new_stop, new_step)
        python_slice.append(new_slice)

    return python_slice


# TODO: add `coord` to select whether image/physical. If physical, header is required.
def slicefy(
    rule: str | int | list[int] | list[slice] | None = None,
    ndim: int = 2,
    order_xyz: bool = True
) -> list[slice] | np.ndarray:
    """ Parse the rule by trimsec, bezels, or slices (in this priority).

    Parameters
    ----------
    rule : str, int, list of int, list of slice, None, optional
        It can have several forms::

          * str: The FITS convention section to trim (e.g., IRAF TRIMSEC).
            Example is ``'[1:2,:]'``.
          * [list of] int: The number of pixels to trim from the edge of the
            image (bezel). Example is ``[1, 2]``.
          * [list of] slice: The slice of each axis (`slice(start, stop,
            step)`). Example is ``[slice(1, 2), slice(2, 3)]``.

        If a single int/slice is given, it will be applied to all the axes.

    ndim : int, optional
        The number of dimensions of the image to convert `rule` into slice.
        (i.e., the length of the final output)

    order_xyz : bool, optional
        Whether the order of rule is in xyz order. Works only if the `rule` is
        bezel-like (int or list of int). If it is slice-like, `rule` must be in
        the pythonic order (i.e., ``[slice_for_axis0, slice_for_axis1, ...]``).

    >>> np.eye(5)[slicefy('[1:2,:]')]
    # array([[1., 0.],
    #       [0., 1.],
    #       [0., 0.],
    #       [0., 0.],
    #       [0., 0.]])
    >>> np.eye(5)[slicefy(1)]  # bezel by 1 pix
    # array([[1., 0., 0.],
    #    [0., 1., 0.],
    #    [0., 0., 1.]])
    >>> np.eye(5)[slicefy((1, 2))]  # bezel by (1, 1), (2, 2) pix (x/y dir)
    # array([[0., 1., 0.]])
    >>> np.eye(5)[slicefy(slice(1, -1, 2))]  # data[1:-1:2, 1:-1:2]
    # array([[1., 0.],
    #    [0., 1.]])
    """
    if rule is None:
        return tuple([slice(None, None, None) for _ in range(ndim)])
    elif isinstance(rule, str):
        fs = np.atleast_1d(rule)
        sl = [slice_from_string(sect, fits_convention=True) for sect in fs]
        return sl[0] if len(sl) == 1 else tuple(sl)
    elif is_list_like(rule):
        if isinstance(rule[0], slice):
            return ndfy(rule, ndim)
        else:  # bezels
            bezels = ndfy([ndfy(b, 2, default=0) for b in listify(rule)], ndim)
            return bezel2slice(bezels, order_xyz=order_xyz)
    elif isinstance(rule, slice):
        return ndfy(rule, ndim)
    elif isinstance(rule, int):  # bezels
        bezels = ndfy([ndfy(b, 2, default=0) for b in listify(rule)], ndim)
        return bezel2slice(bezels, order_xyz=order_xyz)
    else:
        raise TypeError(f"rule must be a string or a list of int/slice. Now {type(rule)=}")


def bezel2slice(
    bezels: list[list[int]],
    order_xyz: bool = True
) -> tuple[slice]:
    """ Convert bezels to slice objects

    Parameters
    ----------
    bezels : list of list of int, optional.
        Must be a list of list of int. Each list of int is in the
        form of ``[lower, upper]``, i.e., the first ``lower`` and last
        ``upper`` rows/columns are ignored.

    order_xyz : bool, optional.
        Whether `bezel` in xyz order or not (python order:
        ``xyz_order[::-1]``). Due to its confusing behavior, it is intended to
        be `True` most of the time.
        Default: `True`.

    Notes
    -----
    Consider a 100x100 image.
    1. ``bezels = [[10, 20], [30, 40]], order_xyz=True`` will ignore the first
       10 columns, the last 20 columns, the **BOTTOM** 30 rows, and the **TOP**
       40 rows.
    2. ``bezels = [[10, 20], [30, 40]], order_xyz=False`` will ignore the
       **BOTTOM** 10 rows (python index of ``[:10]``), the **TOP** 20 columns
       (python index of ``[-20:]``), the first 30 columns, and the last 40
       columns.
    This confusing behavior is due to the (stupid and/or inconsistent?) way
    our world represents xy-coordinates.
    """
    bezels = np.atleast_2d(bezels)
    bezels = bezels[::-1] if order_xyz else bezels
    return tuple([slice(b[0], None if b[1] == 0 else -b[1]) for b in bezels])
