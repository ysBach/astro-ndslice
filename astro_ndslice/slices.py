from numbers import Integral

from .lists import is_list_like, listify, ndfy

__all__ = [
    "slice_from_string",
    "slice_to_string",
    "slicefy",
    "bezel2slice",
]


# TODO: add `coord` to select whether image/physical. If physical, header is required.
def slicefy(
    rule: str | int | list[int] | list[slice] | None = None,
    ndim: int = 2,
    order_xyz: bool = True,
    fits_convention: bool = True,
) -> tuple:
    """Parse the rule by trimsec, bezels, or slices (in this priority).

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
        The number of dimensions of the image to convert `rule` into slices
        (i.e., the length of the final output).

    order_xyz : bool, optional
        Whether the order of rule is in xyz order. Works only if `rule` is
        bezel-like (int or list of int). If it is slice-like, `rule` must be in
        the pythonic order (i.e., ``[slice_for_axis0, slice_for_axis1, ...]``).

    fits_convention : bool, optional
        Whether `rule` (if str) follows the FITS convention: 1-indexed with
        the first axis varying fastest. Ignored for non-string `rule`.
        Default: `True`.

    Returns
    -------
    tuple of slice
        A tuple of `slice` objects that can be used to index a numpy array.

    Examples
    --------
    >>> import numpy as np
    >>> np.eye(5)[slicefy('[1:2,:]')]
    array([[1., 0.],
           [0., 1.],
           [0., 0.],
           [0., 0.],
           [0., 0.]])
    >>> np.eye(5)[slicefy(1)]
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.]])
    >>> np.eye(5)[slicefy((1, 2))]
    array([[0., 1., 0.]])
    >>> np.eye(5)[slicefy(slice(1, -1, 2))]
    array([[1., 0.],
           [0., 1.]])
    """
    if rule is None:
        return (slice(None, None, None),) * ndim
    elif isinstance(rule, str):
        return slice_from_string(rule, fits_convention=fits_convention)
    elif isinstance(rule, slice):
        return (rule,) * ndim
    elif isinstance(rule, Integral):  # bezel-like
        return bezel2slice(rule, ndim=ndim, order_xyz=order_xyz)
    elif is_list_like(rule):
        if isinstance(rule[0], slice):  # list of slice
            if len(rule) == ndim:
                return tuple(rule)
            if len(rule) == 1:
                return tuple(rule) * ndim
            return tuple(ndfy(rule, ndim))
        return bezel2slice(rule, ndim=ndim, order_xyz=order_xyz)
    else:
        raise TypeError(
            f"`rule` must be a str or a list of int/slice. Now {type(rule)=}"
        )


# Directly imported from ccdproc.utils.slices
def slice_from_string(string: str, fits_convention: bool = False) -> tuple:
    """Convert a string to a tuple of slices.

    Parameters
    ----------
    string : str
        A string that can be converted to a slice.

    fits_convention : bool, optional
        If True, assume the input string follows the FITS convention for
        indexing: the indexing is one-based (not zero-based) and the first
        axis is that which changes most rapidly as the index increases.

        .. note::
            `slice_from_string` is almost always used with
            ``fits_convention=True``.

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
    no_space = string.replace(" ", "")

    if not no_space:
        return ()

    if not (no_space.startswith("[") and no_space.endswith("]")):
        raise ValueError("Slice string must be enclosed in square brackets.")

    no_space = no_space.strip("[]")
    if fits_convention:
        # Special cases first
        # Flip dimension, with step
        no_space = no_space.replace("-*:", "::-")
        # Flip dimension
        no_space = no_space.replace("-*", "::-1")
        # Normal wildcard
        no_space = no_space.replace("*", ":")
    string_slices = no_space.split(",")
    slices = []
    for string_slice in string_slices:
        slice_args = [int(arg) if arg else None for arg in string_slice.split(":")]
        a_slice = slice(*slice_args)
        slices.append(a_slice)

    if fits_convention:
        slices = _defitsify_slice(slices)

    return tuple(slices)


# Directly imported from ccdproc.utils.slices
def _defitsify_slice(slices: list) -> list:
    """Convert a FITS-style slice specification into a python slice.

    Subtracts 1 from starting index (FITS is 1-based) and reverses slice
    order (FITS first axis varies fastest, i.e., FORTRAN order).
    Directly imported from ccdproc.utils.slices.

    Parameters
    ----------
    slices : list of slice
        FITS-style slice objects to convert.

    Returns
    -------
    list of slice
        Python-style slice objects.
    """

    python_slice = []
    for a_slice in slices[::-1]:
        new_start = a_slice.start - 1 if a_slice.start is not None else None
        if new_start is not None and new_start < 0:
            raise ValueError("Smallest permissible FITS index is 1")
        if a_slice.stop is not None and a_slice.stop < 0:
            raise ValueError("Negative final index not allowed for FITS slice")
        new_slice = slice(new_start, a_slice.stop, a_slice.step)
        if (
            a_slice.start is not None
            and a_slice.stop is not None
            and a_slice.start > a_slice.stop
        ):
            # FITS use a positive step index when dimension are inverted
            new_step = -1 if a_slice.step is None else -a_slice.step
            # Special case to prevent -1 as slice stop value
            new_stop = None if a_slice.stop == 1 else a_slice.stop - 2
            new_slice = slice(new_start, new_stop, new_step)
        python_slice.append(new_slice)

    return python_slice


def _fitsify_slice(slices: list) -> list:
    """Convert python slices to FITS-style slice specification.

    The inverse of ``_defitsify_slice``: adds 1 to starting index and
    reverses slice order (Python zyx → FITS xyz order).

    Parameters
    ----------
    slices : list of slice
        Python-style slice objects to convert.

    Returns
    -------
    list of slice
        FITS-style slice objects.
    """
    fits_slice = []
    for a_slice in slices[::-1]:  # reverse to FITS order (zyx → xyz)
        if a_slice.step is not None and a_slice.step < 0:
            # inverted: came from a FITS slice where start > stop
            new_start = a_slice.start + 1 if a_slice.start is not None else None
            new_stop = 1 if a_slice.stop is None else a_slice.stop + 2
            new_step = -a_slice.step if a_slice.step != -1 else None
        else:
            new_start = a_slice.start + 1 if a_slice.start is not None else None
            new_stop = a_slice.stop
            new_step = a_slice.step
        if new_start is not None and new_start <= 0:
            raise ValueError(
                f"FITS index must be >= 1; Python start={a_slice.start} maps to 0."
            )
        fits_slice.append(slice(new_start, new_stop, new_step))
    return fits_slice


def slice_to_string(slices: tuple | list, fits_convention: bool = True) -> str:
    """Convert a tuple of slices to a string representation.

    The inverse of ``slice_from_string``.

    Parameters
    ----------
    slices : tuple or list of slice
        `slice` objects to convert. Must be in pythonic (zyx) order when
        `fits_convention` is `True`.

    fits_convention : bool, optional
        If `True`, convert to FITS convention: 1-indexed, first axis varies
        fastest (xyz order in output string), end index included.
        Default: `True`.

    Returns
    -------
    str
        String representation enclosed in square brackets.

    Examples
    --------
    >>> slice_to_string((slice(0, 5), slice(1, 3)))
    '[2:3,1:5]'
    >>> slice_to_string((slice(1, 3), slice(0, 4, 2)), fits_convention=False)
    '[1:3,0:4:2]'
    """
    _slices = _fitsify_slice(list(slices)) if fits_convention else list(slices)
    parts = []
    for s in _slices:
        start = "" if s.start is None else str(s.start)
        stop = "" if s.stop is None else str(s.stop)
        step = "" if s.step is None else str(s.step)
        parts.append(f"{start}:{stop}" if not step else f"{start}:{stop}:{step}")
    return "[" + ",".join(parts) + "]"


def bezel2slice(
    rule: int | list[int] | None = None, ndim: int = 2, order_xyz: bool = True
) -> tuple[slice, ...]:
    """Convert non-slice rule to slice objects.

    Parameters
    ----------
    rule : int, list of int, None, optional
        The number of pixels to trim from the edge of the image (bezel).
        Example is ``[1, 2]``. If a single int is given, it will be applied
        to all the axes.

    ndim : int, optional
        The number of dimensions of the image to convert `bezels` into slices.

    order_xyz : bool, optional
        Whether `bezel` is in xyz order or not (python order:
        ``xyz_order[::-1]``). Due to its confusing behavior, it is intended to
        be `True` most of the time.
        Default: `True`.

    Returns
    -------
    tuple of slice
        A tuple of `slice` objects for indexing a numpy array.

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

    def _pair_from_bezel(bezel):
        if bezel is None:
            return [0, 0]
        if isinstance(bezel, Integral):
            return [bezel, bezel]
        values = list(bezel) if is_list_like(bezel) else [bezel]
        values = [0 if value is None else value for value in values]
        if len(values) == 2:
            return values
        if len(values) == 1:
            return values * 2
        raise ValueError(
            f"`len(item)` must be 1 or `length`(=2). Now it is {len(values)}."
        )

    if rule is None:
        # Preserve the existing ndfy/listify error for the undocumented None case.
        bezels = ndfy([], ndim)
    elif isinstance(rule, Integral):
        bezels = [[rule, rule]] * ndim
    else:
        bezels = [_pair_from_bezel(bezel) for bezel in listify(rule)]
        if len(bezels) == 1:
            bezels *= ndim
        elif len(bezels) != ndim:
            raise ValueError(
                f"`len(item)` must be 1 or `length`(={ndim}). Now it is {len(bezels)}."
            )

    bezels = bezels[::-1] if order_xyz else bezels
    return tuple(slice(b[0], None if b[1] == 0 else -b[1]) for b in bezels)
