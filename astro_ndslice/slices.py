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
    """Build NumPy slices from a section string, edge widths, or slices.

    Parameters
    ----------
    rule : str, int, slice, iterable, or None, optional
        Section string, edge widths, or Python slices. One width or slice
        applies to every axis; iterables supply one item or one per axis.
        `None` selects all pixels.
    ndim : int, optional
        Number of axes for widths and slices. Strings determine their own
        number of axes. Default: 2.
    order_xyz : bool, optional
        Read edge widths in xyz order. Slices use NumPy order; strings
        follow `fits_convention`. Default: `True`.
    fits_convention : bool, optional
        Parse strings with 1-based, inclusive bounds in xyz order.
        Use `False` for Python-style strings. Default: `True`.

    Returns
    -------
    tuple of slice
        Index suitable for ``data[slicefy(rule)]``.

    Notes
    -----
    Timing on MBP 14" [2024, macOS 26.6, M4Pro(8P+4E/G20c/N16c/48G)]
    (2026-09-07; CPython 3.13.11, NumPy 2.4.6)::

        slicefy("[2:5,3:7]")  0.924 +/- 0.007 us
        slicefy(1)  0.680 +/- 0.009 us
        slicefy(slice(0, 5, 2))  0.076 +/- 0.001 us

    Mean +/- std. dev. per call (`timeit`, 7 runs;
    500,000/500,000/5,000,000 loops in row order).

    Examples
    --------
    >>> slicefy("[2:5,3:7]")
    (slice(2, 7, None), slice(1, 5, None))
    >>> slicefy(1)
    (slice(1, -1, None), slice(1, -1, None))
    >>> slicefy(slice(0, 5, 2))
    (slice(0, 5, 2), slice(0, 5, 2))
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
        # Materialize iterators once so generators follow the same API as
        # lists and the first-element dispatch below is safe.
        rule = list(rule)
        if not rule:
            raise ValueError("`rule` must contain at least one item.")
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
    """Parse a section string into NumPy slices.

    Parameters
    ----------
    string : str
        Bracketed, comma-separated ``start:stop[:step]`` sections.
        Omitted bounds and spaces are allowed; ``""`` or ``"[]"`` returns ``()``.
    fits_convention : bool, optional
        Use 1-based, inclusive bounds in xyz order. FITS wildcards ``*``
        and ``-*`` select an axis forward and backward. Otherwise, use
        Python bounds and NumPy axis order. Default: `False`.

    Returns
    -------
    tuple of slice
        Index suitable for a NumPy array.

    Raises
    ------
    ValueError
        For invalid brackets, numbers, zero steps, or FITS endpoint/direction
        conflicts.

    Notes
    -----
    Adapted from ``ccdproc.utils.slices``. This parser handles slices,
    not arbitrary NumPy indexing expressions.

    Timing on MBP 14" [2024, macOS 26.6, M4Pro(8P+4E/G20c/N16c/48G)]
    (2026-09-07; CPython 3.13.11, NumPy 2.4.6)::

        slice_from_string("[2:5,3:7]")  0.462 +/- 0.005 us
        slice_from_string("[2:5,3:7]", fits_convention=True)  0.957 +/- 0.051 us

    Mean +/- std. dev. per call (`timeit`, 7 runs; 500,000/200,000 loops in row order).

    Examples
    --------
    >>> import numpy as np
    >>> np.arange(5)[slice_from_string("[2:5]")]
    array([2, 3, 4])
    >>> np.arange(5)[slice_from_string("[::-2]")]
    array([4, 2, 0])
    >>> slice_from_string("[2:5,3:7]", fits_convention=True)
    (slice(2, 7, None), slice(1, 5, None))
    """
    no_space = string.replace(" ", "")

    if not no_space:
        return ()

    if not (no_space.startswith("[") and no_space.endswith("]")):
        raise ValueError("Slice string must be enclosed in square brackets.")

    no_space = no_space.strip("[]")
    if not no_space:
        return ()
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
        if a_slice.step == 0:
            raise ValueError("Slice step cannot be zero.")
        slices.append(a_slice)

    if fits_convention:
        slices = _defitsify_slice(slices)

    return tuple(slices)


# Directly imported from ccdproc.utils.slices
def _defitsify_slice(slices: list) -> list:
    """Convert FITS slices to NumPy axis order and bounds.

    Parameters
    ----------
    slices : list of slice
        FITS slices with 1-based, inclusive endpoints in xyz order.

    Returns
    -------
    list of slice
        Python slices in reversed axis order, preserving direction.

    Notes
    -----
    Adapted from ``ccdproc.utils.slices``.
    """

    python_slice = []
    for a_slice in slices[::-1]:
        if a_slice.step == 0:
            raise ValueError("Slice step cannot be zero.")
        if a_slice.start is not None and a_slice.start < 1:
            raise ValueError("Smallest permissible FITS index is 1")
        if a_slice.stop is not None and a_slice.stop < 1:
            raise ValueError("Smallest permissible FITS index is 1")

        reverse = (a_slice.step is not None and a_slice.step < 0) or (
            a_slice.start is not None
            and a_slice.stop is not None
            and a_slice.start > a_slice.stop
        )
        if (
            a_slice.step is not None
            and a_slice.step < 0
            and a_slice.start is not None
            and a_slice.stop is not None
            and a_slice.start < a_slice.stop
        ):
            raise ValueError(
                "FITS slice endpoints are incompatible with a negative step."
            )

        new_start = a_slice.start - 1 if a_slice.start is not None else None
        if reverse:
            # FITS uses positive strides for descending endpoints.
            new_step = -1 if a_slice.step is None else -abs(a_slice.step)
            new_stop = (
                None if a_slice.stop is None or a_slice.stop == 1 else a_slice.stop - 2
            )
            new_slice = slice(new_start, new_stop, new_step)
        else:
            new_slice = slice(new_start, a_slice.stop, a_slice.step)
        python_slice.append(new_slice)

    return python_slice


def _fitsify_slice(slices: list) -> list:
    """Convert Python slices to FITS axis order and bounds.

    Parameters
    ----------
    slices : list of slice
        Python slices in NumPy axis order.

    Returns
    -------
    list of slice
        FITS slices with 1-based, inclusive endpoints in xyz order.

    Raises
    ------
    ValueError
        If a zero step, negative bound, or intrinsically empty slice prevents
        faithful conversion without an array shape.
    """
    fits_slice = []
    for a_slice in slices[::-1]:  # reverse to FITS order (zyx → xyz)
        if a_slice.step == 0:
            raise ValueError("Slice step cannot be zero.")
        if a_slice.start is not None and a_slice.start < 0:
            raise ValueError(
                "Negative Python slice starts cannot be represented without shape."
            )
        if a_slice.stop is not None and a_slice.stop < 0:
            raise ValueError(
                "Negative Python slice stops cannot be represented without shape."
            )

        step = 1 if a_slice.step is None else a_slice.step
        if step > 0:
            effective_start = 0 if a_slice.start is None else a_slice.start
            if a_slice.stop is not None and effective_start >= a_slice.stop:
                raise ValueError(
                    "Finite empty Python slices cannot be represented in FITS."
                )
            new_start = a_slice.start + 1 if a_slice.start is not None else None
            new_stop = a_slice.stop
            new_step = a_slice.step
        else:
            if (
                a_slice.start is not None
                and a_slice.stop is not None
                and a_slice.start <= a_slice.stop
            ):
                raise ValueError(
                    "Finite empty Python slices cannot be represented in FITS."
                )
            new_start = a_slice.start + 1 if a_slice.start is not None else None
            new_stop = 1 if a_slice.stop is None else a_slice.stop + 2
            # Bounded reverse slices can use canonical positive FITS strides;
            # open-start reversals must retain a negative stride.
            new_step = (
                a_slice.step
                if a_slice.start is None
                else (None if a_slice.step == -1 else -a_slice.step)
            )
        fits_slice.append(slice(new_start, new_stop, new_step))
    return fits_slice


def slice_to_string(slices: tuple | list, fits_convention: bool = True) -> str:
    """Serialize Python slices as a FITS or Python section string.

    Parameters
    ----------
    slices : tuple or list of slice
        Slices in NumPy axis order.
    fits_convention : bool, optional
        Return 1-based, inclusive bounds in xyz order. Use `False` to keep
        Python bounds and axis order. Default: `True`.

    Returns
    -------
    str
        Comma-separated sections enclosed in square brackets.

    Raises
    ------
    ValueError
        If FITS conversion encounters a zero step, negative Python bound, or
        intrinsically empty slice. Keep such slices in Python form.

    Notes
    -----
    Setup: ``sl = (slice(0, 5), slice(1, 3))``.

    Timing on MBP 14" [2024, macOS 26.6, M4Pro(8P+4E/G20c/N16c/48G)]
    (2026-09-07; CPython 3.13.11, NumPy 2.4.6)::

        slice_to_string(sl)  0.683 +/- 0.063 us
        slice_to_string(sl, fits_convention=False)  0.320 +/- 0.006 us

    Mean +/- std. dev. per call (`timeit`, 7 runs;
    500,000/1,000,000 loops in row order).

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
    """Convert edge-trim widths to NumPy slices.

    Parameters
    ----------
    rule : int, iterable, or None, optional
        Widths in pixels: one integer for all edges, or one item per axis.
        Each item is a symmetric width or ``(low, high)`` pair; a single item
        repeats across axes. `None` means no trim; zero/`None` preserves an edge.
    ndim : int, optional
        Number of axes. Default: 2.
    order_xyz : bool, optional
        Read widths in xyz order, then reverse to NumPy axis order.
        Use `False` for row, column order in 2D. Default: `True`.

    Returns
    -------
    tuple of slice
        Index that removes the requested edges. A zero high-edge width
        leaves the stop open.

    Notes
    -----
    Timing on MBP 14" [2024, macOS 26.6, M4Pro(8P+4E/G20c/N16c/48G)]
    (2026-09-07; CPython 3.13.11, NumPy 2.4.6)::

        bezel2slice(1)  0.499 +/- 0.003 us
        bezel2slice([[1, 2], [3, 4]])  1.784 +/- 0.047 us

    Mean +/- std. dev. per call (`timeit`, 7 runs; 500,000/200,000 loops in row order).

    Examples
    --------
    >>> bezel2slice([[1, 2], [3, 4]])
    (slice(3, -4, None), slice(1, -2, None))
    >>> bezel2slice([[1, 2], [3, 4]], order_xyz=False)
    (slice(1, -2, None), slice(3, -4, None))
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
        return (slice(None, None, None),) * ndim
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
