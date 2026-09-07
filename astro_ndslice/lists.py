"""Normalize scalars and iterables to lists."""

from collections import abc
from collections.abc import Callable
from typing import Any, Optional

import numpy as np

__all__ = [
    "is_list_like",
    "listify",
    "ndfy",
]


def is_list_like(*objs, allow_sets: bool = True, func: Callable = all) -> bool:
    """Check whether inputs are iterable, excluding strings and scalar arrays.

    Parameters
    ----------
    *objs : object
        Objects to check.
    allow_sets : bool, optional
        Count sets as list-like. Default: `True`.
    func : callable, optional
        Combine results with `all` (default), `any`, or another callable.

    Returns
    -------
    bool or object
        Result from `func`; `all` and `any` return a boolean. Strings, bytes,
        and zero-dimensional arrays are not list-like. Other iterables,
        including dictionaries, are.

    Notes
    -----
    Adapted from pandas to accept multiple inputs and a combining function.
    https://github.com/pandas-dev/pandas/blob/bdb00f2d5a12f813e93bc55cdcd56dcb1aae776e/pandas/_libs/lib.pyx#L1026

    Timing on MBP 14" [2024, macOS 26.6, M4Pro(8P+4E/G20c/N16c/48G)]
    (2026-09-07; CPython 3.13.11, NumPy 2.5.2)::

        is_list_like("asdfaer.fits")  0.188 +/- 0.001 us

    Mean +/- std. dev. per call (`timeit`, 7 runs, 2,000,000 loops each).
    """

    def _is_list_like(obj):
        return (
            isinstance(obj, abc.Iterable)
            # we do not count strings/unicode/bytes as list-like
            and not isinstance(obj, (str, bytes))
            # exclude zero-dimensional numpy arrays, effectively scalars
            and not (isinstance(obj, np.ndarray) and obj.ndim == 0)
            # exclude sets if allow_sets is False
            and not (allow_sets is False and isinstance(obj, abc.Set))
        )

    if func is all:
        for obj in objs:
            if not _is_list_like(obj):
                return False
        return True
    if func is any:
        for obj in objs:
            if _is_list_like(obj):
                return True
        return False

    return func(_is_list_like(obj) for obj in objs)


def listify(*objs: Any, scalar2list: bool = True, none2list: bool = False) -> Any:
    """Convert inputs to lists of the same length.

    Parameters
    ----------
    *objs : object
        One or more inputs. Iterables become lists; strings count as scalars.
        With multiple inputs, length-one lists and scalars are repeated to
        match the longest input.
    scalar2list : bool, optional
        Wrap a single scalar in a list. Ignored for multiple inputs and
        iterable inputs. Default: `True`.
    none2list : bool, optional
        Return ``[None]`` for a single `None`, instead of ``[]``.
        With multiple inputs, `None` always becomes a length-one list before
        broadcasting. Default: `False`.

    Returns
    -------
    list or object
        Normalized lists, or the original single scalar when `scalar2list`
        is `False`.

    Raises
    ------
    ValueError
        If no inputs are given or input lengths cannot be broadcast.

    Notes
    -----
    Timing on MBP 14" [2024, macOS 26.6, M4Pro(8P+4E/G20c/N16c/48G)]
    (2026-09-07; CPython 3.13.11, NumPy 2.5.2)::

        listify([12])                         0.331 +/- 0.003 us
        listify("asdf")                       0.249 +/- 0.001 us
        listify("asdf", scalar2list=False)     0.238 +/- 0.001 us

    Mean +/- std. dev. per call (`timeit`, 7 runs, 1,000,000 loops each).

    Examples
    --------
    >>> listify([1, 2], "x", None)
    [[1, 2], ['x', 'x'], [None, None]]
    >>> listify(3, scalar2list=False)
    3
    """

    if len(objs) == 1:
        obj = objs[0]
        if obj is None:
            return [None] if none2list else []
        if is_list_like(obj):
            return list(obj)
        return [obj] if scalar2list else obj

    objlists = [list(obj) if is_list_like(obj) else [obj] for obj in objs]
    lengths = [len(obj) for obj in objlists]
    length = max(lengths)
    for objl in objlists:
        if len(objl) not in [1, length]:
            raise ValueError(f"Each input must be 1 or max(lengths)={length}.")

    return [obj * length if len(obj) == 1 else obj for obj in objlists]


def ndfy(item, length: Optional[int] = None, default: Any = None) -> list:
    """Normalize a list, replace `None`, and repeat a single item as needed.

    Parameters
    ----------
    item : object
        Scalar or iterable to normalize.
    length : int, optional
        Target length; a single item repeats, other lengths must match.
        `None` keeps the input length, or one for a scalar.
    default : object, optional
        Replacement for a `None` input or top-level element. Default: `None`.

    Returns
    -------
    list
        Normalized values. Nested inputs are preserved, not transposed.

    Raises
    ------
    ValueError
        If the input length is neither one nor the requested length.

    Notes
    -----
    Repetition reuses references; nested mutable values are not copied.

    Timing on MBP 14" [2024, macOS 26.6, M4Pro(8P+4E/G20c/N16c/48G)]
    (2026-09-07; CPython 3.13.11, NumPy 2.4.6)::

        ndfy(1, length=2)  0.356 +/- 0.004 us
        ndfy([1, None], default=0)  0.426 +/- 0.010 us

    Mean +/- std. dev. per call (`timeit`, 7 runs;
    1,000,000/500,000 loops in row order).

    Examples
    --------
    >>> ndfy(None, length=2, default=0)
    [0, 0]
    >>> ndfy([[1, 2]], length=2)
    [[1, 2], [1, 2]]
    """
    item = [default if i is None else i for i in listify(item, none2list=True)]
    item_length = len(item)

    if (length is None) or (item_length == length):
        return item
    elif item_length != 1:
        raise ValueError(
            f"`len(item)` must be 1 or `length`(={length}). Now it is {item_length}."
        )

    # Now, item_length == 1
    return item * length
