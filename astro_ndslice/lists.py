"""
Simple tools to make lists
"""

from collections import abc
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = [
    "is_list_like",
    "listify",
    "ndfy",
]


def is_list_like(*objs, allow_sets: bool = True, func: Callable = all) -> bool:
    """Check if inputs are list-like

    Parameters
    ----------
    *objs : object
        Objects to check.
    allow_sets : bool, optional
        If this parameter is `False`, sets will not be considered list-like.
        Default: `True`
    func : functional object, optional
        The function to be applied to each element. Useful ones are `all` and
        `any`.
        Default: `all`

    Notes
    -----
    Direct copy from pandas, with slight modification to accept *args and
    all/any, etc, functionality by `func`.
    https://github.com/pandas-dev/pandas/blob/bdb00f2d5a12f813e93bc55cdcd56dcb1aae776e/pandas/_libs/lib.pyx#L1026

    Note that pd.DataFrame also returns True.

    Timing on MBP 14" [2024, macOS 26.5, M4Pro(8P+4E/G20c/N16c/48G)]
    %timeit yfu.is_list_like("asdfaer.fits")
    0.182 µs +- 0.005 µs per loop
    (mean +- std. dev. of 7 runs, 100000 loops each)
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


def listify(*objs, scalar2list: bool = True, none2list: bool = False) -> list:
    """Make multiple objects into list of same length.

    Parameters
    ----------
    objs : None, str, list-like
        If single object, it will be converted to a list ``[obj]`` or ``obj``,
        depending on `scalar2list`. Any scalar input will be converted to a
        list of a target length (largest length among `objs`). If `None`, an
        empty list (`[]`) or ``[None]`` is returned depending on `none2list`.
        If multiple objects are given, maximum length of them is used as the
        target length.

    scalar2list : bool, optional
        If `True`, a single scalar input will be converted to a list of a
        target length. Otherwise, it will be returned as is. Has no effect
        on list-like inputs, which are always converted with ``list(obj)``.

    none2list : bool, optional
        Whether to return an empty list (`[]`). If `True`, ``[None]`` is
        returned if `objs` is `None`.
        Default: `False`

    Notes
    -----
    If any obj of `None` need to be converted to a length>1 list, it will be
    made as [None, None, ...], rather than an empty list, regardless of
    `none2list`.

    Timing on MBP 14" [2024, macOS 26.5, M4Pro(8P+4E/G20c/N16c/48G)]:
    %timeit yfu.listify([12])
    0.355 µs +- 0.006 µs per loop
    (mean +- std. dev. of 7 runs, 100000 loops each)
    %timeit yfu.listify("asdf")
    0.287 µs +- 0.005 µs per loop
    (mean +- std. dev. of 7 runs, 100000 loops each)
    %timeit yfu.listify("asdf", scalar2list=False)
    0.273 µs +- 0.002 µs per loop
    (mean +- std. dev. of 7 runs, 100000 loops each)

    """

    def _listify_single(obj, none2list=True):
        if obj is None:
            return [obj] if none2list else []
        elif is_list_like(obj):
            return list(obj)
        else:
            return [obj] if scalar2list else obj

    if len(objs) == 1:
        obj = objs[0]
        if obj is None:
            return [None] if none2list else []
        if is_list_like(obj):
            return list(obj)
        return [obj] if scalar2list else obj

    objlists = [_listify_single(obj, none2list=True) for obj in objs]
    lengths = [len(obj) for obj in objlists]
    length = max(lengths)
    for objl in objlists:
        if len(objl) not in [1, length]:
            raise ValueError(f"Each input must be 1 or max(lengths)={length}.")

    return [obj * length if len(obj) == 1 else obj for obj in objlists]


def ndfy(item, length: int | None = None, default: Any = None) -> list:
    """Make an item to a list of `length`.

    Parameters
    ----------
    item : None, general object, list-like
        The item to be made into a list. If `None`, it will be filled by
        `default`.

    length : int, optional
        The length of the final list. If `None`, the length of the input is
        used (if `item` is a scalar, a length-1 list is returned).

    default : general object
        The default value to be used if `item` or any element of `item` is
        `None`. Default is `None`

    Notes
    -----
    Useful for the cases when bezels, sigma, ... are needed. For example, if
    ``bezel_nd = [ndfy(b, length=arr.ndim) for b in listify(bezels)]``
    ``ndfy(bezel_nd, length=arr.ndim)`` will give correct bezel, e.g., ``[[10,
    10], [10, 10]]`` for all of the following cases::

      1. ``bezel=10``
      2. ``bezel=[10, 10]``,
      3. ``bezel=[[10, 10], [10, 10]]``.

    It is also useful for `slicefy`.

    Note that some cases can be ambiguous:
    ``ndfy([[1, 2, 3]], length=3)`` may mean either::

      1. ``((1, 2, 3), (1, 2, 3), (1, 2, 3))``
      2. ``((1, 1, 1), (2, 2, 2), (3, 3, 3))``

    `ndfy` uses the first assumption.
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
