"""
Simple tools to make lists
"""
from collections import abc
from typing import Any

import numpy as np

__all__ = [
    "is_list_like",
    "listify",
    "ndfy",
]


def is_list_like(
    *objs,
    allow_sets: bool = True,
    func: object = all
) -> bool:
    """ Check if inputs are list-like

    Parameters
    ----------
    *objs : object
        Objects to check.
    allow_sets : bool, optional.
        If this parameter is `False`, sets will not be considered list-like.
        Default: `True`
    func : functional object, optional.
        The function to be applied to each element. Useful ones are `all` and
        `any`.
        Default: `all`

    Notes
    -----
    Direct copy from pandas, with slight modification to accept *args and
    all/any, etc, functionality by `func`.
    https://github.com/pandas-dev/pandas/blob/bdb00f2d5a12f813e93bc55cdcd56dcb1aae776e/pandas/_libs/lib.pyx#L1026

    Note that pd.DataFrame also returns True.

    Timing on MBP 14" [2021, macOS 12.2, M1Pro(6P+2E/G16c/N16c/32G)]
    %timeit yfu.is_list_like("asdfaer.fits")
    4.32 µs +- 572 ns per loop (mean +- std. dev. of 7 runs, 100000 loops each)
    """
    # I don't think we need speed boost here but...
    # if `func` is `any`, below can be reorganized by for loop and can return
    # `True` once it reaches `True` for the first time.
    return func(
        isinstance(obj, abc.Iterable)
        # we do not count strings/unicode/bytes as list-like
        and not isinstance(obj, (str, bytes))
        # exclude zero-dimensional numpy arrays, effectively scalars
        and not (isinstance(obj, np.ndarray) and obj.ndim == 0)
        # exclude sets if allow_sets is False
        and not (allow_sets is False and isinstance(obj, abc.Set))
        for obj in objs
    )


def listify(
    *objs,
    scalar2list: bool = True,
    none2list: bool = False
) -> list:
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

    scalar2list : bool, optional.
        If `True`, a single scalar input will be converted to a list of a
        target length. Otherwise, it will be returned as is.

    none2list : bool, optional.
        Whether to return an empty list (`[]`). If `True`, ``[None]`` is
        returned if `objs` is `None`.
        Default: `False`

    Notes
    -----
    If any obj of `None` need to be converted to a length>1 list, it will be
    made as [None, None, ...], rather than an empty list, regardless of
    `empty_if_none`.

    Timing on MBP 14" [2021, macOS 12.2, M1Pro(6P+2E/G16c/N16c/32G)]:
    %timeit yfu.listify([12])
    8.92 µs +- 434 ns per loop (mean +- std. dev. of 7 runs, 100000 loops each)
    %timeit yfu.listify("asdf")
    7.08 µs +- 407 ns per loop (mean +- std. dev. of 7 runs, 100000 loops each)
    %timeit yfu.listify("asdf", scalar2list=False)
    7.37 µs +- 586 ns per loop (mean +- std. dev. of 7 runs, 100000 loops each)

    """
    def _listify_single(obj, none2list=True):
        if obj is None:
            return [obj] if none2list else []
        elif is_list_like(obj):
            return list(obj)
        else:
            return [obj] if scalar2list else obj

    if len(objs) == 1:
        return _listify_single(objs[0], none2list=none2list)

    objlists = [_listify_single(obj, none2list=True) for obj in objs]
    lengths = [len(obj) for obj in objlists]
    length = max(lengths)
    for objl in objlists:
        if len(objl) not in [1, length]:
            raise ValueError(f"Each input must be 1 or max(lengths)={length}.")

    return [obj*length if len(obj) == 1 else obj for obj in objlists]


def ndfy(
    item,
    length: int | None = None,
    default: Any = None
) -> list:
    """ Make an item to a list of `length`.

    Parameters
    ----------
    item : None, general object, list-like
        The item to be made into a list. If `None`, it will be filled by
        `default`.

    length : int, optional.
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

    Note that some cases can be ambiguous: ``ndfy([[1, 2, 3]], length=3)`` may mean either::

      1. ``((1, 2, 3), (1, 2, 3), (1, 2, 3))``
      2. ``((1, 1, 1), (2, 2, 2), (3, 3, 3))``

    `ndfy` uses the first assumption.
    """
    item = [default if i is None else i for i in listify(item, none2list=True)]
    item_length = len(item)

    if (length is None) or (item_length == length):
        return item
    elif item_length != 1:
        _length = "1" if item_length == 1 else f"1 or `length`(={length})"
        raise ValueError(f"`len(item)` must be {_length}. Now it is {item_length}.")

    # Now, item_length == 1
    return item * length
