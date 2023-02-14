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
    func : funtional object, optional.
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
    """Make multiple object into list of same length.

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

    Tests
    -----
    assert listify(12) == [12]
    assert listify([1]) == [1]
    assert listify(12, scalar2list=False) == 12
    assert listify([1], scalar2list=False) == [1]
    assert listify(None) == []
    assert listify(None, none2list=True) == [None]
    assert listify([1, 2]) == [1, 2]
    assert listify([1, "a"]) == [1, "a"]
    with pytest.raises(ValueError):
        listify([1, 2], "a", [3, 4, 5])
    # Below are the most important cases `listify` is useful
    assert listify("ab") == ["ab"]
    assert listify("ab", scalar2list=False) == "ab"
    assert listify([1, 2], "a") == [[1, 2], ['a', 'a']]
    assert listify([1, 2], "a", None) == [[1, 2], ['a', 'a'], [None, None]]

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
    default: Any = 0
):
    """ Make an item to ndarray of length.

    Parameters
    ----------
    item : None, general object, list-like
        The item to be made into an ndarray. If `None`, `default` is used.

    length : int, optional.
        The length of the final ndarray. If `None`, the length of the input is
        used (if `item` is a scalar, a length-1 array is returned).

    default : general object
        The default value to be used if `item` or any element of `item` is
        `None`.

    Notes
    -----
    Useful for the cases when bezels, sigma, ... are needed. Example case when
    you want to make ((20, 20), (20, 20)) bezels from given ``bezels = 20``::
    >>> arr = np.arange(10).reshape(2, 5)  # 2-D array
    >>> bezels1 = 20
    >>> bezels2 = (20, 20)
    >>> bezels3 = ((20, 20), (20, 20))
    >>> ans = [[20, 20], [20, 20]]
    >>> assert ndfy([ndfy(b, 2, default=0) for b in listify(bezels1)], arr.ndim) == ans
    >>> assert ndfy([ndfy(b, 2, default=0) for b in listify(bezels2)], arr.ndim) == ans
    >>> assert ndfy([ndfy(b, 2, default=0) for b in listify(bezels3)], arr.ndim) == ans

    Note that 2-element bezels is ambiguous: ``[a, b]`` may mean either (1)
    both bezels in x-axis to be ``a`` and both bezels in y-axis to be ``b`` or
    (2) both x-/y-axis have bezels ``[a, b]`` for lower/upper regions. `ndfy`
    uses the first assumption::
    >>> ndfy([ndfy(b, 2, default=0) for b in listify([(30, 40)])], arr.ndim)
    >>> # [[30, 30], [40, 40]]
    Because of this, it will raise ValueError if `arr.ndim != bezels.size`.
    """
    item = listify(item)
    item = [default if i is None else i for i in item]

    if (length is None) or (len(item) == length):
        return item
    elif len(item) != 1:
        raise ValueError(f"Length of item must be 1 or {length=}. Now it is {len(item)}.")

    # Now, len(item) == 1
    return [item[0] for _ in range(length)]
