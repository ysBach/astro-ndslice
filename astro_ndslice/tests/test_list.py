import numpy as np
import pytest

from astro_ndslice import is_list_like, listify, ndfy


def test_is_list_like_scalar():
    assert not is_list_like(12)
    assert not is_list_like("asdf")
    assert not is_list_like(None)
    assert not is_list_like(True)
    assert not is_list_like(False)

    # multiple args:
    assert not is_list_like(False, 12)


def test_is_list_like_list_tuple_dict():
    assert is_list_like([])
    assert is_list_like([1])
    assert is_list_like([1], [2])

    assert is_list_like(())
    assert is_list_like((1,))
    assert is_list_like((1,), (2,))

    assert is_list_like([1, (1, 2)])
    assert is_list_like({})
    assert is_list_like({1: 2}, (1, ))


def test_is_list_like_set():
    assert is_list_like(set(), allow_sets=True)
    assert is_list_like(set([1]), allow_sets=True)
    assert is_list_like(set([1, 2]), [1], allow_sets=True)
    assert not is_list_like(set(), allow_sets=False)


def test_is_list_like_any():
    assert is_list_like([1, 2], func=any)
    assert is_list_like([1, 2], 1, func=any)


def test_listify():
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


def test_ndfy():
    assert ndfy(1) == [1]
    assert ndfy(1, length=2) == [1, 1]
    assert ndfy(None, length=2, default=1) == [1, 1]

    arr = np.arange(1000).reshape(20, 50)  # 2-D array
    bezels1 = 20
    bezels2 = (20, 20)
    bezels3 = ((20, 20), (20, 20))
    ans = [[20, 20], [20, 20]]

    bezel1_nd = [ndfy(b, length=arr.ndim) for b in listify(bezels1)]
    bezel2_nd = [ndfy(b, length=arr.ndim) for b in listify(bezels2)]
    bezel3_nd = [ndfy(b, length=arr.ndim) for b in listify(bezels3)]
    assert ndfy(bezel1_nd, length=arr.ndim) == ans
    assert ndfy(bezel2_nd, length=arr.ndim) == ans
    assert ndfy(bezel3_nd, length=arr.ndim) == ans

    bezel = [[1, 2]]
    assert (ndfy(bezel, arr.ndim) == [[1, 2], [1, 2]])