import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from astro_ndslice import bezel2slice, slice_from_string, slice_to_string, slicefy
from astro_ndslice.slices import _defitsify_slice, _fitsify_slice


def test_slice_from_string():
    arr1d = np.arange(5)
    a_slice = slice_from_string("[2:5]")
    assert_array_equal(arr1d[a_slice], np.array([2, 3, 4]))

    a_slice = slice_from_string("[ : : -2] ")
    assert_array_equal(arr1d[a_slice], np.array([4, 2, 0]))

    arr2d = np.array([arr1d, arr1d + 5, arr1d + 10])
    assert_array_equal(
        arr2d, np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14]])
    )

    a_slice = slice_from_string("[1:-1, 0:4:2]")
    assert_array_equal(arr2d[a_slice], np.array([[5, 7]]))

    a_slice = slice_from_string("[0:2,0:3]")
    assert_array_equal(arr2d[a_slice], np.array([[0, 1, 2], [5, 6, 7]]))
    a_slice = slice_from_string("[1:3, 1:2]", fits_convention=True)
    assert_array_equal(arr2d[a_slice], np.array([[0, 1, 2], [5, 6, 7]]))

    assert slice_from_string("") == ()

    with pytest.raises(ValueError):
        slice_from_string("1:2")

    # Empty brackets are the inverse of the empty input string.
    assert slice_from_string("[]") == ()

    with pytest.raises(ValueError, match="step cannot be zero"):
        slice_from_string("[1:2:0]", fits_convention=True)
    with pytest.raises(ValueError, match="index"):
        slice_from_string("[0:2]", fits_convention=True)
    with pytest.raises(ValueError, match="index"):
        slice_from_string("[1:0]", fits_convention=True)
    with pytest.raises(ValueError, match="incompatible"):
        slice_from_string("[1:2:-1]", fits_convention=True)


def test_bezel2slice():
    arr = np.arange(100).reshape(10, 10)

    # scalar input
    assert_array_equal(arr[bezel2slice(4)], np.array([[44, 45], [54, 55]]))
    assert_array_equal(
        arr[bezel2slice(4, order_xyz=False)], np.array([[44, 45], [54, 55]])
    )

    # 1-element input
    assert_array_equal(arr[bezel2slice([4])], np.array([[44, 45], [54, 55]]))
    assert_array_equal(
        arr[bezel2slice([4], order_xyz=False)], np.array([[44, 45], [54, 55]])
    )

    # 2-element input
    assert_array_equal(
        arr[bezel2slice([3, 4])], np.array([[43, 44, 45, 46], [53, 54, 55, 56]])
    )
    assert_array_equal(
        arr[bezel2slice([3, 4], order_xyz=False)],
        np.array([[34, 35], [44, 45], [54, 55], [64, 65]]),
    )

    assert bezel2slice(None, ndim=3) == (slice(None),) * 3


def test_defitsify_slice():
    assert _defitsify_slice([slice(1, 10)]) == [slice(0, 10, None)]
    assert _defitsify_slice([slice(1, 10, 2)]) == [slice(0, 10, 2)]
    assert _defitsify_slice([slice(100, 10)]) == [slice(99, 8, -1)]

    with pytest.raises(ValueError):
        _defitsify_slice([slice(-1, 10)])
    with pytest.raises(ValueError):
        _defitsify_slice([slice(10, -1)])


def test_slicefy():
    arr = np.arange(100).reshape(10, 10)

    # === None
    assert slicefy(None) == (slice(None), slice(None))
    assert_array_equal(arr[slicefy(None)], arr)

    # === Bezel-like
    # --- scalar input
    assert_array_equal(arr[slicefy(4)], np.array([[44, 45], [54, 55]]))
    assert_array_equal(arr[slicefy(4, order_xyz=False)], np.array([[44, 45], [54, 55]]))

    assert_allclose(
        np.eye(5)[slicefy(1)],
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )
    assert_array_equal(arr[slicefy(np.int64(4))], arr[slicefy(4)])

    # --- 1-element input
    assert_array_equal(arr[slicefy([4])], np.array([[44, 45], [54, 55]]))
    assert_array_equal(
        arr[slicefy([4], order_xyz=False)], np.array([[44, 45], [54, 55]])
    )

    # --- 2-element input
    assert_array_equal(
        arr[slicefy([3, 4])], np.array([[43, 44, 45, 46], [53, 54, 55, 56]])
    )
    assert_array_equal(
        arr[slicefy([3, 4], order_xyz=False)],
        np.array([[34, 35], [44, 45], [54, 55], [64, 65]]),
    )

    assert_allclose(
        np.eye(5)[slicefy((1, 2))],  # bezel by (1, 1), (2, 2) pix (x/y dir)
        np.array([[0.0, 1.0, 0.0]]),
    )

    # === str
    assert_allclose(
        np.eye(5)[slicefy("[1:2,:]")],
        np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
    )

    # === slice
    assert_allclose(
        np.eye(5)[slicefy(slice(1, -1, 2))],  # data[1:-1:2, 1:-1:2]
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    )

    assert_allclose(
        np.eye(5)[slicefy([slice(1, -1, 2), slice(1, -1, 2)])],  # data[1:-1:2, 1:-1:2]
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    )

    # === fits_convention=False: Python-style indexing
    assert slicefy("[1:3,0:2]", fits_convention=False) == (
        slice(1, 3, None),
        slice(0, 2, None),
    )
    assert slicefy("[0:5,:]", fits_convention=False) == (
        slice(0, 5, None),
        slice(None, None, None),
    )

    # === error
    with pytest.raises(TypeError):
        slicefy(1.0)

    assert slicefy((slice(1, 3) for _ in range(1)), ndim=2) == (
        slice(1, 3),
        slice(1, 3),
    )
    assert slicefy((1 for _ in range(1)), ndim=2) == bezel2slice(1, ndim=2)
    with pytest.raises(ValueError, match="at least one"):
        slicefy(iter(()))


def test_fitsify_slice():
    # round-trip: defitsify then fitsify must be identity
    fits_inputs = [
        [slice(1, 10)],
        [slice(1, 10, 2)],
        [slice(10, 1)],  # inverted: FITS 10:1
        [slice(1, 5), slice(2, 8)],
    ]
    for s in fits_inputs:
        assert _fitsify_slice(_defitsify_slice(s)) == s

    # known values
    # Python slice(0, 10) → FITS slice(1, 10)
    assert _fitsify_slice([slice(0, 10)]) == [slice(1, 10, None)]
    # Python inverted slice(9, None, -1) → FITS slice(10, 1)
    assert _fitsify_slice([slice(9, None, -1)]) == [slice(10, 1, None)]

    with pytest.raises(ValueError):
        _fitsify_slice([slice(-1, 5)])


def test_slice_to_string():
    arr2d = np.arange(25).reshape(5, 5)

    # round-trip: slice_from_string → slice_to_string
    for s in ["[1:5,2:4]", "[2:3,:]", "[1:5,1:5]"]:
        py_sl = slice_from_string(s, fits_convention=True)
        assert slice_to_string(py_sl, fits_convention=True) == s

    # fits_convention=False: output is in Python (zyx) order, 0-indexed
    assert (
        slice_to_string((slice(1, 3), slice(0, 4, 2)), fits_convention=False)
        == "[1:3,0:4:2]"
    )
    assert slice_to_string((slice(None), slice(None)), fits_convention=False) == "[:,:]"
    assert slice_to_string((slice(None),), fits_convention=False) == "[:]"

    # verify the slice actually selects expected data
    # FITS '[1:3,2:5]': x=1:3 (cols 0-2), y=2:5 (rows 1-4) → Python [1:5, 0:3]
    py_sl = slice_from_string("[1:3,2:5]", fits_convention=True)
    assert slice_to_string(py_sl) == "[1:3,2:5]"
    assert_array_equal(arr2d[py_sl], arr2d[1:5, 0:3])

    assert slice_to_string(()) == "[]"


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        (slice(None, None, -1), "[:1:-1]"),
        (slice(None, 1, -2), "[:3:-2]"),
        (slice(9, None, -1), "[10:1]"),
        (slice(9, 0, -2), "[10:2:2]"),
        (slice(8, 1, -3), "[9:3:3]"),
        # A one-element descending slice has equal serialized endpoints but
        # remains non-empty after parsing.
        (slice(2, 1, -1), "[3:3]"),
    ],
)
def test_slice_to_string_reverse_roundtrip(original, expected):
    values = np.arange(15)
    encoded = slice_to_string((original,))
    assert encoded == expected
    recovered = slice_from_string(encoded, fits_convention=True)
    for size in range(values.size + 1):
        np.testing.assert_array_equal(
            values[:size][(original,)], values[:size][recovered]
        )


@pytest.mark.parametrize(
    "original",
    [
        slice(2, 2),
        slice(2, 2, -1),
        slice(2, 1, 2),
        slice(0, 0, -1),
        slice(-1, None),
        slice(None, -1),
        slice(None, None, 0),
    ],
)
def test_slice_to_string_rejects_unrepresentable_slices(original):
    with pytest.raises(ValueError):
        slice_to_string((original,))
