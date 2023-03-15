from astro_ndslice import slice_from_string, slicefy, bezel2slice
from astro_ndslice.slices import _defitsify_slice
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal as assert_array


def test_slice_from_string():
    arr1d = np.arange(5)
    a_slice = slice_from_string('[2:5]')
    assert_array(arr1d[a_slice], np.array([2, 3, 4]))

    a_slice = slice_from_string('[ : : -2] ')
    assert_array(arr1d[a_slice], np.array([4, 2, 0]))

    arr2d = np.array([arr1d, arr1d + 5, arr1d + 10])
    assert_array(
        arr2d,
        np.array([[ 0, 1, 2, 3, 4],
                  [ 5, 6, 7, 8, 9],
                  [10, 11, 12, 13, 14]])
    )

    a_slice = slice_from_string('[1:-1, 0:4:2]')
    assert_array(arr2d[a_slice], np.array([[5, 7]]))

    a_slice = slice_from_string('[0:2,0:3]')
    assert_array(
        arr2d[a_slice],
        np.array([[0, 1, 2],
                  [5, 6, 7]])
    )
    a_slice = slice_from_string('[1:3, 1:2]', fits_convention=True)
    assert_array(
        arr2d[a_slice],
        np.array([[0, 1, 2],
                  [5, 6, 7]])
    )

    assert slice_from_string('') == ()

    with pytest.raises(ValueError):
        slice_from_string("1:2")


def test_bezel2slice():
    arr = np.arange(100).reshape(10, 10)

    # scalar input
    assert_array(
        arr[bezel2slice(4)],
        np.array([[44, 45],
                  [54, 55]])
    )
    assert_array(
        arr[bezel2slice(4, order_xyz=False)],
        np.array([[44, 45],
                  [54, 55]])
    )

    # 1-element input
    assert_array(
        arr[bezel2slice([4])],
        np.array([[44, 45],
                  [54, 55]])
    )
    assert_array(
        arr[bezel2slice([4], order_xyz=False)],
        np.array([[44, 45],
                  [54, 55]])
    )

    # 2-element input
    assert_array(
        arr[bezel2slice([3, 4])],
        np.array([[43, 44, 45, 46],
                  [53, 54, 55, 56]])
    )
    assert_array(
        arr[bezel2slice([3, 4], order_xyz=False)],
        np.array([[34, 35],
                  [44, 45],
                  [54, 55],
                  [64, 65]])
    )


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
    assert_array(arr[slicefy(None)], arr)

    # === Bezel-like
    # --- scalar input
    assert_array(
        arr[slicefy(4)],
        np.array([[44, 45],
                  [54, 55]])
    )
    assert_array(
        arr[slicefy(4, order_xyz=False)],
        np.array([[44, 45],
                  [54, 55]])
    )

    assert_array(
        np.eye(5)[slicefy(1)],
        np.array([[1., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.]])
    )

    # --- 1-element input
    assert_array(
        arr[slicefy([4])],
        np.array([[44, 45],
                  [54, 55]])
    )
    assert_array(
        arr[slicefy([4], order_xyz=False)],
        np.array([[44, 45],
                  [54, 55]])
    )

    # --- 2-element input
    assert_array(
        arr[slicefy([3, 4])],
        np.array([[43, 44, 45, 46],
                  [53, 54, 55, 56]])
    )
    assert_array(
        arr[slicefy([3, 4], order_xyz=False)],
        np.array([[34, 35],
                  [44, 45],
                  [54, 55],
                  [64, 65]])
    )

    assert_array(
        np.eye(5)[slicefy((1, 2))],  # bezel by (1, 1), (2, 2) pix (x/y dir)
        np.array([[0., 1., 0.]])
    )

    # === str
    assert_array(
        np.eye(5)[slicefy('[1:2,:]')],
        np.array([[1., 0.],
                  [0., 1.],
                  [0., 0.],
                  [0., 0.],
                  [0., 0.]])
    )

    # === slice
    assert_array(
        np.eye(5)[slicefy(slice(1, -1, 2))],  # data[1:-1:2, 1:-1:2]
        np.array([[1., 0.],
                  [0., 1.]])
    )

    assert_array(
        np.eye(5)[slicefy([slice(1, -1, 2), slice(1, -1, 2)])],  # data[1:-1:2, 1:-1:2]
        np.array([[1., 0.],
                  [0., 1.]])
    )

    # === error
    with pytest.raises(TypeError):
        slicefy(1.0)
