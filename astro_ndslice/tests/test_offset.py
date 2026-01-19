import numpy as np
import pytest
from numpy.testing import assert_allclose

from astro_ndslice import (calc_offset_physical, calc_offset_wcs,
                           offseted_shape, offsets2slice, regularize_offsets)


def test_regularize_offsets():
    assert_allclose(
        regularize_offsets([[0, 0, 0], [0, 1, 2], [1, 1.5, 1]]),
        np.array([[0. , 0. , 0. ],
                  [2. , 1. , 0. ],
                  [1. , 1.5, 1. ]])
    )
    assert_allclose(
        regularize_offsets([[0, 0, 0], [0, 1, 2], [1, 1.5, 1]], intify_offsets=True),
        np.array([[0, 0, 0],
                  [2, 1, 0],
                  [1, 2, 1]])
    )

    assert_allclose(
        regularize_offsets([[0, 0, 0], [0, 1, 2], [1, 1.5, 1]], offset_order_xyz=False),
        np.array([[0. , 0. , 0. ],
                  [0. , 1. , 2. ],
                  [1. , 1.5, 1. ]])
    )


def test_offseted_shape():
    shapes = [(10, 10), (10, 10), (10, 10)]
    offsets = [[1, 1], [-1, 1], [1.9, 0]]

    res = offseted_shape(shapes, offsets, method="outer", offset_order_xyz=True,
                         intify_offsets=False, pythonize_offsets=True)
    assert_allclose(
        res[0],
        np.array([[1. , 2. ],
                  [1. , 0. ],
                  [0. , 2.9]])
    )
    assert res[1] == (11, 13)

    res = offseted_shape(shapes, offsets, method="outer", offset_order_xyz=True,
                         intify_offsets=False, pythonize_offsets=False)
    assert_allclose(
        res[0],
        np.array([[2. , 1. ],
                  [0. , 1. ],
                  [2.9, 0. ]])
    )
    assert res[1] == (11, 13)

    res = offseted_shape(shapes, offsets, method="outer", offset_order_xyz=True,
                         intify_offsets=True, pythonize_offsets=True)
    assert_allclose(
        res[0],
        np.array([[1, 2],
                  [1, 0],
                  [0, 3]])
    )
    assert res[1] == (11, 13)

    res = offseted_shape(shapes, offsets, method="outer", offset_order_xyz=False,
                         intify_offsets=False, pythonize_offsets=True)
    assert_allclose(
        res[0],
        np.array([[2. , 1. ],
                  [0. , 1. ],
                  [2.9, 0. ]])
    )
    assert res[1] == (13, 11)

    res = offseted_shape(shapes, offsets, method="inner", offset_order_xyz=True,
                         intify_offsets=False, pythonize_offsets=True)
    assert_allclose(
        res[0],
        np.array([[1. , 2. ],
                  [1. , 0. ],
                  [0. , 2.9]])
    )
    assert res[1] == (9, 7)

    with pytest.raises(ValueError):
        offseted_shape(shapes, offsets, method="invalid")

    with pytest.raises(ValueError):
        # no overlap
        offseted_shape([(10, 10), (10, 10)], offsets=[(10, 10), (-20, -20)], method="inner")


def test_offsets2slice():
    shapes = [(10, 15), (10, 10), (10, 10)]
    offsets = [[1, 1], [-1, 1], [1.9, 0]]
    assert (offsets2slice(shapes, offsets, method="outer", shape_order_xyz=False,
                          offset_order_xyz=True, outer_for_stack=True,
                          fits_convention=False)
            == [[slice(0, 1, None), slice(1, 11, None), slice(2, 17, None)],
                [slice(1, 2, None), slice(1, 11, None), slice(0, 10, None)],
                [slice(2, 3, None), slice(0, 10, None), slice(3, 13, None)]])

    assert (offsets2slice(shapes, offsets, method="outer", shape_order_xyz=True,
                          offset_order_xyz=True, outer_for_stack=True,
                          fits_convention=False)
            == [[slice(0, 1, None), slice(1, 16, None), slice(2, 12, None)],
                [slice(1, 2, None), slice(1, 11, None), slice(0, 10, None)],
                [slice(2, 3, None), slice(0, 10, None), slice(3, 13, None)]])

    assert (offsets2slice(shapes, offsets, method="outer", shape_order_xyz=False,
                          offset_order_xyz=False, outer_for_stack=True,
                          fits_convention=False)
            == [[slice(0, 1, None), slice(2, 12, None), slice(1, 16, None)],
                [slice(1, 2, None), slice(0, 10, None), slice(1, 11, None)],
                [slice(2, 3, None), slice(3, 13, None), slice(0, 10, None)]])

    assert (offsets2slice(shapes, offsets, method="outer", shape_order_xyz=False,
                          offset_order_xyz=True, outer_for_stack=False,
                          fits_convention=False)
            == [[slice(1, 11, None), slice(2, 17, None)],
                [slice(1, 11, None), slice(0, 10, None)],
                [slice(0, 10, None), slice(3, 13, None)]])

    assert (offsets2slice(shapes, offsets, method="outer", shape_order_xyz=False,
                          offset_order_xyz=True, outer_for_stack=True,
                          fits_convention=True)
            == ['[3:17,2:11,1:1]', '[1:10,2:11,2:2]', '[4:13,1:10,3:3]'])

    assert (offsets2slice(shapes, offsets, method="inner", shape_order_xyz=False,
                          offset_order_xyz=True, outer_for_stack=True,
                          fits_convention=False)
            == [[slice(0, 9, None), slice(1, 8, None)],
                [slice(0, 9, None), slice(3, 10, None)],
                [slice(1, 10, None), slice(0, 7, None)]]
            )

    assert (offsets2slice(shapes, offsets, method="inner", shape_order_xyz=False,
                          offset_order_xyz=True, outer_for_stack=True,
                          fits_convention=True)
            == ['[2:8,1:9]', '[4:10,1:9]', '[1:7,2:10]']
            )

    with pytest.raises(ValueError):
        offsets2slice(shapes, offsets, method="invalid")

    with pytest.raises(ValueError):
        # no overlap
        offsets2slice([(10, 10), (10, 10)], offsets=[(10, 10), (-20, -20)], method="inner")

    with pytest.raises(ValueError):
        # shapes.ndim != 2
        offsets2slice([[(10, 10), (10, 10)]], offsets=[(10, 10), (-20, -20)])

    with pytest.raises(ValueError):
        # offsets.ndim != 2
        offsets2slice([(10, 10), (10, 10)], offsets=[[(10, 10), (-20, -20)]])

    with pytest.raises(ValueError):
        # shape mismatch
        offsets2slice(shapes, offsets[1:])


def test_calc_offset_wcs():
    from astropy.io import fits
    from astropy.wcs import WCS

    w1 = WCS(fits.Header.fromstring("""
NAXIS   =                    2
NAXIS1  =                  721
NAXIS2  =                  720
EXTEND  =                    T / FITS dataset may contain extensions
CTYPE1  = 'RA---TAN'
CTYPE2  = 'DEC--TAN'
CRVAL1  =           266.400000
CRVAL2  =           -28.933330
CRPIX1  =                 361.
CRPIX2  =                360.5
CDELT1  =         -0.001388889
CDELT2  =          0.001388889
CROTA2  =             0.000000
EQUINOX =               2000.0""", sep="\n"))
    w2 = WCS(fits.Header.fromstring("""
NAXIS   =                    2
NAXIS1  =                  721
NAXIS2  =                  720
EXTEND  =                    T / FITS dataset may contain extensions
CTYPE1  = 'RA---TAN'
CTYPE2  = 'DEC--TAN'
CRVAL1  =           266.400000
CRVAL2  =           -28.933330
CRPIX1  =                362.9
CRPIX2  =                360.5
CDELT1  =         -0.001388889
CDELT2  =          0.001388889
CROTA2  =             0.000000
EQUINOX =               2000.0""", sep="\n"))
    assert_allclose(
        calc_offset_wcs(w1, w2, loc_target="center", loc_reference="center",
                        order_xyz=True, intify_offset=False),
        np.array([1.9, 0.0]),
        atol=1e-9
    )

    assert_allclose(
        calc_offset_wcs(w1, w2, loc_target="center", loc_reference="origin",
                        order_xyz=True, intify_offset=False),
        np.array([362.4, 360. ]),
        atol=1e-9
    )

    assert_allclose(
        calc_offset_wcs(w1, w2, loc_target="origin", loc_reference="center",
                        order_xyz=True, intify_offset=False),
        np.array([-358.6, -360. ]),
        atol=1e-9
    )

    assert_allclose(
        calc_offset_wcs(w1, w2, loc_target="center", loc_reference="center",
                        order_xyz=True, intify_offset=True),
        np.array([2, 0])
    )

    assert_allclose(
        calc_offset_wcs(w1, w2, loc_target="center", loc_reference="center",
                        order_xyz=False, intify_offset=False),
        np.array([0.0, 1.9]),
        atol=1e-9
    )

    assert_allclose(
        calc_offset_wcs(w1, w2, loc_target="center", loc_reference=(350, 350),
                        order_xyz=True, intify_offset=False),
        np.array([12.4, 10. ]),
        atol=1e-9
    )

    with pytest.raises(TypeError):
        calc_offset_wcs(w1, "asdf")


def test_calc_offset_physical():
    from astropy.io import fits

    hdr = fits.Header.fromstring("""
NAXIS   =                    2 / number of array dimensions
NAXIS1  =                   91
NAXIS2  =                   91
LTV1    =                 -9.5
LTV2    =                  -19
LTM1_1  =                    1
LTM1_2  =                  0.0
LTM2_1  =                  0.0
LTM2_2  =                    1""", sep="\n")

    hdr2 = fits.Header.fromstring("""
NAXIS   =                    2 / number of array dimensions
NAXIS1  =                   91
NAXIS2  =                   91
LTV1    =                    0
LTV2    =                 -1.5
LTM1_1  =                    1
LTM1_2  =                  3.0
LTM2_1  =                  2.0
LTM2_2  =                    1""", sep="\n")

    hdr3 = fits.Header.fromstring("""
NAXIS   =                    2 / number of array dimensions
NAXIS1  =                   91
NAXIS2  =                   91
LTV1    =                    0
LTV2    =                 -1.5""", sep="\n")

    hdr4 = fits.Header.fromstring("""
NAXIS   =                    2 / number of array dimensions
NAXIS1  =                   91
NAXIS2  =                   91
LTV1    =                    0
LTV2    =                 -1.5
LTM1  =                      1
LTM2  =                    3.0""", sep="\n")

    hdr5 = fits.Header.fromstring("""
NAXIS   =                    2 / number of array dimensions
NAXIS1  =                   91
NAXIS2  =                   91""", sep="\n")
    assert_allclose(
        calc_offset_physical(hdr, reference=None, order_xyz=True,
                             ignore_ltm=True, intify_offset=False),
        np.array([ -9.5, -19])
    )

    assert_allclose(
        calc_offset_physical(hdr, reference=None, order_xyz=False,
                             ignore_ltm=True, intify_offset=False),
        np.array([-19, -9.5])
    )

    assert_allclose(
        calc_offset_physical(hdr, reference=None, order_xyz=True,
                             ignore_ltm=False, intify_offset=False),
        np.array([ -9.5, -19])
    )
    assert_allclose(
        calc_offset_physical(hdr3, reference=None, order_xyz=True,
                             ignore_ltm=False, intify_offset=False),
        np.array([ 0. , -1.5])
    )

    assert_allclose(
        calc_offset_physical(hdr, reference=None, order_xyz=True,
                             ignore_ltm=True, intify_offset=True),
        np.array([ -10, -19])
    )

    assert_allclose(
        calc_offset_physical(hdr, reference=hdr2, order_xyz=True,
                             ignore_ltm=True, intify_offset=False),
        np.array([ -9.5, -17.5])
    )

    assert_allclose(
        calc_offset_physical(hdr, reference=hdr3, order_xyz=True,
                             ignore_ltm=True, intify_offset=False),
        np.array([ -9.5, -17.5])
    )

    with pytest.raises(NotImplementedError):
        calc_offset_physical(hdr, reference=hdr2, ignore_ltm=False)

    with pytest.raises(NotImplementedError):
        calc_offset_physical(hdr, reference=hdr4, ignore_ltm=False)

    with pytest.raises(TypeError):
        calc_offset_physical("asdf", reference=hdr4)

    with pytest.raises(TypeError):
        calc_offset_physical(hdr, reference="asdf")

    assert_allclose(
        calc_offset_physical(hdr5),
        np.array([ 0, 0])
    )

    assert_allclose(
        calc_offset_physical(hdr, reference=hdr5),
        np.array([ -9.5, -19.])
    )
