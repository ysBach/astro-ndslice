import numpy as np
import pytest
from numpy.testing import assert_allclose

from astro_ndslice import (
    calc_offset_physical,
    calc_offset_wcs,
    offseted_shape,
    offsets2slice,
    regularize_offsets,
    slice_from_string,
)


def test_regularize_offsets():
    assert_allclose(
        regularize_offsets([[0, 0, 0], [0, 1, 2], [1, 1.5, 1]]),
        np.array([[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [1.0, 1.5, 1.0]]),
    )
    assert_allclose(
        regularize_offsets([[0, 0, 0], [0, 1, 2], [1, 1.5, 1]], intify_offsets=True),
        np.array([[0, 0, 0], [2, 1, 0], [1, 2, 1]]),
    )

    assert_allclose(
        regularize_offsets([[0, 0, 0], [0, 1, 2], [1, 1.5, 1]], offset_order_xyz=False),
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [1.0, 1.5, 1.0]]),
    )


def test_offseted_shape():
    shapes = [(10, 10), (10, 10), (10, 10)]
    offsets = [[1, 1], [-1, 1], [1.9, 0]]

    res = offseted_shape(
        shapes,
        offsets,
        method="outer",
        offset_order_xyz=True,
        intify_offsets=False,
        pythonize_offsets=True,
    )
    assert_allclose(res[0], np.array([[1.0, 2.0], [1.0, 0.0], [0.0, 2.9]]))
    assert res[1] == (11, 13)

    res = offseted_shape(
        shapes,
        offsets,
        method="outer",
        offset_order_xyz=True,
        intify_offsets=False,
        pythonize_offsets=False,
    )
    assert_allclose(res[0], np.array([[2.0, 1.0], [0.0, 1.0], [2.9, 0.0]]))
    assert res[1] == (11, 13)

    res = offseted_shape(
        shapes,
        offsets,
        method="outer",
        offset_order_xyz=True,
        intify_offsets=True,
        pythonize_offsets=True,
    )
    assert_allclose(res[0], np.array([[1, 2], [1, 0], [0, 3]]))
    assert res[1] == (11, 13)

    res = offseted_shape(
        shapes,
        offsets,
        method="outer",
        offset_order_xyz=False,
        intify_offsets=False,
        pythonize_offsets=True,
    )
    assert_allclose(res[0], np.array([[2.0, 1.0], [0.0, 1.0], [2.9, 0.0]]))
    assert res[1] == (13, 11)

    res = offseted_shape(
        shapes,
        offsets,
        method="inner",
        offset_order_xyz=True,
        intify_offsets=False,
        pythonize_offsets=True,
    )
    assert_allclose(res[0], np.array([[1.0, 2.0], [1.0, 0.0], [0.0, 2.9]]))
    assert res[1] == (9, 7)

    with pytest.raises(ValueError):
        offseted_shape(shapes, offsets, method="invalid")

    with pytest.raises(ValueError):
        # no overlap
        offseted_shape(
            [(10, 10), (10, 10)], offsets=[(10, 10), (-20, -20)], method="inner"
        )


def test_offsets2slice():
    shapes = [(10, 15), (10, 10), (10, 10)]
    offsets = [[1, 1], [-1, 1], [1.9, 0]]
    assert offsets2slice(
        shapes,
        offsets,
        method="outer",
        shape_order_xyz=False,
        offset_order_xyz=True,
        outer_for_stack=True,
        fits_convention=False,
    ) == [
        (slice(0, 1, None), slice(1, 11, None), slice(2, 17, None)),
        (slice(1, 2, None), slice(1, 11, None), slice(0, 10, None)),
        (slice(2, 3, None), slice(0, 10, None), slice(3, 13, None)),
    ]

    assert offsets2slice(
        shapes,
        offsets,
        method="outer",
        shape_order_xyz=True,
        offset_order_xyz=True,
        outer_for_stack=True,
        fits_convention=False,
    ) == [
        (slice(0, 1, None), slice(1, 16, None), slice(2, 12, None)),
        (slice(1, 2, None), slice(1, 11, None), slice(0, 10, None)),
        (slice(2, 3, None), slice(0, 10, None), slice(3, 13, None)),
    ]

    assert offsets2slice(
        shapes,
        offsets,
        method="outer",
        shape_order_xyz=False,
        offset_order_xyz=False,
        outer_for_stack=True,
        fits_convention=False,
    ) == [
        (slice(0, 1, None), slice(2, 12, None), slice(1, 16, None)),
        (slice(1, 2, None), slice(0, 10, None), slice(1, 11, None)),
        (slice(2, 3, None), slice(3, 13, None), slice(0, 10, None)),
    ]

    assert offsets2slice(
        shapes,
        offsets,
        method="outer",
        shape_order_xyz=False,
        offset_order_xyz=True,
        outer_for_stack=False,
        fits_convention=False,
    ) == [
        (slice(1, 11, None), slice(2, 17, None)),
        (slice(1, 11, None), slice(0, 10, None)),
        (slice(0, 10, None), slice(3, 13, None)),
    ]

    assert offsets2slice(
        shapes,
        offsets,
        method="outer",
        shape_order_xyz=False,
        offset_order_xyz=True,
        outer_for_stack=True,
        fits_convention=True,
    ) == ["[3:17,2:11,1:1]", "[1:10,2:11,2:2]", "[4:13,1:10,3:3]"]

    assert offsets2slice(
        shapes,
        offsets,
        method="inner",
        shape_order_xyz=False,
        offset_order_xyz=True,
        outer_for_stack=True,
        fits_convention=False,
    ) == [
        (slice(0, 9, None), slice(1, 8, None)),
        (slice(0, 9, None), slice(3, 10, None)),
        (slice(1, 10, None), slice(0, 7, None)),
    ]

    assert offsets2slice(
        shapes,
        offsets,
        method="inner",
        shape_order_xyz=False,
        offset_order_xyz=True,
        outer_for_stack=True,
        fits_convention=True,
    ) == ["[2:8,1:9]", "[4:10,1:9]", "[1:7,2:10]"]

    with pytest.raises(ValueError):
        offsets2slice(shapes, offsets, method="invalid")

    with pytest.raises(ValueError):
        # no overlap
        offsets2slice(
            [(10, 10), (10, 10)], offsets=[(10, 10), (-20, -20)], method="inner"
        )

    with pytest.raises(ValueError):
        # shapes.ndim != 2
        offsets2slice([[(10, 10), (10, 10)]], offsets=[(10, 10), (-20, -20)])

    with pytest.raises(ValueError):
        # offsets.ndim != 2
        offsets2slice([(10, 10), (10, 10)], offsets=[[(10, 10), (-20, -20)]])

    # shape_order_xyz=True, offset_order_xyz=False (untested combination)
    assert offsets2slice(
        shapes,
        offsets,
        method="outer",
        shape_order_xyz=True,
        offset_order_xyz=False,
        outer_for_stack=True,
        fits_convention=False,
    ) == [
        (slice(0, 1, None), slice(2, 17, None), slice(1, 11, None)),
        (slice(1, 2, None), slice(0, 10, None), slice(1, 11, None)),
        (slice(2, 3, None), slice(3, 13, None), slice(0, 10, None)),
    ]

    with pytest.raises(ValueError):
        # shape mismatch
        offsets2slice(shapes, offsets[1:])


def test_calc_offset_wcs():
    pytest.importorskip("astropy")
    from astropy.io import fits
    from astropy.wcs import WCS

    w1 = WCS(
        fits.Header.fromstring(
            """
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
EQUINOX =               2000.0""",
            sep="\n",
        )
    )
    w2 = WCS(
        fits.Header.fromstring(
            """
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
EQUINOX =               2000.0""",
            sep="\n",
        )
    )
    assert_allclose(
        calc_offset_wcs(
            w1,
            w2,
            loc_target="center",
            loc_reference="center",
            order_xyz=True,
            intify_offset=False,
        ),
        np.array([1.9, 0.0]),
        atol=1e-9,
    )

    assert_allclose(
        calc_offset_wcs(
            w1,
            w2,
            loc_target="center",
            loc_reference="origin",
            order_xyz=True,
            intify_offset=False,
        ),
        np.array([362.4, 360.0]),
        atol=1e-9,
    )

    assert_allclose(
        calc_offset_wcs(
            w1,
            w2,
            loc_target="origin",
            loc_reference="center",
            order_xyz=True,
            intify_offset=False,
        ),
        np.array([-358.6, -360.0]),
        atol=1e-9,
    )

    assert_allclose(
        calc_offset_wcs(
            w1,
            w2,
            loc_target="center",
            loc_reference="center",
            order_xyz=True,
            intify_offset=True,
        ),
        np.array([2, 0]),
    )

    assert_allclose(
        calc_offset_wcs(
            w1,
            w2,
            loc_target="center",
            loc_reference="center",
            order_xyz=False,
            intify_offset=False,
        ),
        np.array([0.0, 1.9]),
        atol=1e-9,
    )

    assert_allclose(
        calc_offset_wcs(
            w1,
            w2,
            loc_target="center",
            loc_reference=(350, 350),
            order_xyz=True,
            intify_offset=False,
        ),
        np.array([12.4, 10.0]),
        atol=1e-9,
    )

    # ndarray loc_target: same position as "center" of w1
    center_w1 = np.array(w1._naxis) / 2  # [360.5, 360.0] in xyz
    assert_allclose(
        calc_offset_wcs(
            w1,
            w2,
            loc_target=center_w1,
            loc_reference="center",
            order_xyz=True,
            intify_offset=False,
        ),
        np.array([1.9, 0.0]),
        atol=1e-9,
    )
    # ndarray loc_reference
    assert_allclose(
        calc_offset_wcs(
            w1,
            w2,
            loc_target="center",
            loc_reference=np.array([350.0, 350.0]),
            order_xyz=True,
            intify_offset=False,
        ),
        np.array([12.4, 10.0]),
        atol=1e-9,
    )

    with pytest.raises(TypeError):
        calc_offset_wcs(w1, "asdf")


def test_calc_offset_physical():
    pytest.importorskip("astropy")
    from astropy.io import fits

    hdr = fits.Header.fromstring(
        """
NAXIS   =                    2 / number of array dimensions
NAXIS1  =                   91
NAXIS2  =                   91
LTV1    =                 -9.5
LTV2    =                  -19
LTM1_1  =                    1
LTM1_2  =                  0.0
LTM2_1  =                  0.0
LTM2_2  =                    1""",
        sep="\n",
    )

    hdr2 = fits.Header.fromstring(
        """
NAXIS   =                    2 / number of array dimensions
NAXIS1  =                   91
NAXIS2  =                   91
LTV1    =                    0
LTV2    =                 -1.5
LTM1_1  =                    1
LTM1_2  =                  3.0
LTM2_1  =                  2.0
LTM2_2  =                    1""",
        sep="\n",
    )

    hdr3 = fits.Header.fromstring(
        """
NAXIS   =                    2 / number of array dimensions
NAXIS1  =                   91
NAXIS2  =                   91
LTV1    =                    0
LTV2    =                 -1.5""",
        sep="\n",
    )

    hdr4 = fits.Header.fromstring(
        """
NAXIS   =                    2 / number of array dimensions
NAXIS1  =                   91
NAXIS2  =                   91
LTV1    =                    0
LTV2    =                 -1.5
LTM1  =                      1
LTM2  =                    3.0""",
        sep="\n",
    )

    hdr5 = fits.Header.fromstring(
        """
NAXIS   =                    2 / number of array dimensions
NAXIS1  =                   91
NAXIS2  =                   91""",
        sep="\n",
    )
    assert_allclose(
        calc_offset_physical(
            hdr, reference=None, order_xyz=True, ignore_ltm=True, intify_offset=False
        ),
        np.array([-9.5, -19]),
    )

    assert_allclose(
        calc_offset_physical(
            hdr, reference=None, order_xyz=False, ignore_ltm=True, intify_offset=False
        ),
        np.array([-19, -9.5]),
    )

    assert_allclose(
        calc_offset_physical(
            hdr, reference=None, order_xyz=True, ignore_ltm=False, intify_offset=False
        ),
        np.array([-9.5, -19]),
    )
    assert_allclose(
        calc_offset_physical(
            hdr3, reference=None, order_xyz=True, ignore_ltm=False, intify_offset=False
        ),
        np.array([0.0, -1.5]),
    )

    assert_allclose(
        calc_offset_physical(
            hdr, reference=None, order_xyz=True, ignore_ltm=True, intify_offset=True
        ),
        np.array([-10, -19]),
    )

    assert_allclose(
        calc_offset_physical(
            hdr, reference=hdr2, order_xyz=True, ignore_ltm=True, intify_offset=False
        ),
        np.array([-9.5, -17.5]),
    )

    assert_allclose(
        calc_offset_physical(
            hdr, reference=hdr3, order_xyz=True, ignore_ltm=True, intify_offset=False
        ),
        np.array([-9.5, -17.5]),
    )

    with pytest.raises(NotImplementedError):
        calc_offset_physical(hdr, reference=hdr2, ignore_ltm=False)

    with pytest.raises(NotImplementedError):
        calc_offset_physical(hdr, reference=hdr4, ignore_ltm=False)

    with pytest.raises(TypeError):
        calc_offset_physical("asdf", reference=hdr4)

    with pytest.raises(TypeError):
        calc_offset_physical(hdr, reference="asdf")

    assert_allclose(calc_offset_physical(hdr5), np.array([0, 0]))

    assert_allclose(calc_offset_physical(hdr, reference=hdr5), np.array([-9.5, -19.0]))


@pytest.mark.parametrize("stack_axis", [False, True])
def test_offsets2slice_assigns_images_directly(stack_axis: bool) -> None:
    images = [np.arange(12).reshape(3, 4), np.arange(6).reshape(2, 3) + 20]
    shapes = [image.shape for image in images]
    offsets = [(0, 0), (2, 1)]
    _, shape = offseted_shape(shapes, offsets)
    indices = offsets2slice(shapes, offsets, outer_for_stack=stack_axis)
    sections = offsets2slice(
        shapes, offsets, outer_for_stack=stack_axis, fits_convention=True
    )
    for i, (image, index, section) in enumerate(zip(images, indices, sections)):
        canvas = np.full((len(images), *shape) if stack_axis else shape, -1)
        canvas[index] = image
        expected = image[None] if stack_axis else image
        np.testing.assert_array_equal(canvas[index], expected)
        np.testing.assert_array_equal(
            canvas[slice_from_string(section, fits_convention=True)], expected
        )
        if stack_axis:
            assert np.all(canvas[1 - i] == -1)


@pytest.mark.parametrize("shift", [0.5, 1.5, 2.5])
def test_offseted_shape_matches_rounded_pixel_placements(shift: float) -> None:
    shapes = [(3,), (3,)]
    offsets = [(0,), (shift,)]
    regularized, outer = offseted_shape(shapes, offsets)
    assert_allclose(regularized[:, 0], [0, shift])
    placement = int(np.rint(shift))
    assert outer == (3 + placement,)
    canvas = np.zeros((2, *outer))
    for index in offsets2slice(shapes, offsets):
        canvas[index] = np.arange(3)
        np.testing.assert_array_equal(canvas[index], [[0, 1, 2]])

    _, inner = offseted_shape(shapes, offsets, method="inner")
    assert inner == (3 - placement,)
    sections = offsets2slice(shapes, offsets, method="inner", fits_convention=True)
    for index, section in zip(offsets2slice(shapes, offsets, method="inner"), sections):
        image = np.arange(3)
        assert image[index].shape == inner
        np.testing.assert_array_equal(
            image[index], image[slice_from_string(section, fits_convention=True)]
        )


@pytest.mark.parametrize("function", [offseted_shape, offsets2slice])
def test_offset_helpers_reject_touching_images(function) -> None:
    with pytest.raises(ValueError):
        function([(3,), (3,)], [(0,), (3,)], method="inner")


def test_offsets2slice_empty_image_requires_python_output() -> None:
    image = np.empty((0, 3))
    index = offsets2slice([image.shape], [(0, 0)], outer_for_stack=False)[0]
    assert image[index].shape == (0, 3)
    with pytest.raises(ValueError, match="empty"):
        offsets2slice([image.shape], [(0, 0)], fits_convention=True)


@pytest.mark.parametrize("function", [offseted_shape, offsets2slice])
@pytest.mark.parametrize(
    "shapes",
    [
        [(3, 4), (2, 3)],  # More images than offsets.
        [(3, -1)],
        [(3.5, 4)],
        [(np.nan, 4)],
        [(np.inf, 4)],
        [(float(np.iinfo(np.intp).max) + 1, 4)],
        [("three", "four")],
    ],
)
def test_offset_helpers_reject_invalid_shapes(function, shapes) -> None:
    with pytest.raises(ValueError):
        function(shapes, [(0, 0)])


@pytest.mark.parametrize("function", [offseted_shape, offsets2slice])
@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_offset_helpers_reject_nonfinite_offsets(function, invalid: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        function([(3, 4)], [(0, invalid)])


@pytest.mark.parametrize("invalid", ["invalid", [1], [[1, 2]], [np.nan, 1]])
@pytest.mark.parametrize("location_name", ["loc_target", "loc_reference"])
def test_calc_offset_wcs_rejects_invalid_locations(invalid, location_name: str) -> None:
    wcs = pytest.importorskip("astropy.wcs").WCS(naxis=2)
    wcs.pixel_shape = (10, 8)
    with pytest.raises(ValueError):
        calc_offset_wcs(wcs, wcs, **{location_name: invalid})


def test_calc_offset_wcs_requires_shape_only_for_center() -> None:
    wcs = pytest.importorskip("astropy.wcs").WCS(naxis=2)
    with pytest.raises(ValueError, match="pixel_shape"):
        calc_offset_wcs(wcs, wcs)
    assert_allclose(
        calc_offset_wcs(wcs, wcs, loc_target="origin", loc_reference="origin"),
        [0, 0],
    )
    assert_allclose(
        calc_offset_wcs(wcs, wcs, loc_target=[2, 3], loc_reference=[2, 3]),
        [0, 0],
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [("LTM1_1", 2), ("LTM1_1", 0), ("LTM1_2", 1), ("LTM1", 2), ("LTM1_1", "nan")],
)
@pytest.mark.parametrize("as_reference", [False, True])
def test_calc_offset_physical_rejects_nonidentity_ltm(
    key: str, value, as_reference: bool
) -> None:
    fits = pytest.importorskip("astropy.io.fits")
    header = fits.Header({"NAXIS": 2, key: value})
    identity = fits.Header({"NAXIS": 2})
    target, reference = (identity, header) if as_reference else (header, identity)
    with pytest.raises(NotImplementedError, match="identity"):
        calc_offset_physical(target, reference, ignore_ltm=False)
