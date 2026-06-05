import pathlib

import numpy as np

from bmlab.fits import lorentz, fit_lorentz, fit_circle, \
    fit_double_lorentz, calculate_exact_circle, fit_vipa, VIPA, \
    are_points_on_line, fit_quadruple_lorentz

from bmlab.models.setup import AVAILABLE_SETUPS

import pytest


def test_fit_lorentz():
    # Arrange
    x = np.linspace(0, 30, 100)
    w0 = 15.
    fwhm = 4
    offset = 10.
    intensity = 10.
    y_data = lorentz(x, w0, fwhm, intensity) + offset

    actual_w0, actual_fwhm, actual_intensity, actual_offset =\
        fit_lorentz(x, y_data)
    np.testing.assert_almost_equal(actual_w0, w0, decimal=3)
    np.testing.assert_almost_equal(actual_fwhm, fwhm, decimal=3)
    np.testing.assert_almost_equal(actual_intensity, intensity, decimal=3)
    np.testing.assert_almost_equal(actual_offset, offset, decimal=3)


def test_fit_lorentz_real_image_data():
    """ The data for this test case has been extracted manually
        from the running BMicro application.
    """

    data_dir = pathlib.Path(__file__).parent / 'data'

    region = np.load(data_dir / 'rayleigh_reg0_region.npy')
    xdata = np.load(data_dir / 'rayleigh_reg0_xdata.npy')
    ydata = np.load(data_dir / 'rayleigh_reg0_ydata.npy')

    w0, fwhm, intensity, offset = fit_lorentz(xdata[range(*region)],
                                              ydata[range(*region)])

    assert w0 == pytest.approx(115, 0.2)
    assert fwhm == pytest.approx(6, 0.5)
    assert intensity == pytest.approx(1186, 1)
    assert offset == pytest.approx(47, 1)


def test_fit_double_lorentz():
    # Arrange
    x = np.linspace(0, 60, 200)
    w0_left = 10.
    intensity_left = 8.
    fwhm_left = 2.
    w0_right = 20.
    intensity_right = 12.
    fwhm_right = 4.
    offset = 10.
    y_data = lorentz(x, w0_left, fwhm_left, intensity_left)
    y_data += lorentz(x, w0_right, fwhm_right, intensity_right) + offset

    w0s, fwhms, intens, actual_offset\
        = fit_double_lorentz(x, y_data)

    np.testing.assert_almost_equal(w0s[0], w0_left, decimal=3)
    np.testing.assert_almost_equal(fwhms[0], fwhm_left, decimal=3)
    np.testing.assert_almost_equal(intens[0], intensity_left, decimal=3)

    np.testing.assert_almost_equal(w0s[1], w0_right, decimal=3)
    np.testing.assert_almost_equal(fwhms[1], fwhm_right, decimal=3)
    np.testing.assert_almost_equal(intens[1], intensity_right, decimal=3)

    np.testing.assert_almost_equal(actual_offset, offset, decimal=3)


def test_fit_quadruple_lorentz():
    # Arrange
    x = np.linspace(0, 60, 200)

    w0_0 = 10.
    intensity_0 = 8.
    fwhm_0 = 2.

    w0_1 = 20.
    intensity_1 = 12.
    fwhm_1 = 4.

    w0_2 = 30.
    intensity_2 = 14.
    fwhm_2 = 4.

    w0_3 = 40.
    intensity_3 = 16.
    fwhm_3 = 4.

    offset = 10.

    y_data = lorentz(x, w0_0, fwhm_0, intensity_0)
    y_data += lorentz(x, w0_1, fwhm_1, intensity_1)
    y_data += lorentz(x, w0_2, fwhm_2, intensity_2)
    y_data += lorentz(x, w0_3, fwhm_3, intensity_3) + offset

    w0s, fwhms, intens, actual_offset\
        = fit_quadruple_lorentz(x, y_data)

    np.testing.assert_almost_equal(w0s[0], w0_0, decimal=3)
    np.testing.assert_almost_equal(fwhms[0], fwhm_0, decimal=3)
    np.testing.assert_almost_equal(intens[0], intensity_0, decimal=3)

    np.testing.assert_almost_equal(w0s[1], w0_1, decimal=3)
    np.testing.assert_almost_equal(fwhms[1], fwhm_1, decimal=3)
    np.testing.assert_almost_equal(intens[1], intensity_1, decimal=3)

    np.testing.assert_almost_equal(w0s[2], w0_2, decimal=3)
    np.testing.assert_almost_equal(fwhms[2], fwhm_2, decimal=3)
    np.testing.assert_almost_equal(intens[2], intensity_2, decimal=3)

    np.testing.assert_almost_equal(w0s[3], w0_3, decimal=3)
    np.testing.assert_almost_equal(fwhms[3], fwhm_3, decimal=3)
    np.testing.assert_almost_equal(intens[3], intensity_3, decimal=3)

    np.testing.assert_almost_equal(actual_offset, offset, decimal=3)


def test_fit_double_lorentz_with_bounds():
    # Arrange
    x = np.linspace(0, 60, 500)
    w0_left = 20.
    intensity_left = 8.
    fwhm_left = 4.0
    w0_right = 40.
    intensity_right = 12.
    fwhm_right = 5.0
    offset = 10.
    y_data = lorentz(x, w0_left, fwhm_left, intensity_left)
    y_data += lorentz(x, w0_right, fwhm_right, intensity_right) + offset

    bounds_w0 = ((19, 19.9), (-np.inf, np.inf))

    w0s, fwhms, intens, actual_offset \
        = fit_double_lorentz(x, y_data, bounds_w0)

    np.testing.assert_almost_equal(w0s[0], 19.9, decimal=2)
    np.testing.assert_almost_equal(fwhms[0], fwhm_left, decimal=1)
    np.testing.assert_almost_equal(intens[0], intensity_left, decimal=1)

    np.testing.assert_almost_equal(w0s[1], w0_right, decimal=1)
    np.testing.assert_almost_equal(fwhms[1], fwhm_right, decimal=1)
    np.testing.assert_almost_equal(intens[1], intensity_right, decimal=1)

    np.testing.assert_almost_equal(actual_offset, offset, decimal=1)

    bounds_w0 = ((-np.inf, np.inf), (20.1, 21))

    w0s, fwhms, intens, actual_offset \
        = fit_double_lorentz(x, y_data, bounds_w0)

    np.testing.assert_almost_equal(w0s[0], w0_right, decimal=1)
    np.testing.assert_almost_equal(fwhms[0], fwhm_right, decimal=1)
    np.testing.assert_almost_equal(intens[0], intensity_right, decimal=1)

    np.testing.assert_almost_equal(w0s[1], 20.1, decimal=1)
    np.testing.assert_almost_equal(fwhms[1], fwhm_left, decimal=1)
    np.testing.assert_almost_equal(intens[1], intensity_left, decimal=1)

    np.testing.assert_almost_equal(actual_offset, offset, decimal=1)

    bounds_w0 = ((20.1, 21), (-np.inf, np.inf))
    bounds_fwhm = ((-np.inf, np.inf), (5.1, np.inf))

    w0s, fwhms, intens, actual_offset \
        = fit_double_lorentz(x, y_data,
                             bounds_w0=bounds_w0, bounds_fwhm=bounds_fwhm)

    np.testing.assert_almost_equal(w0s[0], 20.1, decimal=1)
    np.testing.assert_almost_equal(fwhms[0], fwhm_left, decimal=1)
    np.testing.assert_almost_equal(intens[0], intensity_left, decimal=1)

    np.testing.assert_almost_equal(w0s[1], w0_right, decimal=1)
    np.testing.assert_almost_equal(fwhms[1], 5.1, decimal=1)
    np.testing.assert_almost_equal(intens[1], intensity_right, decimal=1)

    np.testing.assert_almost_equal(actual_offset, offset, decimal=1)

    bounds_fwhm = ((-np.inf, np.inf), (5.1, np.inf))

    w0s, fwhms, intens, actual_offset \
        = fit_double_lorentz(x, y_data,
                             bounds_w0=None, bounds_fwhm=bounds_fwhm)

    np.testing.assert_almost_equal(w0s[0], w0_left, decimal=1)
    np.testing.assert_almost_equal(fwhms[0], fwhm_left, decimal=1)
    np.testing.assert_almost_equal(intens[0], intensity_left, decimal=1)

    np.testing.assert_almost_equal(w0s[1], w0_right, decimal=1)
    np.testing.assert_almost_equal(fwhms[1], 5.1, decimal=1)
    np.testing.assert_almost_equal(intens[1], intensity_right, decimal=1)

    np.testing.assert_almost_equal(actual_offset, offset, decimal=1)


def test_fit_quadruple_lorentz_with_bounds():
    # Arrange
    x = np.linspace(0, 60, 200)

    w0_0 = 10.
    intensity_0 = 8.
    fwhm_0 = 2.

    w0_1 = 20.
    intensity_1 = 12.
    fwhm_1 = 4.

    w0_2 = 30.
    intensity_2 = 14.
    fwhm_2 = 4.

    w0_3 = 40.
    intensity_3 = 16.
    fwhm_3 = 4.

    offset = 10.

    y_data = lorentz(x, w0_0, fwhm_0, intensity_0)
    y_data += lorentz(x, w0_1, fwhm_1, intensity_1)
    y_data += lorentz(x, w0_2, fwhm_2, intensity_2)
    y_data += lorentz(x, w0_3, fwhm_3, intensity_3) + offset

    bounds_w0 = (
        (10.1, 11),
        (-np.inf, np.inf),
        (29.0, 29.9),
        (-np.inf, np.inf)
    )
    bounds_fwhm = (
        (-np.inf, np.inf),
        (4.1, np.inf),
        (-np.inf, np.inf),
        (-np.inf, np.inf)
    )

    w0s, fwhms, intens, actual_offset\
        = fit_quadruple_lorentz(x, y_data,
                                bounds_w0=bounds_w0, bounds_fwhm=bounds_fwhm)

    np.testing.assert_almost_equal(w0s[0], 10.1, decimal=1)
    np.testing.assert_almost_equal(fwhms[0], fwhm_0, decimal=1)
    np.testing.assert_almost_equal(intens[0], intensity_0, decimal=1)

    np.testing.assert_almost_equal(w0s[1], w0_1, decimal=1)
    np.testing.assert_almost_equal(fwhms[1], 4.1, decimal=1)
    np.testing.assert_almost_equal(intens[1], intensity_1, decimal=1)

    np.testing.assert_almost_equal(w0s[2], 29.9, decimal=1)
    np.testing.assert_almost_equal(fwhms[2], fwhm_2, decimal=1)
    np.testing.assert_almost_equal(intens[2], intensity_2, decimal=1)

    np.testing.assert_almost_equal(w0s[3], w0_3, decimal=1)
    np.testing.assert_almost_equal(fwhms[3], fwhm_3, decimal=1)
    np.testing.assert_almost_equal(intens[3], intensity_3, decimal=1)

    np.testing.assert_almost_equal(actual_offset, offset, decimal=1)


def test_fit_double_lorentz_weak_peak_with_noise():
    """
    A strong narrow peak next to a weak broad peak with noise:
    global peak finding used to seed the second peak on a noise
    spike, collapsing its fit. The per-window seeding finds it.
    """
    rng = np.random.RandomState(0)
    x = np.linspace(0, 60, 400)
    y_data = lorentz(x, 20, 2, 200) + lorentz(x, 42, 8, 12) + 150
    y_data += rng.normal(0, 6, x.shape)

    bounds_w0 = ((15, 25), (30, 55))

    w0s, fwhms, intens, offset = fit_double_lorentz(x, y_data, bounds_w0)

    assert abs(w0s[0] - 20) < 0.5
    assert abs(w0s[1] - 42) < 2.0
    assert intens[0] > 100
    assert intens[1] > 5


def test_fit_double_lorentz_peak_near_region_edge():
    """
    A peak whose maximum lies beyond the region edge is invisible
    to peak finding, but is recovered by the per-window seeding.
    """
    rng = np.random.RandomState(2)
    x = np.linspace(0, 59, 400)
    y_data = lorentz(x, 20, 2, 100) + lorentz(x, 59.5, 5, 40) + 50
    y_data += rng.normal(0, 2, x.shape)

    bounds_w0 = ((15, 25), (50, 61))

    w0s, fwhms, intens, offset = fit_double_lorentz(x, y_data, bounds_w0)

    assert abs(w0s[0] - 20) < 0.5
    assert abs(w0s[1] - 59.5) < 1.0
    assert intens[1] > 20


def test_fit_double_lorentz_single_peak_graceful():
    """
    With fewer detectable peaks than requested, the fit used to
    return None (silently turning the result into NaN upstream).
    It should return a valid result instead.
    """
    x = np.linspace(0, 60, 200)
    y_data = lorentz(x, 30, 5, 10) + 5

    result = fit_double_lorentz(x, y_data)

    assert result is not None
    w0s, fwhms, intens, offset = result
    assert len(w0s) == 2
    assert np.all(np.isfinite(w0s))
    assert np.all(np.isfinite(intens))


def test_fit_quadruple_lorentz_weak_peaks_with_noise():
    rng = np.random.RandomState(3)
    x = np.linspace(0, 60, 400)
    y_data = lorentz(x, 10, 2, 150)\
        + lorentz(x, 20, 6, 10)\
        + lorentz(x, 40, 6, 12)\
        + lorentz(x, 50, 2, 180) + 100
    y_data += rng.normal(0, 4, x.shape)

    bounds_w0 = ((5, 15), (15, 30), (30, 45), (45, 55))

    w0s, fwhms, intens, offset = \
        fit_quadruple_lorentz(x, y_data, bounds_w0)

    assert abs(w0s[0] - 10) < 0.5
    assert abs(w0s[1] - 20) < 2.0
    assert abs(w0s[2] - 40) < 2.0
    assert abs(w0s[3] - 50) < 0.5
    assert intens[1] > 4
    assert intens[2] > 4


def test_fit_quadruple_lorentz_too_few_peaks_graceful():
    x = np.linspace(0, 60, 200)
    y_data = lorentz(x, 20, 5, 10) + lorentz(x, 40, 5, 10) + 5

    result = fit_quadruple_lorentz(x, y_data)

    assert result is not None
    w0s, fwhms, intens, offset = result
    assert len(w0s) == 4
    assert np.all(np.isfinite(w0s))


def test_find_degenerate_peaks():
    from bmlab.fits import _find_degenerate_peaks, _create_lorentz_bounds

    x = np.linspace(0, 60, 200)
    y_data = lorentz(x, 20, 4, 10) + lorentz(x, 40, 4, 10) + 5
    bounds = _create_lorentz_bounds(x, 2, ((15, 25), (35, 45)), None)

    # Healthy fit
    params = np.array([20, 4, 10, 40, 4, 10, 5])
    assert _find_degenerate_peaks(x, y_data, params, bounds, 2) == []

    # Second peak intensity collapsed
    params = np.array([20, 4, 10, 40, 4, 0.01, 5])
    assert _find_degenerate_peaks(x, y_data, params, bounds, 2) == [1]

    # First peak pinned at its lower position bound
    params = np.array([15.0, 4, 10, 40, 4, 10, 5])
    assert 0 in _find_degenerate_peaks(x, y_data, params, bounds, 2)


def test_create_retry_x0_reseeds_collapsed_peak():
    from bmlab.fits import _create_retry_x0, _create_lorentz_bounds

    x = np.linspace(0, 60, 600)
    y_data = lorentz(x, 20, 2, 20) + lorentz(x, 40, 6, 8) + 10
    bounds = _create_lorentz_bounds(x, 2, ((15, 25), (30, 50)), None)

    # The first fit found the first peak,
    # but the second peak collapsed
    params = np.array([20, 2, 20, 30.0, 2, 0.0, 10])

    x0 = _create_retry_x0(x, y_data, params, [1], bounds, 2)

    # The healthy peak keeps its values
    np.testing.assert_allclose(x0[0:3], params[0:3])
    # The collapsed peak is reseeded near the residual maximum
    assert abs(x0[3] - 40) < 1
    assert x0[5] > 0


def test_circle_fit():
    expect_r = 550
    expect_c = (-200, -220)
    n_test_data_points = 6
    noise_strength = 15
    np.random.seed(1)
    x_noise = np.random.random(n_test_data_points) * noise_strength
    y_noise = np.random.random(n_test_data_points) * noise_strength

    test_points_noise = [(expect_r * np.cos(phi) + x_noise[i] + expect_c[0],
                          expect_r * np.sin(phi) + y_noise[i] + expect_c[1])
                         for i, phi in enumerate(
        np.linspace(0.1, np.pi / 2, n_test_data_points))
    ]

    actual_c_noise, actual_r_noise = fit_circle(test_points_noise)

    np.testing.assert_allclose(actual_c_noise, expect_c, rtol=0.05)
    np.testing.assert_allclose(actual_r_noise, expect_r, rtol=0.05)

    test_points = [(expect_r * np.cos(phi) + 0 * x_noise[i] + expect_c[0],
                    expect_r * np.sin(phi) + 0 * y_noise[i] + expect_c[1])
                   for i, phi in enumerate(
        np.linspace(0.1, np.pi / 2, int(n_test_data_points / 2) + 1))
    ]

    actual_c, actual_r = fit_circle(test_points)
    np.testing.assert_allclose(actual_c, expect_c, rtol=1e-2)
    np.testing.assert_allclose(actual_r, expect_r, rtol=1e-2)

    # Test that radius is always positive
    test_points_2 = [(np.sqrt(2), 0), (1, 1), (0, np.sqrt(2))]
    actual_c_2, actual_r_2 = fit_circle(test_points_2)

    np.testing.assert_allclose(actual_c_2, (0, 0), rtol=0.001, atol=0.0001)
    assert (actual_r_2 > 0)
    np.testing.assert_allclose(actual_r_2, np.sqrt(2),
                               rtol=0.0001, atol=0.000001)

    test_points_3 = [
        (10.0, 0.0),
        (7.0710678118654755, 7.0710678118654755),
        (6.123233995736766e-16, 10.0)
    ]
    actual_c_3, actual_r_3 = fit_circle(test_points_3)

    np.testing.assert_allclose(actual_c_3, (0, 0), rtol=0.001, atol=0.0001)
    assert (actual_r_3 > 0)
    np.testing.assert_allclose(actual_r_3, 10, rtol=0.0001, atol=0.000001)


def test_calculate_exact_circle():
    points = [[1, 0], [0, 1], [-1, 0]]
    center, radius = calculate_exact_circle(points)

    np.testing.assert_allclose(center, [0, 0], rtol=1e-2)
    np.testing.assert_allclose(radius, 1, rtol=1e-2)

    points = [[-1, -2], [-2, -1], [-3, -2]]
    center, radius = calculate_exact_circle(points)

    np.testing.assert_allclose(center, [-2, -2], rtol=1e-2)
    np.testing.assert_allclose(radius, 1, rtol=1e-2)

    points = [[8, -2], [-2, 8], [-12, -2]]
    center, radius = calculate_exact_circle(points)

    np.testing.assert_allclose(center, [-2, -2], rtol=1e-2)
    np.testing.assert_allclose(radius, 10, rtol=1e-2)


def test_are_points_on_line():
    assert are_points_on_line([(0, 0), (1, 1), (2, 2)])
    assert are_points_on_line([(0, 0), (1, 1)])
    assert not are_points_on_line([(0, 0), (1, 1), (3, 2)])
    assert not are_points_on_line([(0, 0), (1, 1), (3, 2), (4, 4)])
    assert are_points_on_line([(0, 0), (1, 1), (2, 2), (3, 4)])


def test_fit_vipa():
    setup = AVAILABLE_SETUPS[0]
    peaks = np.array([
        84.2957567375179,
        147.651886970066,
        166.035559534916,
        229.678232333124,
        244.118149639528,
        287.316083765756
    ])

    vipa_params = fit_vipa(peaks, setup)

    # Values extracted from the previous Matlab version
    vipa_params_expected =\
        np.array([
            2.602626743299098e-15, -2.565391389072813e-22,
            -6.473779072623057e-25, 14.89190812511701e+9
        ])

    actual = VIPA(peaks, vipa_params) - setup.f0
    expected = VIPA(peaks, vipa_params_expected) - setup.f0

    # Values are in GHz, differences below 5 MHz should be good
    np.testing.assert_allclose(actual, expected, atol=5e3)
