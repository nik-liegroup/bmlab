import logging

import numpy as np
from scipy.optimize import least_squares, minimize, fmin
from scipy.signal import find_peaks


logger = logging.getLogger(__name__)


class FitError(Exception):
    pass


def lorentz(x, w0, fwhm, intensity):
    with np.errstate(divide='ignore', invalid='ignore'):
        return intensity *\
               ((fwhm / 2) ** 2) / ((x - w0) ** 2 + (fwhm / 2) ** 2)


def fit_lorentz(x, y):
    w0_guess = float(x[np.argmax(y)])
    offset_guess = (y[0] + y[-1]) / 2.
    intensity_guess = np.max(y) - offset_guess
    fwhm_guess = float(2 * np.abs(
        w0_guess - x[np.argmax(y > (offset_guess + intensity_guess / 2))]))
    # Ensure a value larger than zero for the FWHM
    # (it might fail if the peak is not symmetric and its maximum is the
    # first value higher than offset_guess + intensity_guess/2)
    if fwhm_guess <= 0.0:
        fwhm_guess = 2 * (x[-1] - x[0]) / x.shape[0]

    def error(params, xdata, ydata):
        return (ydata
                - lorentz(xdata, *params[0:3])
                - params[3]) ** 2

    bounds_lower = (x[0], (x[-1] - x[0]) / x.shape[0], 0, 0)
    bounds_upper = (x[-1], (x[-1] - x[0]), 2 * np.max(y), 2 * np.max(y))
    bounds = (bounds_lower, bounds_upper)

    opt_result = least_squares(
        error,
        x0=(w0_guess, fwhm_guess, intensity_guess, offset_guess),
        args=(x, y),
        bounds=bounds
    )

    if not opt_result.success:
        raise FitError('Lorentz fit failed.')

    w0, fwhm, intensity, offset = opt_result.x

    return w0, fwhm, intensity, offset


def _smooth_spectrum(y, window=5):
    """
    Lightly smooths the given data with a moving average.
    Only used to get more robust initial guesses for the fits,
    never for the fit itself.
    """
    window = min(window, len(y))
    if window <= 1:
        return np.asarray(y, dtype=float)
    kernel = np.ones(window) / window
    y_padded = np.pad(
        y, (window // 2, window - 1 - window // 2), mode='edge')
    return np.convolve(y_padded, kernel, mode='valid')


def _get_peak_windows(x, nr_peaks, bounds_w0):
    """
    Returns the window in which to search for the initial
    position guess of every peak.
    """
    if bounds_w0 is None:
        return nr_peaks * [np.array((x[0], x[-1]))]
    return [np.clip(np.array(bound, dtype=float), x[0], x[-1])
            for bound in bounds_w0]


def _guess_without_bounds(x, y, nr_peaks):
    """
    Calculates the initial guesses for a multi-peak Lorentz fit
    without w0 bounds. This reproduces the original behavior
    (the most prominent peaks, sorted by position, with a shared
    intensity guess), but instead of giving up when fewer peaks
    than requested are found, it falls back to evenly spaced
    guesses so we still attempt a fit.
    """
    offset_guess = (y[0] + y[-1]) / 2.
    intensity_guess = nr_peaks * [np.max(y) - offset_guess]

    peaks, properties = find_peaks(y, prominence=1)
    idx = np.argsort(properties['prominences'])[::-1]
    if len(idx) >= nr_peaks:
        idx_sort = np.sort(peaks[idx[0:nr_peaks]])
        w0_guess = list(x[idx_sort])
    else:
        # Use the peaks we found and fill up with evenly
        # spaced positions, avoiding the ones already found
        found = sorted(x[peaks[idx]])
        evenly = list(np.linspace(x[0], x[-1], nr_peaks + 2)[1:-1])
        w0_guess = (found + evenly)[:nr_peaks]
        w0_guess.sort()

    return w0_guess, intensity_guess, offset_guess


def _guess_with_bounds(x, y, nr_peaks, bounds_w0, bounds_fwhm):
    """
    Calculates the initial guesses for a multi-peak Lorentz fit
    with w0 bounds. Every peak is seeded from within its own
    bounds window, so that a strong peak in a different window
    cannot shadow a weak one.
    """
    offset_guess = (y[0] + y[-1]) / 2.
    y_s = _smooth_spectrum(y)
    data_range = np.nanmax(y) - np.nanmin(y)
    min_distance = 10 * (x[-1] - x[0]) / x.shape[0]

    # We run peak finding on the smoothed data to get candidates
    # for the peak positions, sorted by descending prominence.
    # The prominence is relative to the data range, so it works
    # independently of the intensity scale.
    peaks, properties = find_peaks(
        y_s, prominence=max(1., 0.05 * data_range))
    candidates = list(peaks[np.argsort(properties['prominences'])[::-1]])

    windows = _get_peak_windows(x, nr_peaks, bounds_w0)

    w0_guess = nr_peaks * [None]

    def fallback_position(window, used):
        # The maximum of the smoothed data within the window,
        # avoiding positions already used
        in_window = (x >= window[0]) & (x <= window[1])
        for w0 in used:
            in_window &= np.abs(x - w0) > min_distance
        if not in_window.any():
            in_window = (x >= window[0]) & (x <= window[1])
        if in_window.any():
            return float(x[np.flatnonzero(in_window)[
                np.argmax(y_s[in_window])]])
        return float(np.mean(window))

    # Peaks with a finite bounds window are seeded from within
    # their own window first, so that a strong peak in a different
    # window cannot shadow a weak one.
    is_windowed = [
        bool(np.all(np.isfinite(np.array(bound, dtype=float))))
        for bound in bounds_w0]

    for i in range(nr_peaks):
        if not is_windowed[i]:
            continue
        window = windows[i]
        # Use the most prominent candidate within the window
        candidate = next((c for c in candidates
                          if window[0] <= x[c] <= window[1]), None)
        if candidate is not None:
            w0_guess[i] = float(x[candidate])
        else:
            used = [w0 for w0 in w0_guess if w0 is not None]
            w0_guess[i] = fallback_position(window, used)
        # Remove candidates close to the used position
        candidates = [c for c in candidates
                      if abs(x[c] - w0_guess[i]) > min_distance]

    # The remaining peaks get the most prominent remaining
    # candidates, sorted by position and matched to the windows
    # sorted by their center (this e.g. keeps the peaks of an
    # anti-Stokes region in the expected order).
    remaining = [i for i in range(nr_peaks) if w0_guess[i] is None]
    positions = []
    for i in remaining:
        if candidates:
            positions.append(float(x[candidates.pop(0)]))
        else:
            used = [w0 for w0 in w0_guess if w0 is not None] + positions
            positions.append(fallback_position(windows[i], used))
    positions.sort()
    remaining.sort(key=lambda i: float(np.mean(windows[i])))
    for i, position in zip(remaining, positions):
        w0_guess[i] = position

    # Check that the initial guesses are within the bounds
    # and estimate the intensity at the guessed positions
    intensity_guess = []
    for i in range(nr_peaks):
        w0_guess[i] = float(np.clip(w0_guess[i], *windows[i]))
        idx = np.argmin(np.abs(x - w0_guess[i]))
        intensity_guess.append(float(max(
            y_s[idx] - offset_guess, 0.05 * data_range)))

    return w0_guess, intensity_guess, offset_guess


def _guess_starting_values(x, y, nr_peaks, bounds_w0, bounds_fwhm):
    """
    Calculates the initial guesses for a multi-peak Lorentz fit.

    Returns
    -------
    w0_guess, fwhm_guess, intensity_guess (lists with one entry
    per peak) and offset_guess
    """
    if bounds_w0 is None:
        w0_guess, intensity_guess, offset_guess = \
            _guess_without_bounds(x, y, nr_peaks)
    else:
        w0_guess, intensity_guess, offset_guess = \
            _guess_with_bounds(x, y, nr_peaks, bounds_w0, bounds_fwhm)

    fwhm_guess = list(10 * (x[-1] - x[0]) / x.shape[0] * np.ones(nr_peaks))
    if bounds_fwhm is not None:
        # Check that the initial guesses are within the bounds
        fwhm_guess = [np.clip(guess, *bounds_fwhm[idx])
                      for idx, guess in enumerate(fwhm_guess)]

    return w0_guess, fwhm_guess, intensity_guess, offset_guess


def _create_lorentz_bounds(x, nr_peaks, bounds_w0, bounds_fwhm):
    """
    Creates the bounds array for a multi-peak Lorentz fit
    with the parameter layout (w0, fwhm, intensity) per peak
    plus a common offset as the last parameter.
    """
    if bounds_w0 is None and bounds_fwhm is None:
        return -np.inf, np.inf

    nr_params = 3 * nr_peaks + 1
    # Lower limits
    bounds_lower = -np.inf * np.ones(nr_params)
    # Upper limits
    bounds_upper = np.inf * np.ones(nr_params)

    # full-width-half-maximum
    # The VIPA spectrometer has an instrument width of
    # approx. 750 MHz for the FOB setup
    # and 180 MHz for the 780 nm setup.
    # This is far higher than the step size,
    # so we limit it to this.
    fwhm_lower_bound = (x[-1] - x[0]) / x.shape[0]

    for i in range(nr_peaks):
        # full-width-half-maximum
        bounds_lower[3 * i + 1] = fwhm_lower_bound
        # intensity
        bounds_lower[3 * i + 2] = 0

    # offset
    bounds_lower[-1] = 0

    # If we have w0 bounds, set them
    if bounds_w0 is not None:
        for i in range(nr_peaks):
            # peak central position
            bounds_lower[3 * i] = bounds_w0[i][0]
            bounds_upper[3 * i] = bounds_w0[i][1]

    if bounds_fwhm is not None:
        for i in range(nr_peaks):
            # peak FWHM
            bounds_lower[3 * i + 1] = bounds_fwhm[i][0]
            bounds_upper[3 * i + 1] = bounds_fwhm[i][1]

    return bounds_lower, bounds_upper


def _clip_to_bounds(x0, bounds):
    """ Clips the initial guesses into the given bounds. """
    if np.isscalar(bounds[0]):
        return x0
    return np.clip(x0, bounds[0], bounds[1])


def _find_degenerate_peaks(x, y, params, bounds, nr_peaks):
    """
    Returns the indices of peaks for which the fit is degenerate,
    i.e. the peak intensity collapsed to (nearly) zero or the peak
    position is pinned at one of its bounds.
    """
    data_range = np.nanmax(y) - np.nanmin(y)
    step = (x[-1] - x[0]) / x.shape[0]
    degenerate = []
    for i in range(nr_peaks):
        w0 = params[3 * i]
        intensity = params[3 * i + 2]
        collapsed = intensity <= 0.02 * data_range
        pinned = False
        if not np.isscalar(bounds[0]):
            lower = bounds[0][3 * i]
            upper = bounds[1][3 * i]
            pinned = (np.isfinite(lower) and abs(w0 - lower) < step) or \
                     (np.isfinite(upper) and abs(w0 - upper) < step)
        if collapsed or pinned:
            degenerate.append(i)
    return degenerate


def _create_retry_x0(x, y, params, degenerate, bounds, nr_peaks):
    """
    Creates alternative starting values for a retry of a degenerate
    multi-peak Lorentz fit. Healthy peaks keep their fitted values,
    degenerate peaks are reseeded at the maximum of the smoothed
    residual within their bounds window.
    """
    x0 = np.array(params, dtype=float)
    data_range = np.nanmax(y) - np.nanmin(y)

    # The residual of the healthy peaks
    model = x0[-1] * np.ones_like(y, dtype=float)
    for i in range(nr_peaks):
        if i not in degenerate:
            model = model + lorentz(x, *x0[3 * i:3 * i + 3])
    residual = _smooth_spectrum(y - model)

    for i in degenerate:
        if np.isscalar(bounds[0]):
            window = np.array((x[0], x[-1]))
        else:
            window = np.clip(
                np.array((bounds[0][3 * i], bounds[1][3 * i])),
                x[0], x[-1])
        in_window = (x >= window[0]) & (x <= window[1])
        if in_window.any():
            idx = np.flatnonzero(in_window)[
                np.argmax(residual[in_window])]
            x0[3 * i] = x[idx]
            x0[3 * i + 2] = max(residual[idx], 0.3 * data_range)
        else:
            x0[3 * i] = np.mean(window)
            x0[3 * i + 2] = 0.3 * data_range
        x0[3 * i + 1] = max(
            0.25 * (window[1] - window[0]),
            10 * (x[-1] - x[0]) / x.shape[0])

    return _clip_to_bounds(x0, bounds)


def fit_multi_lorentz(x, y, nr_peaks, bounds_w0=None, bounds_fwhm=None):
    """
    Fits the sum of nr_peaks Lorentz curves and a common offset
    to the given data.

    If the fit is degenerate (a peak collapsed to zero intensity or
    got pinned at a position bound), it is retried once with
    alternative starting values and the better result is returned.

    Returns
    -------
    Tuples of center, full-width-half-maximum and intensity
    (one entry per peak), and the offset
    """
    w0_guess, fwhm_guess, intensity_guess, offset_guess = \
        _guess_starting_values(x, y, nr_peaks, bounds_w0, bounds_fwhm)

    def error(params, xdata, ydata):
        res = ydata - params[-1]
        for i in range(nr_peaks):
            res = res - lorentz(xdata, *params[3 * i:3 * i + 3])
        return res ** 2

    bounds = _create_lorentz_bounds(x, nr_peaks, bounds_w0, bounds_fwhm)

    x0 = []
    for i in range(nr_peaks):
        x0.extend((w0_guess[i], fwhm_guess[i], intensity_guess[i]))
    x0.append(offset_guess)
    x0 = _clip_to_bounds(np.array(x0), bounds)

    opt_result = least_squares(
        error,
        x0=x0,
        args=(x, y),
        bounds=bounds
    )

    # If the fit is degenerate, we retry once with alternative
    # starting values and keep the better result
    degenerate = _find_degenerate_peaks(
        x, y, opt_result.x, bounds, nr_peaks)
    if degenerate:
        x0_retry = _create_retry_x0(
            x, y, opt_result.x, degenerate, bounds, nr_peaks)
        opt_retry = least_squares(
            error,
            x0=x0_retry,
            args=(x, y),
            bounds=bounds
        )
        if opt_retry.success and (not opt_result.success
                                  or opt_retry.cost < opt_result.cost):
            opt_result = opt_retry

    if not opt_result.success:
        raise FitError('Lorentz fit failed.')

    res = opt_result.x
    w0s = tuple(res[3 * i] for i in range(nr_peaks))
    fwhms = tuple(res[3 * i + 1] for i in range(nr_peaks))
    intens = tuple(res[3 * i + 2] for i in range(nr_peaks))
    offset = res[-1]
    return w0s, fwhms, intens, offset


def fit_double_lorentz(x, y, bounds_w0=None, bounds_fwhm=None):
    return fit_multi_lorentz(x, y, 2, bounds_w0, bounds_fwhm)


def fit_quadruple_lorentz(x, y, bounds_w0=None, bounds_fwhm=None):
    return fit_multi_lorentz(x, y, 4, bounds_w0, bounds_fwhm)


def fit_circle(points):
    """
    Fits a circle to a given set of points. Returnes the center and the radius
    of the cricle. The inital parameters for the fiting process are arbitrarily
    chosen.

    Parameters
    ----------
    points: list of tuples

    Returns
    -------
    center_opt: tuple
                Coordinates of the circle center (x,y)
    radius_opt: float
                Radius of the circle
    """

    center, radius = calculate_exact_circle(points)

    # No need to fit if we only have three points
    if len(points) <= 3:
        return center, radius

    param_guess = [center[0], center[1], radius]
    x_coords = np.array([xy[0] for xy in points])
    y_coords = np.array([xy[1] for xy in points])
    bnds = ((None, None), (None, None), (0.0, None))
    opt_result = minimize(_circle_opt,
                          param_guess,
                          args=(x_coords, y_coords),
                          bounds=bnds)

    return (opt_result['x'][0], opt_result['x'][1]), abs(opt_result['x'][2])


def _circle_opt(c, x_coord, y_coord):
    """
    Cost function to fit a circle to a given set of points.

    Parameters
    ----------
    c: array like
        Parameters to optimize: [x coordinates of circle center,
        y coordinates of circle center, radius of the cricle]
    x_coord:
        x coordinates of the points to fit
    y_coord
        y coordinates of the points to fit

    Returns
    -------
    out: float
        Cost value representing the deviation of the given set of points to
        the circle, which is represented by the parameter vector c.
    """
    return np.sum(
        ((x_coord - c[0]) ** 2
         + (y_coord - c[1]) ** 2
         - c[2] ** 2) ** 2)


def fit_lorentz_region(region, xdata, ydata, nr_peaks=1,
                       bounds_w0=None, bounds_fwhm=None):
    """
    Fits a lorentz or double lorentz fit to the given region

    Parameters
    ----------
    region: The section of the data to fit
    xdata: The x-data
    ydata: The y-data to fit
    nr_peaks: The number of peaks to fit
    bounds_w0: The bounds for the lorentz fit value of the maximum position
    bounds_fwhm: The bounds for the lorentz fit value of the peak width

    Returns
    -------
    center, full-width-half-maximum, intensity and offset
    """
    try:
        idx_l = np.nanargmin(np.abs(xdata - region[0]))
        idx_r = np.nanargmin(np.abs(xdata - region[1]))
        x = xdata[idx_l:idx_r]
        y = ydata[idx_l:idx_r]
        if nr_peaks == 1:
            w0s, fwhms, intensities, offset = fit_lorentz(
                x, y)
        elif nr_peaks == 2:
            w0s, fwhms, intensities, offset = fit_double_lorentz(
                x, y,
                bounds_w0=bounds_w0,
                bounds_fwhm=bounds_fwhm)
        elif nr_peaks == 4:
            w0s, fwhms, intensities, offset = fit_quadruple_lorentz(
                x, y,
                bounds_w0=bounds_w0,
                bounds_fwhm=bounds_fwhm)
        else:
            return
    except Exception:
        w0s = fwhms = intensities = offset = np.nan
    finally:
        return w0s, fwhms, intensities, offset


def calculate_exact_circle(points):
    # We need at least three points to fit a circle
    if len(points) < 3:
        return

    # If there are more than three points given,
    # we use the first and last and one in the center
    points = [points[idx]
              for idx in np.linspace(0, len(points) - 1, 3, dtype=int)]

    # If the first three points are all on the same line,
    # we slightly shift the first point so that we can
    # fit a circle
    if are_points_on_line(points):
        (x, y) = points[0]
        points[0] = (x - 1, y)

    x1 = points[0][0]
    y1 = points[0][1]
    x2 = points[1][0]
    y2 = points[1][1]
    x3 = points[2][0]
    y3 = points[2][1]

    x12 = x1 - x2
    x13 = x1 - x3

    y12 = y1 - y2
    y13 = y1 - y3

    sx13 = x1 ** 2 - x3 ** 2
    sy13 = y1 ** 2 - y3 ** 2

    sx21 = x2 ** 2 - x1 ** 2
    sy21 = y2 ** 2 - y1 ** 2

    x0 = -(sx13 * y12 +
           sy13 * y12 +
           sx21 * y13 +
           sy21 * y13) /\
        (2 * (x12 * y13 - x13 * y12))

    y0 = -(sx13 * x12 +
           sy13 * x12 +
           sx21 * x13 +
           sy21 * x13) /\
        (2 * (y12 * x13 - y13 * x12))

    c = (-x1 ** 2 - y1 ** 2 +
         2 * x0 * x1 + 2 * y0 * y1)

    r = np.sqrt(x0 ** 2 + y0 ** 2 - c)

    return (x0, y0), r


def are_points_on_line(points):
    """
    This function checks if the first three points in the given
    list lay on the same line
    Parameters
    ----------
    points: list of tuples representing points

    Returns
    -------
    boolean
    Whether the given points lay on the same line
    """
    if len(points) < 3:
        return True
    # We calculate a line form the first two points
    # and check whether the remaining points lay on it
    m = (points[0][1] - points[1][1]) /\
        (points[0][0] - points[1][0])
    n = points[0][1] - m * points[0][0]

    for point in points[:3]:
        if point[1] != (m * point[0] + n):
            return False

    return True


def fit_vipa(peaks, setup):
    """
    Fits the VIPA frequency axis

    Parameters
    ----------
    peaks: The peak positions
    setup: The setup parameters

    Returns
    -------
    The fitted VIPA parameters
    """
    # Check that we were given enough peaks
    if peaks is None:
        return
    if len(peaks) < (setup.calibration.num_brillouin_samples * 2 + 2):
        return

    # Calculate the start parameters for the VIPA fit
    r0 = peaks[0] * setup.pixel_size

    v1 = (setup.VIPA_PARAMS[0] +
          setup.VIPA_PARAMS[1] * r0 +
          setup.VIPA_PARAMS[2] * r0 ** 2) /\
         ((setup.vipa.m + setup.vipa.order) * np.pi)
    v2 = (setup.VIPA_PARAMS[1] +
          setup.VIPA_PARAMS[2] * 2 * r0) /\
         ((setup.vipa.m + setup.vipa.order) * np.pi) *\
        setup.pixel_size
    v3 = setup.VIPA_PARAMS[2] /\
        ((setup.vipa.m + setup.vipa.order) * np.pi) *\
        setup.pixel_size ** 2
    vipa_start = np.array([v1, v2, v3, setup.vipa.FSR])

    def error(vipa_params, peaks1, shifts1):
        fsr = vipa_params[3]
        frequencies = VIPA(peaks1, vipa_params) - setup.f0

        # Should match the expected frequencies
        d1 = frequencies - shifts1 - fsr * setup.calibration.orders

        # Should give equal values for Stokes and Anti-Stokes
        d2 = np.array([
            (frequencies[1] - frequencies[0]) -
            (frequencies[-1] - frequencies[-2]),
            (frequencies[2] - frequencies[0]) -
            (frequencies[-1] - frequencies[-3])
        ])

        # Should match the Brillouin shifts
        d3 = np.array([
            shifts1[1] - frequencies[1] + frequencies[0],
            shifts1[2] - frequencies[2] + frequencies[0],
            shifts1[2] - frequencies[5] + frequencies[3],
            shifts1[1] - frequencies[5] + frequencies[4]])

        return np.sum(d1 ** 2) + np.sum(d2 ** 2) + np.sum(d3 ** 2)

    opt_result = fmin(error,
                      vipa_start,
                      args=(peaks, setup.calibration.shifts))

    return opt_result[0], opt_result[1], opt_result[2], opt_result[3]


def VIPA(x, vipa_params):
    """
    Returns the absolute frequency in Hz of a given point
    on the spectrum in pixels.
    Subtract the absolute laser frequency from this value
    to get the relative shift.

    Parameters
    ----------
    x:   [pix]   The point on the spectrum
    vipa_params: Fit parameters of the VIPA fit

    Returns
    -------
    [Hz]   The frequency of the given point on the spectrum
    """
    return 1. / (vipa_params[0] + vipa_params[1] * x + vipa_params[2] * x ** 2)
