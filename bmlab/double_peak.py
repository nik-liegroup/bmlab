import numpy as np
from scipy.signal import find_peaks, peak_widths


# A second peak is only reported as a clear double peak if it
#  - rises at least this many noise standard deviations above its
#    surroundings (prominence) - and the main peak at least
#    MIN_MAIN_PEAK_SNR of them, since next to no signal there is
#    nothing to call a double peak,
#  - is at least MIN_WIDTH_SAMPLES wide at half its height, so single
#    noisy points and spikes don't count,
#  - has at least this fraction of the main peak's prominence, and
#  - lies at least this many main-peak FWHMs away from the main peak,
#    i.e. is a separate peak and not just a shoulder or noise on the
#    flank of the main one.
MIN_PROMINENCE_SNR = 8.0
MIN_MAIN_PEAK_SNR = 15.0
MIN_WIDTH_SAMPLES = 3.0
MIN_RELATIVE_PROMINENCE = 0.3
MIN_SEPARATION_FWHM = 1.0


def has_double_peak(region, xdata, ydata,
                    min_prominence_snr=MIN_PROMINENCE_SNR,
                    min_main_peak_snr=MIN_MAIN_PEAK_SNR,
                    min_width_samples=MIN_WIDTH_SAMPLES,
                    min_relative_prominence=MIN_RELATIVE_PROMINENCE,
                    min_separation_fwhm=MIN_SEPARATION_FWHM):
    """
    Checks whether the spectrum shows two clearly separate peaks in the
    given region - e.g. a sample and its surrounding medium inside one
    measurement volume - where fitting a single Lorentzian is not
    meaningful.

    Parameters
    ----------
    region: The section of the data to check, same as in
        fit_lorentz_region()
    xdata: The x-data
    ydata: The y-data

    Returns
    -------
    float
        1.0 if a clear double peak was found, 0.0 if not, NaN if the
        region holds too little valid data to tell.
    """
    try:
        idx_l = np.nanargmin(np.abs(xdata - region[0]))
        idx_r = np.nanargmin(np.abs(xdata - region[1]))
        x = np.asarray(xdata[idx_l:idx_r], dtype=float)
        y = np.asarray(ydata[idx_l:idx_r], dtype=float)
        valid = ~(np.isnan(x) | np.isnan(y))
        y = y[valid]
        if len(y) < 9:
            return np.nan

        # Noise level from the point-to-point differences, which a
        # smooth peak hardly contributes to (MAD -> standard deviation,
        # and the difference of two independent samples has sqrt(2)
        # times the noise). Camera spectra are interpolated along the
        # arc, so their noise is correlated between neighbours and a
        # local-baseline estimate would be far too low.
        diffs = np.diff(y)
        noise = 1.4826 * np.median(
            np.abs(diffs - np.median(diffs))) / np.sqrt(2)
        if not noise > 0:
            return 0.0

        # Light smoothing so single noisy points aren't mistaken for peaks
        smoothed = np.convolve(y, np.ones(3) / 3, mode='same')
        smoothed[0], smoothed[-1] = y[0], y[-1]

        peaks, properties = find_peaks(
            smoothed, prominence=min_prominence_snr * noise)
        if len(peaks) < 2:
            return 0.0

        prominences = properties['prominences']
        order = np.argsort(prominences)[::-1]
        main = order[0]
        if prominences[main] < min_main_peak_snr * noise:
            return 0.0
        widths = peak_widths(smoothed, peaks, rel_height=0.5)[0]
        width = widths[main]
        if width < min_width_samples:
            return 0.0
        for other in order[1:]:
            if prominences[other] < min_relative_prominence \
                    * prominences[main]:
                break
            if widths[other] >= min_width_samples \
                    and abs(peaks[other] - peaks[main]) \
                    >= min_separation_fwhm * width:
                return 1.0
        return 0.0
    except Exception:
        return np.nan
