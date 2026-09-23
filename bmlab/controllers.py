import logging

import numpy as np
from scipy.signal import medfilt2d, find_peaks
from skimage.measure import label
from skimage.morphology import closing, disk

from math import floor

from bmlab import Session
from bmlab.fits import fit_vipa, VIPA, fit_lorentz_region
from bmlab.image import extract_lines_along_arc, find_max_in_radius
from bmlab.export import FluorescenceExport, \
    FluorescenceCombinedExport, BrillouinExport, \
    SurfaceExport, OverviewBrightfieldExport

import warnings

logger = logging.getLogger(__name__)


class ExtractionController(object):

    def __init__(self):
        self.session = Session.get_instance()
        return

    def add_point(self, calib_key, point):
        time = self.session.get_calibration_time(calib_key)
        if time is None:
            return
        em = self.session.extraction_model()
        em.add_point(calib_key, time, *point)

    def set_point(self, calib_key, index, point):
        time = self.session.get_calibration_time(calib_key)
        if time is None:
            return
        em = self.session.extraction_model()
        em.set_point(calib_key, index, time, *point)

    def optimize_points(self, calib_key, radius=10):
        em = self.session.extraction_model()

        imgs = self.session.get_calibration_image(calib_key)
        if imgs is None:
            return
        img = np.nanmean(imgs, axis=0)
        img = medfilt2d(img)

        points = em.get_points(calib_key)
        time = em.get_time(calib_key)
        em.clear_points(calib_key)

        for p in points:
            new_point = find_max_in_radius(img, p, radius)
            # Warning: x-axis in imshow is 1-axis in img, y-axis is 0-axis
            em.add_point(
                calib_key, time, new_point[0], new_point[1])

    def find_points_all(self):
        calib_keys = self.session.get_calib_keys()

        if not calib_keys:
            return

        for calib_key in calib_keys:
            self.find_points(calib_key)

    def find_points(self, calib_key, min_height=10,
                    min_area=20, max_distance=50):
        em = self.session.extraction_model()

        imgs = self.session.get_calibration_image(calib_key)
        if imgs is None:
            return
        img = np.nanmean(imgs, axis=0)
        time = self.session.get_calibration_time(calib_key)
        if time is None:
            return

        # Account for binning for default values
        disc_size = 10
        binning = self.session.get_calibration_binning_factor(calib_key)
        if binning:
            max_distance = max_distance / binning
            disc_size = disc_size / binning
            min_area = min_area / (binning**2)

        img = medfilt2d(img)
        # This is the background level
        threshold = np.median(img)

        # Try to find a signal dependent estimate
        # for the minimal peak height.
        # Discard all values smaller than the background noise
        # plus a minimal peak height
        img_peaks = img[img > threshold + min_height]
        # Calculate the signal dependent peak threshold
        height = (np.nanmean(img_peaks) + threshold) / 2
        img_closed = closing(img > height, disk(disc_size))

        # Find all peaks higher than the min_height
        image_label, num = label(img_closed, return_num=True)

        all_peaks = []
        for region in range(1, num + 1):
            # Mask of everything but the peak
            mask = (image_label != region)
            # Set everything but the peak to zero
            tmp = img.copy()
            tmp[mask] = 0

            # Discard all peaks with an area that is too small
            if np.sum(np.logical_not(mask)) >= min_area:
                # Find indices of peak maximum
                ind = np.unravel_index(np.argmax(tmp, axis=None), tmp.shape)

                all_peaks.append(ind)

        # Filter found peaks
        p0 = (0, img.shape[1])
        p1 = (img.shape[0], 0)
        peaks = filter(
            lambda peak:
            self.distance_point_to_line(peak, p0, p1) < max_distance,
            all_peaks)

        # Add found peaks to model
        em.set_points(calib_key, time, list(peaks))

    def distance_point_to_line(self, point, line0, line1):
        return abs(
            (line1[1] - line0[1]) * (line0[0] - point[0]) -
            (line0[1] - point[1]) * (line1[0] - line0[0]))\
               / np.sqrt((line1[1] - line0[1])**2 + (line1[0] - line0[0])**2)


class ImageController(object):

    def __init__(self, model, get_image, get_time, get_exposure):
        self.session = Session.get_instance()
        self.model = model
        self.get_image = get_image
        self.get_time = get_time
        self.get_exposure = get_exposure

    def extract_spectra(self, image_key, frame_num=None):
        em = self.session.extraction_model()
        if not em:
            return None, None, None
        time = self.get_time(image_key)
        arc = em.get_arc_by_time(time)
        if arc.size == 0:
            return None, None, None

        imgs = self.get_image(image_key)
        if imgs is None:
            return None, None, None
        if frame_num is not None:
            imgs = imgs[frame_num:frame_num+1]

        # Extract values from *all* frames in the current calibration
        spectra = []
        for img in imgs:
            values_by_img = extract_lines_along_arc(
                img,
                arc
            )
            spectra.append(values_by_img)

        exposure = self.get_exposure(image_key)
        times = exposure * np.arange(len(imgs)) + time

        intensities = np.nanmean(imgs, axis=(1, 2))

        # We only set the spectra if we extracted all
        if frame_num is None:
            self.model().set_spectra(image_key, spectra)
        return spectra, times, intensities


class CalibrationController(ImageController):

    def __init__(self, *args, **kwargs):
        session = Session.get_instance()
        super(CalibrationController, self).__init__(
            model=session.calibration_model,
            get_image=session.get_calibration_image,
            get_time=session.get_calibration_time,
            get_exposure=session.get_calibration_exposure
        )
        return

    def find_peaks(self, calib_key, min_prominence=15,
                   num_brillouin_samples=2, min_height=15):
        spectra, _, _ = self.extract_spectra(calib_key)
        if spectra is None:
            return
        spectrum = np.mean(spectra, axis=0)
        # This is the background value
        base = np.nanmedian(spectrum)
        peaks, properties = find_peaks(
            spectrum, prominence=min_prominence, width=True,
            height=min_height+base)

        # Number of peaks we are searching for
        # (2 Rayleigh + 2 times number calibration samples)
        num_peaks = 2 + 2 * num_brillouin_samples

        # Check if we found enough peak candidates
        if len(peaks) < num_peaks:
            # If we didn't find enough peaks, we try again
            # without a minimum height
            peaks, properties = find_peaks(
                spectrum, prominence=min_prominence, width=True)
            # If there a still too few, we give up
            if len(peaks) < num_peaks:
                return

        # We need to identify the position between the
        # Stokes and Anti-Stokes Brillouin peaks

        # In case there are just enough peaks,
        # we use the position in the middle:
        if len(peaks) == num_peaks:
            idx = int(num_peaks / 2)
            center = np.mean(peaks[idx - 1:idx + 1])
        # Otherwise we use the center of mass as the middle
        else:
            # Set everything below the background value to zero,
            # so it does not affect the center calculation
            spectrum[spectrum < base] = 0
            # Calculate the center of mass
            center = np.nansum(spectrum * range(1, len(spectrum) + 1))\
                / np.nansum(spectrum)

            # Check that we have enough peaks on both sides of the center
            num_peaks_right = len(peaks[peaks > center])
            num_peaks_left = len(peaks[peaks <= center])

            # If not enough peaks on the right, shift center to the left
            if num_peaks_right < (num_brillouin_samples + 1):
                center = np.mean(
                    peaks[-(num_brillouin_samples + 2):-num_brillouin_samples]
                )
            # If not enough peaks on the left, shift center to the right
            elif num_peaks_left < (num_brillouin_samples + 1):
                center = np.mean(
                    peaks[num_brillouin_samples:num_brillouin_samples + 2]
                )

        num_peaks_left = len(peaks[peaks <= center])

        indices_brillouin = range(
            num_peaks_left - num_brillouin_samples,
            num_peaks_left + num_brillouin_samples
        )
        indices_rayleigh = [
            num_peaks_left - num_brillouin_samples - 1,
            num_peaks_left + num_brillouin_samples
        ]

        def peak_to_region(i):
            r = (
                        peaks[i]
                        + properties['widths'][i] * np.array((-4, 4))
                ).astype(int)
            r[r > len(spectrum)] = len(spectrum)
            return tuple(r)

        regions_brillouin = list(map(peak_to_region, indices_brillouin))
        # Merge the Brillouin regions if necessary
        if num_brillouin_samples > 1:
            regions_brillouin = [
                (regions_brillouin[0][0],
                 regions_brillouin[num_brillouin_samples - 1][1]),
                (regions_brillouin[num_brillouin_samples][0],
                 regions_brillouin[-1][1]),
            ]

        regions_rayleigh = map(peak_to_region, indices_rayleigh)

        cm = self.session.calibration_model()
        # Add Brillouin regions
        for i, region in enumerate(regions_brillouin):
            # We use "set_brillouin_region" here so overlapping
            # regions don't get merged
            cm.set_brillouin_region(calib_key, i, region)
        # Add Rayleigh regions
        for i, region in enumerate(regions_rayleigh):
            # We use "set_brillouin_region" here so overlapping
            # regions don't get merged
            cm.set_rayleigh_region(calib_key, i, region)

    def calibrate(self, calib_key, count=None, max_count=None):

        setup = self.session.setup
        em = self.session.extraction_model()
        cm = self.session.calibration_model()

        if not calib_key\
                or not setup\
                or not em\
                or not cm:
            if max_count is not None:
                max_count.value = -1
            return

        spectra, _, _ = self.extract_spectra(calib_key)
        time = self.session.get_calibration_time(calib_key)

        if spectra is None or len(spectra) == 0:
            if max_count is not None:
                max_count.value = -1
            return

        self.fit_rayleigh_regions(calib_key)
        self.fit_brillouin_regions(calib_key)

        vipa_params = []
        frequencies = []

        if max_count is not None:
            max_count.value += len(spectra)

        for frame_num, spectrum in enumerate(spectra):
            peaks = cm.get_sorted_peaks(calib_key, frame_num)

            params = fit_vipa(peaks, setup)
            if params is None:
                continue
            vipa_params.append(params)
            xdata = np.arange(len(spectrum))

            frequencies.append(VIPA(xdata, params) - setup.f0)
            if count is not None:
                count.value += 1

        cm.set_vipa_params(calib_key, vipa_params)
        cm.set_frequencies(calib_key, time, frequencies)

        evm = self.session.evaluation_model()
        if evm is not None:
            evm.invalidate_results()

    def clear_calibration(self, calib_key):
        cm = self.session.calibration_model()
        if not cm:
            return

        cm.clear_brillouin_fits(calib_key)
        cm.clear_rayleigh_fits(calib_key)
        cm.clear_frequencies(calib_key)
        cm.clear_vipa_params(calib_key)

        evm = self.session.evaluation_model()
        if evm is not None:
            evm.invalidate_results()

    def fit_rayleigh_regions(self, calib_key):
        cm = self.session.calibration_model()
        spectra = cm.get_spectra(calib_key)
        regions = cm.get_rayleigh_regions(calib_key)

        cm.clear_rayleigh_fits(calib_key)
        for frame_num, spectrum in enumerate(spectra):
            for region_key, region in enumerate(regions):
                xdata = np.arange(len(spectrum))
                w0, fwhm, intensity, offset, *_ = \
                    fit_lorentz_region(region, xdata, spectrum)
                cm.add_rayleigh_fit(calib_key, region_key, frame_num,
                                    w0, fwhm, intensity, offset)

    def fit_brillouin_regions(self, calib_key):
        cm = self.session.calibration_model()
        spectra = cm.get_spectra(calib_key)
        regions = cm.get_brillouin_regions(calib_key)
        setup = self.session.setup
        if not setup:
            return

        cm.clear_brillouin_fits(calib_key)
        for frame_num, spectrum in enumerate(spectra):
            for region_key, region in enumerate(regions):
                xdata = np.arange(len(spectrum))
                w0s, fwhms, intensities, offset, *_ = \
                    fit_lorentz_region(
                        region,
                        xdata,
                        spectrum,
                        setup.calibration.num_brillouin_samples
                    )
                cm.add_brillouin_fit(calib_key, region_key, frame_num,
                                     w0s, fwhms, intensities, offset)

    def expected_frequencies(self, calib_key=None, current_frame=None):
        cm = self.session.calibration_model()

        if calib_key not in cm.vipa_params or \
                current_frame > len(cm.vipa_params[calib_key]) - 1:
            return None

        return self.session.setup.calibration.shifts \
            + self.session.setup.calibration.orders \
            * cm.vipa_params[calib_key][current_frame][3]


class PeakSelectionController(object):

    def __init__(self):
        self.session = Session.get_instance()
        return

    def add_brillouin_region_frequency(self, region_frequency):
        psm = self.session.peak_selection_model()
        psm.add_brillouin_region(region_frequency)

    def add_rayleigh_region_frequency(self, region_frequency):
        psm = self.session.peak_selection_model()
        psm.add_rayleigh_region(region_frequency)


def _combine_spectra(spectra, frequencies):
    """
    'sum'-mode helper (see EvaluationModel.evaluation_mode): interpolates
    every frame's spectrum onto a common frequency axis - frequencies[0],
    the reference - and sums them elementwise, the "sum-then-fit"
    counterpart to fitting each frame separately (closer to a sufficient
    statistic for the shared peak position under Poisson counting
    statistics than averaging separately-fit shifts, especially at low
    SNR).

    Frequencies are not guaranteed to be increasing along the pixel axis
    (the VIPA dispersion direction isn't fixed by this code), so each
    frame's (frequency, spectrum) pair is sorted by frequency before
    calling np.interp, which requires an increasing `xp`. NaNs in a
    frame's spectrum or frequency axis are masked out per-frame before
    interpolating (the same approach fit_lorentz_region uses), and the
    reference-axis points a frame's data doesn't cover are left NaN
    (via np.interp's left/right) rather than extrapolated - so the sum
    at each reference-axis point is the sum over only the frames that
    actually had a valid value there (np.nansum semantics), and a point
    with no valid data in *any* frame stays NaN instead of becoming a
    fabricated zero.

    Parameters
    ----------
    spectra: list of array-like
        One spectrum per frame.
    frequencies: list of array-like
        One frequency axis per frame, same length as `spectra`.

    Returns
    -------
    summed_spectrum: np.ndarray
        The elementwise sum, across frames, of every frame's spectrum
        interpolated onto `reference_frequencies`.
    reference_frequencies: np.ndarray
        The common frequency axis (frequencies[0]) every frame was
        interpolated onto.
    """
    reference_frequencies = np.asarray(frequencies[0], dtype=float)
    interpolated = np.full(
        (len(spectra), reference_frequencies.size), np.nan)
    for frame_num, (spectrum, freq) in enumerate(zip(spectra, frequencies)):
        freq = np.asarray(freq, dtype=float)
        spectrum = np.asarray(spectrum, dtype=float)
        mask = ~(np.isnan(freq) | np.isnan(spectrum))
        freq = freq[mask]
        spectrum = spectrum[mask]
        if freq.size == 0:
            continue
        order = np.argsort(freq)
        freq = freq[order]
        spectrum = spectrum[order]
        interpolated[frame_num] = np.interp(
            reference_frequencies, freq, spectrum,
            left=np.nan, right=np.nan)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore', message='All-NaN axis encountered')
        summed_spectrum = np.nansum(interpolated, axis=0)
    # np.nansum silently returns 0 where every frame is NaN at that
    # pixel - restore NaN there instead of a fabricated zero.
    all_nan = np.all(np.isnan(interpolated), axis=0)
    summed_spectrum[all_nan] = np.nan

    return summed_spectrum, reference_frequencies


class EvaluationController(ImageController):

    def __init__(self, *args, **kwargs):
        session = Session.get_instance()
        super(EvaluationController, self).__init__(
            model=session.evaluation_model,
            get_image=session.get_payload_image,
            get_time=session.get_payload_time,
            get_exposure=session.get_payload_exposure
        )
        return

    def set_nr_brillouin_peaks(self, nr_brillouin_peaks):
        evm = self.session.evaluation_model()
        if not evm:
            return
        evm.setNrBrillouinPeaks(nr_brillouin_peaks)

    def set_bounds(self, bounds):
        evm = self.session.evaluation_model()
        if not evm:
            return
        evm.bounds_w0 = bounds

    def set_bounds_fwhm(self, bounds):
        evm = self.session.evaluation_model()
        if not evm:
            return
        evm.bounds_fwhm = bounds

    def evaluate(self, abort=None, count=None, max_count=None):
        em = self.session.extraction_model()
        if not em:
            if max_count is not None:
                max_count.value = -1
            return

        cm = self.session.calibration_model()
        if not cm:
            if max_count is not None:
                max_count.value = -1
            return

        pm = self.session.peak_selection_model()
        if not pm:
            if max_count is not None:
                max_count.value = -1
            return

        evm = self.session.evaluation_model()
        if not evm:
            if max_count is not None:
                max_count.value = -1
            return

        image_keys = self.session.get_image_keys(True)

        if max_count is not None:
            max_count.value = len(image_keys)

        brillouin_regions = pm.get_brillouin_regions()
        rayleigh_regions = pm.get_rayleigh_regions()

        resolution = self.session.get_payload_resolution()

        if not image_keys:
            if max_count is not None:
                max_count.value = -1
            return

        # Get first spectrum to find number of images
        spectra, _, _ = self.extract_spectra(image_keys[0])

        if not spectra:
            if max_count is not None:
                max_count.value = -1
            return

        # We create a variable for this value here,
        # so changing nr_brillouin_peaks during evaluation
        # does not create issues
        nr_brillouin_peaks = evm.nr_brillouin_peaks
        evm.initialize_results_arrays({
            # measurement points in x direction
            'dim_x': resolution[0],
            # measurement points in y direction
            'dim_y': resolution[1],
            # measurement points in z direction
            'dim_z': resolution[2],
            # number of images per measurement point
            'nr_images': len(spectra),
            # number of Brillouin regions
            'nr_brillouin_regions': len(brillouin_regions),
            # number of peaks to fit per region
            'nr_brillouin_peaks': nr_brillouin_peaks,
            # number of Rayleigh regions
            'nr_rayleigh_regions': len(rayleigh_regions),
        })

        # Initialize the Rayleigh shift
        # used for compensating drifts
        rayleigh_peak_initial =\
            np.nan * np.ones((len(spectra), len(rayleigh_regions), 1))
        # Loop over all measurement positions
        for idx, image_key in enumerate(image_keys):

            if count is not None:
                count.value += 1

            # Calculate the indices for the given key
            (ind_x, ind_y, ind_z) =\
                self.get_indices_from_key(resolution, image_key)

            if (abort is not None) and abort.value:
                calculate_derived_values()
                if max_count is not None:
                    max_count.value = -1
                return
            spectra, times, intensities =\
                self.extract_spectra(image_key)
            if spectra is None:
                continue
            evm.results['time'][ind_x, ind_y, ind_z, :, 0, 0] =\
                times
            evm.results['intensity'][ind_x, ind_y, ind_z, :, 0, 0] =\
                intensities

            frequencies = cm.get_frequencies_by_time(times)
            # If we don't have frequency axis, we cannot evaluate on it
            if frequencies is None:
                continue
            frequencies = list(frequencies)
            evm.set_frequencies(image_key, frequencies)

            for region_key, region in enumerate(brillouin_regions):
                results = self.fit_spectra(spectra, frequencies, region)
                for frame_num, _ in enumerate(spectra):
                    ind = (ind_x, ind_y, ind_z,
                           frame_num, region_key, 0)
                    evm.results['brillouin_peak_position_f'][ind] =\
                        results[frame_num][0]
                    evm.results['brillouin_peak_fwhm_f'][ind] =\
                        results[frame_num][1]
                    evm.results['brillouin_peak_intensity'][ind] =\
                        results[frame_num][2]
                    evm.results['brillouin_peak_offset'][ind] =\
                        results[frame_num][3]
                    evm.results['brillouin_peak_snr'][ind] =\
                        results[frame_num][4]
                    evm.results['brillouin_peak_nrmse'][ind] =\
                        results[frame_num][5]
                    evm.results[
                        'brillouin_peak_center_uncertainty'][ind] =\
                        results[frame_num][6]

                if evm.evaluation_mode == 'sum':
                    combined = self.fit_spectrum_combined(
                        spectra, frequencies, region)
                    ind_c = (ind_x, ind_y, ind_z,
                             slice(None), region_key, 0)
                    evm.results[
                        'brillouin_peak_position_f_combined'][ind_c] =\
                        combined[0]
                    evm.results[
                        'brillouin_peak_fwhm_f_combined'][ind_c] =\
                        combined[1]
                    evm.results[
                        'brillouin_peak_intensity_combined'][ind_c] =\
                        combined[2]
                    evm.results[
                        'brillouin_peak_offset_combined'][ind_c] =\
                        combined[3]
                    evm.results[
                        'brillouin_peak_snr_combined'][ind_c] =\
                        combined[4]
                    evm.results[
                        'brillouin_peak_nrmse_combined'][ind_c] =\
                        combined[5]
                    evm.results[
                        'brillouin_peak_center_uncertainty_combined'][
                        ind_c] = combined[6]

            for region_key, region in enumerate(rayleigh_regions,):
                results = self.fit_spectra(spectra, frequencies, region)
                for frame_num, _ in enumerate(spectra):
                    ind = (ind_x, ind_y, ind_z, frame_num, region_key)
                    evm.results['rayleigh_peak_position_f'][ind] =\
                        results[frame_num][0]
                    evm.results['rayleigh_peak_fwhm_f'][ind] =\
                        results[frame_num][1]
                    evm.results['rayleigh_peak_intensity'][ind] =\
                        results[frame_num][2]
                    evm.results['rayleigh_peak_offset'][ind] =\
                        results[frame_num][3]
                    evm.results['rayleigh_peak_snr'][ind] =\
                        results[frame_num][4]
                    evm.results['rayleigh_peak_nrmse'][ind] =\
                        results[frame_num][5]
                    evm.results[
                        'rayleigh_peak_center_uncertainty'][ind] =\
                        results[frame_num][6]

                if evm.evaluation_mode == 'sum':
                    combined = self.fit_spectrum_combined(
                        spectra, frequencies, region)
                    ind_c = (ind_x, ind_y, ind_z,
                             slice(None), region_key, 0)
                    evm.results[
                        'rayleigh_peak_position_f_combined'][ind_c] =\
                        combined[0]
                    evm.results[
                        'rayleigh_peak_fwhm_f_combined'][ind_c] =\
                        combined[1]
                    evm.results[
                        'rayleigh_peak_intensity_combined'][ind_c] =\
                        combined[2]
                    evm.results[
                        'rayleigh_peak_offset_combined'][ind_c] =\
                        combined[3]
                    evm.results[
                        'rayleigh_peak_snr_combined'][ind_c] =\
                        combined[4]
                    evm.results[
                        'rayleigh_peak_nrmse_combined'][ind_c] =\
                        combined[5]
                    evm.results[
                        'rayleigh_peak_center_uncertainty_combined'][
                        ind_c] = combined[6]

            # We can only do a multi-peak fit after the single-peak
            # Rayleigh fit is done, because we have to know the
            # Rayleigh peak positions in GHz in order to convert
            # the multi-peak fit bounds given in GHz into the position
            # in pixels.
            if nr_brillouin_peaks > 1:
                ind =\
                    (ind_x, ind_y, ind_z, slice(None), slice(None), 0)
                rayleigh_peaks = np.transpose(
                    evm.results['rayleigh_peak_position_f'][ind]
                )
                bounds_w0 = self.create_bounds(
                    brillouin_regions,
                    rayleigh_peaks
                )
                bounds_fwhm = self.create_bounds_fwhm(
                    brillouin_regions,
                    rayleigh_peaks
                )

                # 'sum'-mode bounds are built from the combined
                # (single-value, not per-frame) Rayleigh peak position -
                # already populated above, since the Rayleigh single-
                # peak fit block runs before this one. rayleigh_peaks_
                # combined keeps the same (nr_rayleigh_regions, nr
                # "times") shape create_bounds()/create_bounds_fwhm()
                # expect, just with nr "times" == 1 (the combined fit is
                # a single fit, not one per frame).
                bounds_w0_combined = bounds_fwhm_combined = None
                if evm.evaluation_mode == 'sum':
                    ind_rc = (ind_x, ind_y, ind_z,
                             slice(0, 1), slice(None), 0)
                    rayleigh_peaks_combined = np.transpose(
                        evm.results[
                            'rayleigh_peak_position_f_combined'][ind_rc]
                    )
                    bounds_w0_combined = self.create_bounds(
                        brillouin_regions,
                        rayleigh_peaks_combined
                    )
                    bounds_fwhm_combined = self.create_bounds_fwhm(
                        brillouin_regions,
                        rayleigh_peaks_combined
                    )

                for region_key, region in enumerate(
                        brillouin_regions):
                    results_multi_peak\
                        = self.fit_spectra(spectra,
                                           frequencies,
                                           region,
                                           nr_brillouin_peaks,
                                           bounds_w0[region_key],
                                           bounds_fwhm[region_key])
                    for frame_num, spectrum in enumerate(spectra):
                        ind = (ind_x, ind_y, ind_z,
                               frame_num, region_key,
                               slice(1, nr_brillouin_peaks+1))
                        evm.results[
                            'brillouin_peak_position_f'][ind] = \
                            results_multi_peak[frame_num][0]
                        evm.results[
                            'brillouin_peak_fwhm_f'][ind] = \
                            results_multi_peak[frame_num][1]
                        evm.results[
                            'brillouin_peak_intensity'][ind] = \
                            results_multi_peak[frame_num][2]
                        evm.results[
                            'brillouin_peak_offset'][ind] = \
                            results_multi_peak[frame_num][3]
                        evm.results[
                            'brillouin_peak_snr'][ind] = \
                            results_multi_peak[frame_num][4]
                        evm.results[
                            'brillouin_peak_nrmse'][ind] = \
                            results_multi_peak[frame_num][5]
                        evm.results[
                            'brillouin_peak_center_uncertainty'][ind] = \
                            results_multi_peak[frame_num][6]

                    if evm.evaluation_mode == 'sum':
                        bw0 = bounds_w0_combined[region_key][0] \
                            if bounds_w0_combined is not None else None
                        bfwhm = bounds_fwhm_combined[region_key][0] \
                            if bounds_fwhm_combined is not None else None
                        combined_multi_peak = self.fit_spectrum_combined(
                            spectra, frequencies, region,
                            nr_brillouin_peaks, bw0, bfwhm)
                        ind_c = (ind_x, ind_y, ind_z,
                                 slice(None), region_key,
                                 slice(1, nr_brillouin_peaks + 1))
                        evm.results[
                            'brillouin_peak_position_f_combined'][
                            ind_c] = combined_multi_peak[0]
                        evm.results[
                            'brillouin_peak_fwhm_f_combined'][
                            ind_c] = combined_multi_peak[1]
                        evm.results[
                            'brillouin_peak_intensity_combined'][
                            ind_c] = combined_multi_peak[2]
                        evm.results[
                            'brillouin_peak_offset_combined'][
                            ind_c] = combined_multi_peak[3]
                        evm.results[
                            'brillouin_peak_snr_combined'][
                            ind_c] = combined_multi_peak[4]
                        evm.results[
                            'brillouin_peak_nrmse_combined'][
                            ind_c] = combined_multi_peak[5]
                        evm.results[
                            'brillouin_peak_center_uncertainty_combined'][
                            ind_c] = combined_multi_peak[6]

            # Calculate the shift of the Rayleigh peaks,
            # in order to follow the peaks in case of a drift
            rayleigh_peak_current = evm.results['rayleigh_peak_position_f'][
                    ind_x, ind_y, ind_z, :, :, :]
            # If we haven't found a valid Rayleigh peak position,
            # but the current one is valid, use it
            if not np.isnan(rayleigh_peak_current).all()\
                    and np.isnan(rayleigh_peak_initial).all():
                rayleigh_peak_initial = rayleigh_peak_current
            shift = rayleigh_peak_current - rayleigh_peak_initial
            evm.results['rayleigh_shift'][ind_x, ind_y, ind_z, :, :, :] = shift

            # Calculate the derived values every ten steps
            if not (idx % 10):
                calculate_derived_values()

        calculate_derived_values()

        return

    @staticmethod
    def fit_spectra(spectra, frequencies, region, nr_peaks=1,
                    bounds_w0=None, bounds_fwhm=None):
        fits = []
        for frame_num, spectrum in enumerate(spectra):
            if bounds_w0 is None and bounds_fwhm is None:
                fit = fit_lorentz_region(
                    region,
                    frequencies[frame_num],
                    spectrum,
                    nr_peaks
                )
            elif bounds_fwhm is None:
                fit = fit_lorentz_region(
                    region,
                    frequencies[frame_num],
                    spectrum,
                    nr_peaks,
                    bounds_w0[frame_num]
                )
            else:
                fit = fit_lorentz_region(
                    region,
                    frequencies[frame_num],
                    spectrum,
                    nr_peaks,
                    bounds_w0[frame_num],
                    bounds_fwhm[frame_num]
                )
            fits.append(fit)
        return fits

    @staticmethod
    def fit_spectrum_combined(spectra, frequencies, region, nr_peaks=1,
                              bounds_w0=None, bounds_fwhm=None):
        """
        The 'sum'-mode counterpart to fit_spectra() (see
        EvaluationModel.evaluation_mode): combines every frame's
        spectrum via _combine_spectra() and fits the resulting single
        summed spectrum once with fit_lorentz_region(), instead of
        fitting each frame separately and averaging the results.

        Unlike fit_spectra(), `bounds_w0`/`bounds_fwhm` here are the
        bounds for this one combined "frame" directly - not a
        per-frame list to index by frame_num. The caller
        (EvaluationController.evaluate()) builds them from the
        combined Rayleigh peak position (a single value, not one per
        frame) via create_bounds()/create_bounds_fwhm().

        Returns
        -------
        tuple
            The 7-tuple fit_lorentz_region() returns: w0, fwhm,
            intensity, offset, snr, nrmse, center_uncertainty.
        """
        summed_spectrum, reference_frequencies = _combine_spectra(
            spectra, frequencies)
        return fit_lorentz_region(
            region, reference_frequencies, summed_spectrum, nr_peaks,
            bounds_w0, bounds_fwhm)

    def create_bounds(self, brillouin_regions, rayleigh_peaks):
        """
        This function converts the bounds settings into
        a bounds object for the fitting function
        Allowed parameters for the bounds settings are
        - 'min'/'max'   -> Will be converted to the respective
            lower or upper limit of the given region
        - '-Inf', 'Inf' -> Will be converted to -np.inf or np.inf
        - number [GHz]  -> Will be converted into the pixel
            position of the given frequency
        Parameters
        ----------
        brillouin_regions
        rayleigh_peaks

        Returns
        -------

        """
        evm = self.session.evaluation_model()
        bounds = evm.bounds_w0
        if bounds is None:
            return None

        # We need two Rayleigh peak positions to determine the FSR
        # and decide whether a peak is Stokes or Anti-Stokes
        if rayleigh_peaks.shape[0] != 2:
            return None

        w0_bounds = []
        # We have to create a separate bound for every region
        for region_idx, region in enumerate(brillouin_regions):
            local_time = []
            for rayleigh_idx in range(rayleigh_peaks.shape[1]):
                anti_stokes_limit = np.nanmean(rayleigh_peaks[:, rayleigh_idx])
                # We need to figure out, whether a peak is a
                # Stokes or Anti-Stokes peak in order to decide
                # to which Rayleigh peak we need to relate to.
                tmp = region >= anti_stokes_limit
                is_pure_region = (tmp == tmp[0]).all()

                is_anti_stokes_region = np.mean(region) >\
                    anti_stokes_limit

                local_bound = []
                for bound in bounds:
                    parsed_bound = []
                    for limit in bound:
                        try:
                            parsed_bound.append(float(limit))
                        except ValueError:
                            parsed_bound.append(np.nan)
                    # We don't treat Inf as a value
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            action='ignore',
                            message='Mean of empty slice'
                        )
                        is_anti_stokes_peak =\
                            np.nanmean(
                                np.array(parsed_bound)[
                                    np.isfinite(parsed_bound)]
                            ) < 0

                    is_anti_stokes = is_anti_stokes_region if\
                        is_pure_region else is_anti_stokes_peak

                    local_limit = []
                    for limit in bound:
                        if limit.lower() == 'min':
                            val = region[int(is_anti_stokes)]
                            # print(val)
                        elif limit.lower() == 'max':
                            val = region[int(not is_anti_stokes)]
                        elif limit.lower() == '-inf':
                            val = -((-1) ** is_anti_stokes)\
                                          * np.inf
                        elif limit.lower() == 'inf':
                            val = ((-1) ** is_anti_stokes)\
                                          * np.inf
                        else:
                            # Try to convert the value in GHz into
                            # a value in pixel depending on the time
                            try:
                                val = ((-1) ** is_anti_stokes)\
                                          * 1e9 * abs(float(limit))\
                                          + rayleigh_peaks[
                                          int(is_anti_stokes), rayleigh_idx]
                            except BaseException:
                                val = np.inf
                        local_limit.append(val)
                    # Check that the bounds are sorted ascendingly
                    # (for anti-stokes, they might not).
                    local_limit.sort()
                    local_bound.append(local_limit)
                local_time.append(local_bound)
            w0_bounds.append(local_time)

        return w0_bounds

    def create_bounds_fwhm(self, brillouin_regions, rayleigh_peaks):
        """
        This function converts the fwhm bounds settings into
        a fwhm bounds object for the fitting function
        Allowed parameters for the bounds settings are
        - 'min'/'max'   -> Will be converted to 0/np.inf
        - '-Inf', 'Inf' -> Will be converted to 0 or np.inf
        - number [GHz]  -> Will be converted into the value in Hz
        Parameters
        ----------
        brillouin_regions
        rayleigh_peaks

        Returns
        -------

        """
        evm = self.session.evaluation_model()
        bounds_fwhm = evm.bounds_fwhm
        if bounds_fwhm is None:
            return None

        fwhm_bounds = []
        # We have to create a separate bound for every region
        for region_idx, _ in enumerate(brillouin_regions):
            region_bound = []
            for rayleigh_idx in range(rayleigh_peaks.shape[1]):
                peak_bound = []
                for bound in bounds_fwhm:
                    local_bound = []
                    for limit in bound:
                        if limit.lower() == 'min':
                            val = 0
                        elif limit.lower() == '-inf':
                            val = 0
                        elif limit.lower() == 'max':
                            val = np.inf
                        elif limit.lower() == 'inf':
                            val = np.inf
                        else:
                            try:
                                val = 1e9*abs(float(limit))
                            except ValueError:
                                val = np.inf
                        local_bound.append(val)
                    peak_bound.append(local_bound)
                region_bound.append(peak_bound)
            fwhm_bounds.append(region_bound)
        return fwhm_bounds

    def get_data(self, parameter_key, brillouin_peak_index=0):
        """
        This function returns the evaluated data,
        its positions, dimensionality and labels
        Parameters
        ----------
        parameter_key: str
            The key of the parameter requested.
            See bmlab.model.evaluation_model.get_parameter_keys()
        brillouin_peak_index: int
            The index of the Brillouin peak to show in case
            we did a multi-peak fit
            0: the single-peak fit
            1:nr_brillouin_peaks: the multi-peak fits
            nr_brillouin_peaks+1: all multi-peak fits average
            nr_brillouin_peaks+2: all multi-peak fits weighted average

        Returns
        -------
        data: np.ndarray or None
            The data to show. This is always a 3-dimensional array,
            or None if the current repetition has no valid
            measurement grid (e.g. an aborted/restarted acquisition
            that never wrote any positions).
        positions: list or None
            This is a list of length 3 containing ndarrays with
            the spatial positions of the data points, or None.
        dimensionality: int or None
            Whether it's a 0, 1, 2, or 3D measurement, or None.
        labels: list or None
            The labels of the positions, or None.
        """
        resolution = self.session.get_payload_resolution()
        if resolution is None:
            return None, None, None, None

        dimensionality = sum(np.array(resolution) > 1)

        # Get the positions and squeeze them
        pos = self.session.get_payload_positions()

        positions = list(pos.values())
        labels = list(map(lambda axis_label:
                          r'$' + axis_label + '$ [$\\mu$m]', ['x', 'y', 'z']))

        evm = self.session.evaluation_model()
        # 'sum' mode (see EvaluationModel.evaluation_mode): transparently
        # redirect fit-derived keys to their single-sum-fit '_combined'
        # backing array instead of the per-frame one - this one redirect
        # is what makes QUALITY_METRIC_KEYS thresholds, the GUI plot and
        # CSV export all pick up sum-mode values with no changes needed
        # anywhere else, since they all go through get_data(). The
        # original parameter_key is kept for everything else below (e.g.
        # evm.parameters[parameter_key]['scaling']) - '_combined' keys
        # are not registered in evm.parameters.
        results_key = parameter_key
        if evm.evaluation_mode == 'sum' \
                and (parameter_key + '_combined') in evm.results:
            results_key = parameter_key + '_combined'
        data = evm.results[results_key]

        # Ensure that we always get the expected shape
        # (even if the array was not initialized yet)
        if data.size == 0:
            data = np.empty(resolution)
            data[:] = np.nan

        # Slice the appropriate Brillouin peak if necessary and possible
        if data.ndim >= 6:
            nr_peaks_stored = data.shape[5]
            if nr_peaks_stored > 1\
                    and brillouin_peak_index < nr_peaks_stored + 2:
                if brillouin_peak_index < nr_peaks_stored:
                    sliced = data[:, :, :, :, :, brillouin_peak_index]
                # Average all multi-peak fits
                if brillouin_peak_index == nr_peaks_stored:
                    sliced = data[:, :, :, :, :, 1:]
                # Weighted average of all multi-peak fits
                if brillouin_peak_index == nr_peaks_stored + 1:
                    intensity_key = 'brillouin_peak_intensity'
                    fwhm_key = 'brillouin_peak_fwhm_f'
                    if evm.evaluation_mode == 'sum':
                        intensity_key += '_combined'
                        fwhm_key += '_combined'
                    weight =\
                        evm.results[
                            intensity_key][:, :, :, :, :, 1:]\
                        * evm.results[fwhm_key][
                          :, :, :, :, :, 1:]
                    # Nansum returns 0 if all entries are NaN, dividing by this
                    # hence gives an invalid value error
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            action='ignore',
                            message='invalid value encountered in divide'
                        )
                        sliced = \
                            np.nansum(data[:, :, :, :, :, 1:] *
                                      weight, axis=5)\
                            / np.nansum(weight, axis=5)
            else:
                sliced = data[:, :, :, :, :, 0]
        else:
            sliced = data

        # Average all non-spatial dimensions.
        # Do not show warning which occurs when a slice contains only NaNs.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore',
                message='Mean of empty slice'
            )
            data = np.nanmean(
                sliced,
                axis=tuple(range(3, sliced.ndim))
            )

        # Scale the date in case of GHz
        data = evm.parameters[parameter_key]['scaling'] * data

        return data, positions, dimensionality, labels

    # Parameter keys meaningful as a quality filter - see
    # EvaluationModel.get_default_parameters() for what each measures.
    # Every key here except 'brillouin_peak_fwhm_f' also has a default
    # threshold in EvaluationModel.get_default_quality_thresholds() -
    # kept in sync by tests/test_evaluation_controller.py::
    # test_quality_metric_keys_matches_default_thresholds(), since the
    # two lists are otherwise independently maintained.
    QUALITY_METRIC_KEYS = [
        'brillouin_peak_snr',
        'brillouin_peak_center_uncertainty',
        'brillouin_shift_frame_spread',
        'rayleigh_peak_snr',
        'rayleigh_peak_center_uncertainty',
        'rayleigh_shift_frame_spread',
        'brillouin_peak_nrmse',
        'rayleigh_peak_nrmse',
        'brillouin_peak_fwhm_f',
    ]

    # Sentinel default for set_quality_threshold()'s keyword arguments,
    # distinct from None: None is a legitimate value there (clear that
    # bound entirely), so "not passed - leave this field as it was"
    # needs its own marker instead of overloading None for both.
    _UNSET = object()

    def set_quality_threshold(self, metric_key, enabled=_UNSET,
                              min_value=_UNSET, max_value=_UNSET):
        """
        Updates the stored threshold for one quality metric (a key in
        QUALITY_METRIC_KEYS), creating it if not already set. Pass
        only the fields that changed - an omitted field keeps its
        previous value (defaults for a new entry: enabled=True,
        min_value=None i.e. no lower bound, max_value=None i.e. no
        upper bound); an explicit min_value=None/max_value=None clears
        that bound. Values are in the same display units get_data()
        returns (e.g. GHz), matching what a threshold slider would
        show.
        """
        evm = self.session.evaluation_model()
        if not evm:
            return
        current = evm.quality_thresholds.get(
            metric_key, {'enabled': True, 'min': None, 'max': None})
        if enabled is not self._UNSET:
            current['enabled'] = enabled
        if min_value is not self._UNSET:
            current['min'] = min_value
        if max_value is not self._UNSET:
            current['max'] = max_value
        evm.quality_thresholds[metric_key] = current

    def _default_quality_peak_index(self, evm):
        """
        The Brillouin peak index (see get_data()'s own docstring for
        what each index means) quality checks fall back to when the
        caller doesn't pick one explicitly: index 0 - the always-
        computed single-peak fit - is only actually the fit result a
        user cares about when nr_brillouin_peaks == 1. Once multi-peak
        fitting is on, index 0 is just an internal reference fit, so
        default instead to the weighted average of the real multi-peak
        fits (nr_peaks_stored + 1, i.e. nr_brillouin_peaks + 2) - the
        same combined estimate BrillouinExport's '_peak-average-
        weighted' column reports.
        """
        nr_brillouin_peaks = getattr(evm, 'nr_brillouin_peaks', 1)
        if nr_brillouin_peaks <= 1:
            return 0
        return nr_brillouin_peaks + 2

    def compute_quality_mask(self, brillouin_peak_index=None):
        """
        Evaluates every enabled quality threshold (see
        set_quality_threshold()) against the current evaluation
        results and combines them with AND.

        Parameters
        ----------
        brillouin_peak_index: int or None
            Which Brillouin peak fit's diagnostics to check (see
            get_data()'s own docstring for index semantics). None (the
            default) resolves via _default_quality_peak_index(): the
            single-peak fit if nr_brillouin_peaks == 1, otherwise the
            weighted average across the real multi-peak fits - never
            the always-computed single-peak reference fit, which a
            multi-peak measurement isn't actually using.

        Returns
        -------
        mask: np.ndarray of bool, shape (dim_x, dim_y, dim_z), or None
            True where the point passes every enabled threshold. None
            if there's no valid measurement grid.
        measured: np.ndarray of bool, same shape, or None
            True where the point was actually measured (has a 'time'
            value) - an unmeasured point is never counted as passing
            or failing a quality check, since there was nothing to
            check.
        """
        evm = self.session.evaluation_model()
        if not evm:
            return None, None

        if brillouin_peak_index is None:
            brillouin_peak_index = self._default_quality_peak_index(evm)

        time_data, _, _, _ = self.get_data('time', brillouin_peak_index)
        if time_data is None:
            return None, None
        measured = ~np.isnan(time_data)

        mask = measured.copy()
        for metric_key, threshold in evm.quality_thresholds.items():
            if not threshold.get('enabled', True):
                continue
            if threshold.get('min') is None \
                    and threshold.get('max') is None:
                continue
            data, _, _, _ = self.get_data(metric_key, brillouin_peak_index)
            metric_mask = np.ones(mask.shape, dtype=bool)
            if threshold.get('min') is not None:
                metric_mask &= (data >= threshold['min'])
            if threshold.get('max') is not None:
                metric_mask &= (data <= threshold['max'])
            mask &= metric_mask

        return mask, measured

    def apply_quality_thresholds(self, brillouin_peak_index=None):
        """
        Computes compute_quality_mask() and stores it as
        evm.results['quality_pass'] (1.0 where a measured point passes
        every enabled threshold, 0.0 where it fails one, NaN where the
        point was never measured) - so it flows through get_data(),
        the standard plots, and BrillouinExport's combined CSV exactly
        like any other parameter, without touching any other result.
        `brillouin_peak_index` is passed straight through to
        compute_quality_mask() - see its docstring for the default.
        """
        evm = self.session.evaluation_model()
        if not evm:
            return
        mask, measured = self.compute_quality_mask(brillouin_peak_index)
        if mask is None:
            return
        quality_pass = np.full(mask.shape, np.nan)
        quality_pass[measured] = mask[measured].astype(float)
        # quality_pass has no frame/region/peak axes of its own (it's
        # already a per-point pass/fail) - broadcast to the standard
        # 6D results shape so it flows through get_data() like every
        # other parameter. evm.results['time'] normally already has
        # that shape, but a repetition that was never evaluate()'d
        # keeps its EvaluationModel.__init__ placeholder (an empty
        # (0,) array) instead - in that case fall back to a minimal
        # shape built from mask itself (get_data()'s own resolution
        # fallback), rather than broadcasting into a shape that
        # doesn't even agree with mask on its first three dimensions.
        time_shape = evm.results['time'].shape
        if len(time_shape) >= 3 and time_shape[:3] == mask.shape:
            shape = time_shape
        else:
            shape = mask.shape + (1, 1, 1)
        evm.results['quality_pass'] = np.broadcast_to(
            quality_pass[:, :, :, np.newaxis, np.newaxis, np.newaxis],
            shape).copy()

    def quality_metrics_need_recompute(self):
        """
        True if this repetition has measured points but the fit-
        residual-based quality metrics (brillouin_peak_snr and its
        siblings - only evaluate()'s fitting loop populates those,
        unlike e.g. brillouin_shift_frame_spread, which
        calculate_derived_values() can backfill from already-fitted
        peak positions alone) were never computed for them - i.e. this
        session's saved fit predates those metrics. Used by BMicro's
        Quality tab to auto-trigger a fresh evaluate() when opened
        against such a session, rather than showing an empty view.
        """
        evm = self.session.evaluation_model()
        if not evm:
            return False
        time_data, _, _, _ = self.get_data('time', 0)
        if time_data is None:
            return False
        measured = ~np.isnan(time_data)
        if not measured.any():
            return False
        snr_data, _, _, _ = self.get_data('brillouin_peak_snr', 0)
        if snr_data is None:
            return False
        return not np.isfinite(snr_data[measured]).any()

    def get_fits(self, image_key):
        resolution = self.session.get_payload_resolution()
        indices = self.get_indices_from_key(resolution, image_key)

        evm = self.session.evaluation_model()
        if not evm:
            return
        return evm.get_fits(*indices)

    def get_fits_combined(self, image_key):
        resolution = self.session.get_payload_resolution()
        indices = self.get_indices_from_key(resolution, image_key)

        evm = self.session.evaluation_model()
        if not evm:
            return
        return evm.get_fits_combined(*indices)

    def get_combined_spectrum(self, image_key):
        """
        Live preview of the 'sum'-mode combined spectrum (see
        _combine_spectra()) for a single already-evaluated point,
        regardless of the model's *current* evaluation_mode - lets a
        viewer (e.g. BMicro's per-pixel spectrum dialog) show what the
        combined spectrum trace would look like without having to
        re-run evaluate() in 'sum' mode first. Uses the same cached
        per-frame spectra and per-frame timestamps evaluate() itself
        wrote (evm.get_spectra()/evm.results['time']), so the
        combination is consistent with what a 'sum'-mode evaluation
        would actually fit.

        Returns
        -------
        (summed_spectrum, reference_frequencies) or (None, None) if
        this point has no cached spectra/frequencies yet (e.g. never
        evaluated).
        """
        evm = self.session.evaluation_model()
        cm = self.session.calibration_model()
        if not evm or not cm:
            return None, None
        spectra = evm.get_spectra(image_key)
        if not spectra:
            return None, None
        resolution = self.session.get_payload_resolution()
        ind_x, ind_y, ind_z = self.get_indices_from_key(
            resolution, image_key)
        times = evm.results['time'][ind_x, ind_y, ind_z, :, 0, 0]
        frequencies = cm.get_frequencies_by_time(times)
        if frequencies is None:
            return None, None
        return _combine_spectra(spectra, list(frequencies))

    @staticmethod
    def get_key_from_indices(resolution, ind_x, ind_y, ind_z):
        if len(resolution) != 3:
            raise ValueError('resolution has wrong dimension')
        if ind_x >= resolution[0]:
            raise IndexError('x index out of range')
        if ind_y >= resolution[1]:
            raise IndexError('y index out of range')
        if ind_z >= resolution[2]:
            raise IndexError('z index out of range')
        return str(int(ind_z * (resolution[0] * resolution[1])
                   + ind_y * resolution[0] + ind_x))

    @staticmethod
    def get_indices_from_key(resolution, key):
        key = int(key)
        ind_z = floor(key / (resolution[0] * resolution[1]))
        ind_y = floor(
            (key - ind_z * (resolution[0] * resolution[1])) / resolution[0])
        ind_x = (key % (resolution[0] * resolution[1])) % resolution[0]
        # ind_y = (key - ind_x) % resolution[0]
        if ind_x >= resolution[0]\
                or ind_y >= resolution[1]\
                or ind_z >= resolution[2]:
            raise ValueError('Invalid key')
        return ind_x, ind_y, ind_z


def _get_fsr(cm):
    """
    A robust single estimate (Hz) of the VIPA etalon's free spectral
    range - a fixed physical property of the etalon (thickness,
    refractive index, angle - see bmlab.models.setup.Setup), not
    expected to drift between measurement frames the way a per-frame
    Rayleigh peak fit can. Computed as the median of the 4th parameter
    fitted by fits.fit_vipa() (`fsr` in its `error()` closure) across
    every calibration frame this repetition has - i.e. from the clean,
    high-SNR calibration sample fits, not from the (often much weaker)
    Rayleigh peaks of the actual measurement.

    Parameters
    ----------
    cm: CalibrationModel

    Returns
    -------
    float or None
        The estimated FSR in Hz, or None if no calibration has been
        fitted yet.
    """
    fsrs = [abs(params[3])
            for frame_params in cm.vipa_params.values()
            for params in frame_params]
    if not fsrs:
        return None
    return np.median(fsrs)


def calculate_derived_values():
    """
    We calculate the Brillouin shift in GHz here
    """
    session = Session.get_instance()
    evm = session.evaluation_model()
    if not evm:
        return

    if evm.results['brillouin_peak_position_f'].size == 0:
        return

    if evm.results['rayleigh_peak_position_f'].size == 0:
        return

    shape_brillouin = evm.results['brillouin_peak_position_f'].shape
    shape_rayleigh = evm.results['rayleigh_peak_position_f'].shape

    # We calculate every possible combination of
    # Brillouin peak and Rayleigh peak position difference
    # and then use the smallest absolute value.
    # That saves us from sorting Rayleigh peaks to Brillouin peaks,
    # because a Brillouin peak always belongs to the Rayleigh peak nearest.
    brillouin_shift_f = np.nan * np.ones((*shape_brillouin, shape_rayleigh[4]))
    for idx in range(shape_rayleigh[4]):
        brillouin_shift_f[:, :, :, :, :, :, idx] = abs(
            evm.results['brillouin_peak_position_f'] -
            np.tile(
                evm.results['rayleigh_peak_position_f'][:, :, :, :, [idx], :],
                (1, 1, 1, 1, 1, shape_brillouin[5])
            )
        )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            action='ignore',
            message='All-NaN slice encountered'
        )
        evm.results['brillouin_shift_f'] = np.nanmin(brillouin_shift_f, 6)

    # Stokes/Anti-Stokes-distance shift - only defined when exactly 2
    # Brillouin regions are selected (the Stokes/Anti-Stokes layout this
    # method assumes). Only the primary (peak-index 0) fit is used;
    # multi-component sub-peak slots (index >= 1) are left NaN.
    #
    # A VIPA spectrum is periodic with period FSR, so within one order
    # (Rayleigh, Stokes, Anti-Stokes, Rayleigh) the Anti-Stokes peak sits
    # near the *next* Rayleigh order, not mirrored directly around the
    # same Rayleigh peak as Stokes: distance(Anti-Stokes, Stokes) =
    # FSR - 2 * shift, NOT 2 * shift. FSR comes from the calibration fit
    # (_get_fsr()), not from this frame's own (often much weaker,
    # biological-sample) Rayleigh peaks - that's what keeps this method
    # useful even when the Rayleigh fit itself is unreliable.
    evm.results['brillouin_shift_f_stokes_anti_stokes'][:] = np.nan
    cm = session.calibration_model()
    fsr = _get_fsr(cm) if cm else None
    if shape_brillouin[4] == 2 and fsr is not None:
        positions = evm.results['brillouin_peak_position_f'][..., 0]
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=RuntimeWarning)
            distance = (np.nanmax(positions, axis=4)
                        - np.nanmin(positions, axis=4))
        shift_sa = (fsr - distance) / 2
        evm.results['brillouin_shift_f_stokes_anti_stokes'][..., 0] = \
            np.repeat(shift_sa[..., np.newaxis], 2, axis=4)

    # @since 0.14.0 - 'sum'-mode combined-fit derived values (see
    # EvaluationModel.evaluation_mode). Same two computations as above,
    # sourced from the single sum-fit's peak positions
    # (*_position_f_combined) instead of the per-frame ones. This runs
    # unconditionally, regardless of evaluation_mode - a harmless no-op
    # (all-NaN result) in 'single' mode, since the _combined position
    # arrays stay NaN then.
    if 'brillouin_peak_position_f_combined' in evm.results \
            and 'rayleigh_peak_position_f_combined' in evm.results:
        brillouin_shift_f_combined = np.nan * np.ones(
            (*shape_brillouin, shape_rayleigh[4]))
        for idx in range(shape_rayleigh[4]):
            brillouin_shift_f_combined[:, :, :, :, :, :, idx] = abs(
                evm.results['brillouin_peak_position_f_combined'] -
                np.tile(
                    evm.results['rayleigh_peak_position_f_combined'][
                        :, :, :, :, [idx], :],
                    (1, 1, 1, 1, 1, shape_brillouin[5])
                )
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore', message='All-NaN slice encountered')
            evm.results['brillouin_shift_f_combined'] = np.nanmin(
                brillouin_shift_f_combined, 6)

        evm.results['brillouin_shift_f_stokes_anti_stokes_combined'][:] = \
            np.nan
        if shape_brillouin[4] == 2 and fsr is not None:
            positions_combined = evm.results[
                'brillouin_peak_position_f_combined'][..., 0]
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action='ignore', category=RuntimeWarning)
                distance_combined = (
                    np.nanmax(positions_combined, axis=4)
                    - np.nanmin(positions_combined, axis=4))
            shift_sa_combined = (fsr - distance_combined) / 2
            evm.results[
                'brillouin_shift_f_stokes_anti_stokes_combined'][..., 0] = \
                np.repeat(shift_sa_combined[..., np.newaxis], 2, axis=4)

    # Quality metrics (see EvaluationModel.get_default_parameters() for
    # what each one means) - all NaN-safe, since a point can easily
    # have some frames/regions missing. Silence numpy's "empty slice"
    # warnings for the (normal, e.g. unmeasured or single-frame) cases
    # where a whole axis being reduced over is all-NaN.
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', category=RuntimeWarning)
        frame_spread = (
            np.nanmax(evm.results['brillouin_shift_f'], axis=3) -
            np.nanmin(evm.results['brillouin_shift_f'], axis=3))
        evm.results['brillouin_shift_frame_spread'] = np.repeat(
            frame_spread[:, :, :, np.newaxis, :, :],
            shape_brillouin[3], axis=3)

        rayleigh_spread = (
            np.nanmax(evm.results['rayleigh_peak_position_f'], axis=3) -
            np.nanmin(evm.results['rayleigh_peak_position_f'], axis=3))
        evm.results['rayleigh_shift_frame_spread'] = np.repeat(
            rayleigh_spread[:, :, :, np.newaxis, :, :],
            shape_rayleigh[3], axis=3)


class BackgroundController(ImageController):
    """
    Fits a repetition's background reference points (see
    bmlab.file.Background) exactly like EvaluationController fits the
    main payload grid - same per-repetition calibration
    (cm.get_frequencies_by_time()), same extraction arc
    (em.get_arc_by_time()) and the same Rayleigh/Brillouin regions
    (peak_selection_model()) - the only structural difference is that
    background points are a flat, ROI-drawn list (Brillouin::
    backgroundGridPoints()), not a rectangular (x, y, z) grid, so
    results are indexed by a flat point axis instead of by grid
    indices, and only a single-peak fit is done (see BackgroundModel).
    """

    def __init__(self, *args, **kwargs):
        session = Session.get_instance()
        super(BackgroundController, self).__init__(
            model=session.background_model,
            get_image=session.get_background_image,
            get_time=session.get_background_time,
            get_exposure=session.get_background_exposure
        )
        return

    def evaluate(self, abort=None, count=None, max_count=None):
        em = self.session.extraction_model()
        if not em:
            if max_count is not None:
                max_count.value = -1
            return

        cm = self.session.calibration_model()
        if not cm:
            if max_count is not None:
                max_count.value = -1
            return

        pm = self.session.peak_selection_model()
        if not pm:
            if max_count is not None:
                max_count.value = -1
            return

        bgm = self.session.background_model()
        if not bgm:
            if max_count is not None:
                max_count.value = -1
            return

        point_keys = self.session.get_background_keys(sort_by_time=True)

        if max_count is not None:
            max_count.value = len(point_keys)

        if not point_keys:
            if max_count is not None:
                max_count.value = -1
            return

        brillouin_regions = pm.get_brillouin_regions()
        rayleigh_regions = pm.get_rayleigh_regions()

        # Get first spectrum to find number of images
        spectra, _, _ = self.extract_spectra(point_keys[0])

        if not spectra:
            if max_count is not None:
                max_count.value = -1
            return

        bgm.point_keys = point_keys
        bgm.initialize_results_arrays({
            'nr_points': len(point_keys),
            'nr_images': len(spectra),
            'nr_brillouin_regions': len(brillouin_regions),
            'nr_rayleigh_regions': len(rayleigh_regions),
        })

        for idx, point_key in enumerate(point_keys):

            if count is not None:
                count.value += 1

            if (abort is not None) and abort.value:
                self.calculate_derived_values()
                if max_count is not None:
                    max_count.value = -1
                return

            position = self.session.get_background_position(point_key)
            if position is not None:
                for axis in ('x', 'y', 'z'):
                    bgm.positions[axis][idx] = position[axis]
            stage_position = \
                self.session.get_background_stage_position(point_key)
            if stage_position is not None:
                for axis in ('x', 'y', 'z'):
                    bgm.stage_positions[axis][idx] = stage_position[axis]

            spectra, times, intensities = self.extract_spectra(point_key)
            if spectra is None:
                continue
            bgm.results['time'][idx, :] = times
            bgm.results['intensity'][idx, :] = intensities

            frequencies = cm.get_frequencies_by_time(times)
            # If we don't have frequency axis, we cannot evaluate on it
            if frequencies is None:
                continue
            frequencies = list(frequencies)
            bgm.set_frequencies(point_key, frequencies)

            for region_key, region in enumerate(brillouin_regions):
                results = EvaluationController.fit_spectra(
                    spectra, frequencies, region)
                for frame_num, _ in enumerate(spectra):
                    ind = (idx, frame_num, region_key)
                    bgm.results['brillouin_peak_position_f'][ind] = \
                        results[frame_num][0]
                    bgm.results['brillouin_peak_fwhm_f'][ind] = \
                        results[frame_num][1]
                    bgm.results['brillouin_peak_intensity'][ind] = \
                        results[frame_num][2]
                    bgm.results['brillouin_peak_offset'][ind] = \
                        results[frame_num][3]
                    bgm.results['brillouin_peak_snr'][ind] = \
                        results[frame_num][4]
                    bgm.results['brillouin_peak_nrmse'][ind] = \
                        results[frame_num][5]
                    bgm.results[
                        'brillouin_peak_center_uncertainty'][ind] = \
                        results[frame_num][6]

            for region_key, region in enumerate(rayleigh_regions):
                results = EvaluationController.fit_spectra(
                    spectra, frequencies, region)
                for frame_num, _ in enumerate(spectra):
                    ind = (idx, frame_num, region_key)
                    bgm.results['rayleigh_peak_position_f'][ind] = \
                        results[frame_num][0]
                    bgm.results['rayleigh_peak_fwhm_f'][ind] = \
                        results[frame_num][1]
                    bgm.results['rayleigh_peak_intensity'][ind] = \
                        results[frame_num][2]
                    bgm.results['rayleigh_peak_offset'][ind] = \
                        results[frame_num][3]
                    bgm.results['rayleigh_peak_snr'][ind] = \
                        results[frame_num][4]
                    bgm.results['rayleigh_peak_nrmse'][ind] = \
                        results[frame_num][5]
                    bgm.results[
                        'rayleigh_peak_center_uncertainty'][ind] = \
                        results[frame_num][6]

            if not (idx % 10):
                self.calculate_derived_values()

        self.calculate_derived_values()

        return

    @staticmethod
    def calculate_derived_values():
        """
        Computes brillouin_shift_f the same way controllers.
        calculate_derived_values() does for the main payload grid: for
        every Brillouin/Rayleigh region pair, the absolute difference
        between their fitted positions, then the smallest of those per
        Brillouin region - so a Brillouin peak is always matched to
        its nearest Rayleigh peak without having to sort regions to
        peaks by hand.
        """
        session = Session.get_instance()
        bgm = session.background_model()
        if not bgm:
            return

        if bgm.results['brillouin_peak_position_f'].size == 0:
            return

        if bgm.results['rayleigh_peak_position_f'].size == 0:
            return

        shape_brillouin = bgm.results['brillouin_peak_position_f'].shape
        shape_rayleigh = bgm.results['rayleigh_peak_position_f'].shape

        brillouin_shift_f = np.nan * np.ones(
            (*shape_brillouin, shape_rayleigh[2]))
        for idx in range(shape_rayleigh[2]):
            brillouin_shift_f[:, :, :, idx] = abs(
                bgm.results['brillouin_peak_position_f'] -
                np.tile(
                    bgm.results['rayleigh_peak_position_f'][:, :, [idx]],
                    (1, 1, shape_brillouin[2])
                )
            )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore',
                message='All-NaN slice encountered'
            )
            bgm.results['brillouin_shift_f'] = np.nanmin(
                brillouin_shift_f, 3)

        # Stokes/Anti-Stokes-distance shift - see controllers.
        # calculate_derived_values() for the main payload grid (including
        # why the FSR correction below is needed); same formula, only
        # defined for exactly 2 Brillouin regions, applied here to the
        # flat background-point shape (no peak-index axis).
        bgm.results['brillouin_shift_f_stokes_anti_stokes'][:] = np.nan
        cm = session.calibration_model()
        fsr = _get_fsr(cm) if cm else None
        if shape_brillouin[2] == 2 and fsr is not None:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    action='ignore', category=RuntimeWarning)
                distance = (
                    np.nanmax(
                        bgm.results['brillouin_peak_position_f'], axis=2)
                    - np.nanmin(
                        bgm.results['brillouin_peak_position_f'], axis=2)
                )
            shift_sa = (fsr - distance) / 2
            bgm.results['brillouin_shift_f_stokes_anti_stokes'] = np.repeat(
                shift_sa[:, :, np.newaxis], 2, axis=2)

    def get_data(self, parameter_key):
        """
        Returns the evaluated data for every background point,
        averaged over frames and regions, together with each point's
        own (x, y, z) target position - the background-point
        equivalent of EvaluationController.get_data(), but for a flat
        point list rather than a spatial grid (there is no resolution
        to lay the points out on).

        Parameters
        ----------
        parameter_key: str
            The key of the parameter requested.
            See bmlab.models.background_model.BackgroundModel.
            get_parameter_keys()

        Returns
        -------
        data: np.ndarray or None
            1D array, one value per background point (in point_keys
            order), or None if no evaluation has been run yet.
        positions: dict or None
            {'x': ndarray, 'y': ndarray, 'z': ndarray}, the target
            position of each point, in the same order as `data`.
        point_keys: list or None
            The background image key each entry in `data` belongs to.
        """
        bgm = self.session.background_model()
        if not bgm:
            return None, None, None

        data = bgm.results.get(parameter_key)
        if data is None or data.size == 0:
            return None, None, None

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore',
                message='Mean of empty slice'
            )
            data = np.nanmean(data, axis=tuple(range(1, data.ndim)))

        data = bgm.parameters[parameter_key]['scaling'] * data

        return data, bgm.positions, bgm.point_keys


class Controller(object):

    def __init__(self):
        self.session = Session.get_instance()
        return

    def evaluate(self, filepath, setup, orientation,
                 brillouin_regions, rayleigh_regions,
                 repetitions=None, nr_brillouin_peaks=1,
                 multi_peak_bounds=None,
                 multi_peak_bounds_fwhm=None):
        # Load data file
        self.session.set_file(filepath)

        # Evaluate all repetitions if not requested differently
        if repetitions is None:
            repetitions = self.session.file.repetition_keys()

        for repetition in repetitions:
            # Select repetition
            self.session.set_current_repetition(repetition)
            self.session.set_setup(setup)

            # Set orientation
            self.session.orientation = orientation

            ec = ExtractionController()
            cc = CalibrationController()
            psc = PeakSelectionController()
            evc = EvaluationController()

            # First add all extraction points because this
            # can influence the extraction for other calibrations
            ec.find_points_all()

            # Then do the calibration
            for calib_key in self.session.get_calib_keys():
                cc.find_peaks(calib_key)

                cc.calibrate(calib_key)

            for region in brillouin_regions:
                psc.add_brillouin_region_frequency(region)

            for region in rayleigh_regions:
                psc.add_rayleigh_region_frequency(region)

            evc.set_nr_brillouin_peaks(nr_brillouin_peaks)
            evc.set_bounds(multi_peak_bounds)
            evc.set_bounds_fwhm(multi_peak_bounds_fwhm)

            evc.evaluate()

        return self.session


class ExportController(object):

    def __init__(self):
        return

    @staticmethod
    def get_configuration():
        return {
            'fluorescence': {
                'export': True,
            },
            'fluorescenceCombined': {
                'export': True,
            },
            'brillouin': {
                'export': True,
                # Which Brillouin repetition keys (e.g. '0', '1', ...)
                # to export - None means "all of them" (every caller
                # other than BMicro's export dialog leaves this alone).
                'repetitions': None,
            },
            'surface': {
                'export': True,
            },
            'overviewBrightfield': {
                'export': True,
            },
        }

    def export(self, configuration=None):
        if not configuration:
            configuration = self.get_configuration()

        FluorescenceExport().export(configuration)
        FluorescenceCombinedExport().export(configuration)

        # BrillouinExport needs the EvaluationController
        # to nicely get the data, so we provide it here.
        # Not really nice, but importing it in BrillouinExport
        # leads to a circular dependency.
        BrillouinExport(EvaluationController()).export(configuration)
        SurfaceExport().export(configuration)
        OverviewBrightfieldExport().export(configuration)
