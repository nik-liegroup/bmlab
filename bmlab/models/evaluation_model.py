import logging
import numpy as np
from collections import OrderedDict

from bmlab.serializer import Serializer

logger = logging.getLogger(__name__)


class EvaluationModel(Serializer):

    def __init__(self):

        self.nr_brillouin_peaks = 1
        self.spectra = {}
        self.frequencies = {}

        # @since 0.1.0
        self.parameters = self.get_default_parameters()
        # @since 0.8.0
        self.bounds_w0 = None
        # @since 0.8.0
        self.bounds_fwhm = None
        # @since 0.13.0
        # Per-metric quality thresholds set from BMicro's Quality tab,
        # e.g. {'brillouin_shift_frame_spread': {'enabled': True,
        # 'max': 1.5e8}} - see EvaluationController.apply_quality_
        # thresholds(). Kept here (not just in the GUI) so it round-
        # trips through a saved session like every other setting.
        # Starts at get_default_quality_thresholds(), not {} - see
        # that method's docstring for where those numbers come from.
        self.quality_thresholds = self.get_default_quality_thresholds()

        self.results = {}
        for key in self.parameters.keys():
            self.results[key] = np.empty((0,))

    def post_deserialize(self):
        # Migrations from 0.0.13 to 0.1.0
        # Check that the parameters attribute is present
        # @since 0.1.0
        if not hasattr(self, 'parameters')\
                or not isinstance(self, OrderedDict):
            self.parameters = self.get_default_parameters()
        # Migrations from 0.1.7 to 0.1.8
        # Check that the bounds attribute is present
        # @since 0.1.8
        if not hasattr(self, 'bounds'):
            self.bounds = None
        # Migrations from 0.3.0 to 0.4.0
        # Check that the results array also stores the peak offset
        # @since 0.4.0
        if 'brillouin_peak_offset' not in self.results:
            self.results['brillouin_peak_offset'] = np.empty(
                self.results['brillouin_peak_intensity'].shape
            )
            self.results['brillouin_peak_offset'][:] = np.nan
        if 'rayleigh_peak_offset' not in self.results:
            self.results['rayleigh_peak_offset'] = np.empty(
                self.results['rayleigh_peak_intensity'].shape
            )
            self.results['rayleigh_peak_offset'][:] = np.nan
        # Migrations from 0.4.0 to 0.5.0
        # @since 0.5.0
        if 'rayleigh_shift' not in self.results:
            # Stores how much the Rayleigh peak moved
            # Used for shifting the regions evaluated
            self.results['rayleigh_shift'] = np.empty(
                self.results['rayleigh_peak_intensity'].shape
            )
        # Migrations from 0.5.1 to 0.6.0
        # @since 0.6.0
        if 'brillouin_peak_position_f' not in self.results:
            self.results['brillouin_peak_position_f'] = np.empty(
                self.results['brillouin_peak_position'].shape
            )
            self.results['brillouin_peak_position_f'][:] = np.nan
        if 'rayleigh_peak_position_f' not in self.results:
            self.results['rayleigh_peak_position_f'] = np.empty(
                self.results['rayleigh_peak_position'].shape
            )
            self.results['rayleigh_peak_position_f'][:] = np.nan
        del_keys = [
            'brillouin_shift',
            'brillouin_peak_fwhm',
            'brillouin_peak_position',
            'rayleigh_peak_fwhm',
            'rayleigh_peak_position'
        ]
        for key in del_keys:
            if key in self.results:
                del self.results[key]
        # Migrations from 0.7.0 to 0.8.0
        # @since 0.8.0
        if not hasattr(self, 'bounds_w0'):
            self.bounds_w0 = self.bounds
        if hasattr(self, 'bounds'):
            delattr(self, 'bounds')
        if not hasattr(self, 'bounds_fwhm'):
            self.bounds_fwhm = None

        # Migrations from 0.11.0 to 0.12.0
        # @since 0.12.0
        if not hasattr(self, 'frequencies'):
            self.frequencies = {}

        # Migrations from 0.12.3 to 0.13.0
        # @since 0.13.0
        if 'brillouin_shift_frame_spread' not in self.results:
            self.results['brillouin_shift_frame_spread'] = np.full(
                self.results['brillouin_peak_position_f'].shape, np.nan)
        if 'rayleigh_shift_frame_spread' not in self.results:
            self.results['rayleigh_shift_frame_spread'] = np.full(
                self.results['rayleigh_peak_position_f'].shape, np.nan)
        for key in ('brillouin_peak_snr', 'brillouin_peak_nrmse',
                    'brillouin_peak_center_uncertainty'):
            if key not in self.results:
                self.results[key] = np.full(
                    self.results['brillouin_peak_position_f'].shape, np.nan)
        for key in ('rayleigh_peak_snr', 'rayleigh_peak_nrmse',
                    'rayleigh_peak_center_uncertainty'):
            if key not in self.results:
                self.results[key] = np.full(
                    self.results['rayleigh_peak_position_f'].shape, np.nan)
        # A stray 'brillouin_snr' key (the old, less accurate
        # intensity/offset SNR proxy this replaces) can only exist in
        # sessions saved by a pre-release build of this feature - drop
        # it rather than carrying dead data forward.
        self.results.pop('brillouin_snr', None)
        if 'quality_pass' not in self.results:
            self.results['quality_pass'] = np.full(
                self.results['time'].shape, np.nan)
        if not hasattr(self, 'quality_thresholds'):
            self.quality_thresholds = self.get_default_quality_thresholds()

    def invalidate_results(self):
        for key in self.parameters:
            self.results[key][:] = np.nan

    @staticmethod
    def get_default_parameters():
        return OrderedDict({
            'brillouin_shift_f': {          # [GHz] Brillouin frequency shift
                'unit': 'GHz',
                'symbol': r'$\nu_\mathrm{B}$',
                'label': 'Brillouin frequency shift',
                'scaling': 1e-9,
            },
            'brillouin_peak_fwhm_f': {      # [GHz] Brillouin peak FWHM
                'unit': 'GHz',
                'symbol': r'$\Delta_\mathrm{B}$',
                'label': 'Brillouin peak width',
                'scaling': 1e-9,
            },
            'brillouin_peak_position_f': {    # [GHz] Brillouin peak position
                'unit': 'GHz',
                'symbol': r'$s_\mathrm{B}$',
                'label': 'Brillouin peak position',
                'scaling': 1e-9,
            },
            'brillouin_peak_intensity': {   # [a.u.] Brillouin peak intensity
                'unit': 'a.u.',
                'symbol': r'$I_\mathrm{B}$',
                'label': 'Brillouin peak intensity',
                'scaling': 1,
            },
            'brillouin_peak_offset': {   # [a.u.] Brillouin peak offset
                'unit': 'a.u.',
                'symbol': r'$I_{0,\mathrm{B}}$',
                'label': 'Brillouin peak offset',
                'scaling': 1,
            },
            'rayleigh_peak_fwhm_f': {       # [GHz] Rayleigh peak FWHM
                'unit': 'GHz',
                'symbol': r'$\Delta_\mathrm{R}$',
                'label': 'Rayleigh peak width',
                'scaling': 1e-9,
            },
            'rayleigh_peak_position_f': {     # [GHz] Rayleigh peak position
                'unit': 'GHz',
                'symbol': r'$s_\mathrm{R}$',
                'label': 'Rayleigh peak position',
                'scaling': 1e-9,
            },
            'rayleigh_peak_intensity': {    # [a.u.] Rayleigh peak intensity
                'unit': 'a.u.',
                'symbol': r'$I_\mathrm{R}$',
                'label': 'Rayleigh peak intensity',
                'scaling': 1,
            },
            'rayleigh_peak_offset': {   # [a.u.] Rayleigh peak offset
                'unit': 'a.u.',
                'symbol': r'$I_{0,\mathrm{R}}$',
                'label': 'Rayleigh peak offset',
                'scaling': 1,
            },
            'intensity': {                  # [a.u.] Overall intensity of image
                'unit': 'a.u.',
                'symbol': r'$I_\mathrm{total}$',
                'label': 'Intensity',
                'scaling': 1,
            },
            'time': {                       # [s] The time the measurement
                'unit': 's',                # point was taken at
                'symbol': r'$t$',
                'label': 'Time',
                'scaling': 1,
            },
            # Quality metrics - a converged least_squares fit to noise
            # looks the same, numerically, as a fit to a real peak, so
            # none of the "real" parameters above can tell good fits
            # from bad ones on their own. brillouin_peak_snr/_nrmse/
            # _center_uncertainty (and their rayleigh_ equivalents) are
            # computed directly by the fit itself, from its own raw
            # residuals (bmlab.fits._fit_noise_and_covariance()/
            # _param_uncertainty()) - NOT from e.g. intensity/offset,
            # since a spectrum can have a high peak/background ratio
            # and still be extremely noisy. The *_frame_spread metrics
            # below are a complementary, independent check: computed
            # after fitting (calculate_derived_values()), from how much
            # separately-fit frames at the same point disagree.
            'brillouin_peak_snr': {
                # [a.u.] Fitted peak amplitude / residual noise SD
                # (std of y - fitted_curve, in the fitted region) - how
                # far the peak actually rises above fluctuations the
                # model can't explain.
                'unit': 'a.u.',
                'symbol': r'$\mathrm{SNR}_\mathrm{B}$',
                'label': 'Brillouin peak SNR (quality)',
                'scaling': 1,
            },
            'brillouin_peak_nrmse': {
                # [-] Residual noise SD normalized by the fitted peak
                # amplitude (the inverse of brillouin_peak_snr, kept as
                # its own metric since a threshold on it reads directly
                # as "residual is at most X% of the peak height").
                # Low = clean, well-described peak. High = noisy
                # spectrum and/or a poor Lorentzian description of it.
                'unit': '',
                'symbol': r'$\mathrm{NRMSE}_\mathrm{B}$',
                'label': 'Brillouin peak fit NRMSE (quality)',
                'scaling': 1,
            },
            'brillouin_peak_center_uncertainty': {
                # [GHz] The fitted peak center's own standard error,
                # from the fit's parameter covariance - whether the
                # Brillouin shift could actually be localized
                # precisely, independent of whether the fit converged.
                'unit': 'GHz',
                'symbol': r'$\sigma_{\nu_\mathrm{B}}$',
                'label': 'Brillouin peak center uncertainty (quality)',
                'scaling': 1e-9,
            },
            'rayleigh_peak_snr': {
                # [a.u.] Same as brillouin_peak_snr, for the Rayleigh
                # fit - the Brillouin shift is computed relative to the
                # Rayleigh peak position, so a noisy Rayleigh fit
                # corrupts the shift even when the Brillouin fit itself
                # looks clean.
                'unit': 'a.u.',
                'symbol': r'$\mathrm{SNR}_\mathrm{R}$',
                'label': 'Rayleigh peak SNR (quality)',
                'scaling': 1,
            },
            'rayleigh_peak_nrmse': {
                # [-] Same as brillouin_peak_nrmse, for the Rayleigh fit.
                'unit': '',
                'symbol': r'$\mathrm{NRMSE}_\mathrm{R}$',
                'label': 'Rayleigh peak fit NRMSE (quality)',
                'scaling': 1,
            },
            'rayleigh_peak_center_uncertainty': {
                # [GHz] Same as brillouin_peak_center_uncertainty, for
                # the Rayleigh fit.
                'unit': 'GHz',
                'symbol': r'$\sigma_{\nu_\mathrm{R}}$',
                'label': 'Rayleigh peak center uncertainty (quality)',
                'scaling': 1e-9,
            },
            'brillouin_shift_frame_spread': {
                # [GHz] How much independently-fit frames at the same
                # point disagree on the Brillouin shift (max - min
                # across frames) - catches cases where a fit looks
                # locally plausible (good SNR, low center uncertainty)
                # but has actually jumped to a different, spurious
                # peak between frames.
                'unit': 'GHz',
                'symbol': r'$\Delta\nu_\mathrm{B,frames}$',
                'label': 'Brillouin shift frame spread (quality)',
                'scaling': 1e-9,
            },
            'rayleigh_shift_frame_spread': {
                # [GHz] Same idea as brillouin_shift_frame_spread, but
                # for the Rayleigh peak position.
                'unit': 'GHz',
                'symbol': r'$\Delta\nu_\mathrm{R,frames}$',
                'label': 'Rayleigh position frame spread (quality)',
                'scaling': 1e-9,
            },
            'quality_pass': {
                # [-] 1 = passes every enabled quality threshold, 0 =
                # fails at least one, NaN = never measured - see
                # EvaluationController.apply_quality_thresholds().
                'unit': '',
                'symbol': r'$Q$',
                'label': 'Quality check passed',
                'scaling': 1,
            },
        })

    @staticmethod
    def get_default_quality_thresholds():
        """
        Starting thresholds for the quality metrics in
        get_default_parameters() above - not universal constants, a
        reasonable starting point to adjust from, not something to
        trust blindly on a new sample/setup.

        brillouin_shift_frame_spread's 0.2 GHz max is the one
        empirically grounded number here: inspecting a real dataset
        (Xenopus brain tissue, 1938 measured points) showed a clean
        bimodal split - a well-behaved cluster from 0-0.12 GHz, a
        near-empty gap, then a distinct population of clear fit
        failures from 1.7-2.9 GHz - and 0.2 GHz sits right in that
        gap. rayleigh_shift_frame_spread mirrors it, since it's the
        same reproducibility check on the other peak. SNR >= 5 and
        center_uncertainty <= 0.08 GHz are user-set starting points.
        NRMSE's default (residual <= peak amplitude) is a physically-
        reasoned starting point, not yet checked against real data the
        same way. No default is set for brillouin_peak_fwhm_f - FWHM
        can carry real biological information and isn't recommended as
        a primary filter (see the Brillouin peak fitting discussion
        this was designed around).
        """
        return {
            'brillouin_shift_frame_spread': {
                'enabled': True, 'min': None, 'max': 0.2},
            'rayleigh_shift_frame_spread': {
                'enabled': True, 'min': None, 'max': 0.2},
            'brillouin_peak_snr': {
                'enabled': True, 'min': 5.0, 'max': None},
            'rayleigh_peak_snr': {
                'enabled': True, 'min': 5.0, 'max': None},
            'brillouin_peak_nrmse': {
                'enabled': True, 'min': None, 'max': 1.0},
            'rayleigh_peak_nrmse': {
                'enabled': True, 'min': None, 'max': 1.0},
            'brillouin_peak_center_uncertainty': {
                'enabled': True, 'min': None, 'max': 0.08},
            'rayleigh_peak_center_uncertainty': {
                'enabled': True, 'min': None, 'max': 0.08},
        }

    def initialize_results_arrays(self, dims):
        shape_general = (
            dims['dim_x'],
            dims['dim_y'],
            dims['dim_z'],
            dims['nr_images'],
            1,  # We just add this so it matches the ndims of the
            1,  # Brillouin array and reshapes are reduced
        )

        self.results['intensity'] = np.empty(shape_general)
        self.results['intensity'][:] = np.nan

        self.results['time'] = np.empty(shape_general)
        self.results['time'][:] = np.nan

        # We always do a single-peak fit, plus a multi-peak fit if requested.
        # Hence, we have to store
        # (nr_brillouin_peaks + 1) peaks, if nr_brillouin_peaks > 1.
        nr_brillouin_peaks_to_store = dims['nr_brillouin_peaks']
        if dims['nr_brillouin_peaks'] > 1:
            nr_brillouin_peaks_to_store = nr_brillouin_peaks_to_store + 1

        shape_brillouin = (
            dims['dim_x'],
            dims['dim_y'],
            dims['dim_z'],
            dims['nr_images'],
            dims['nr_brillouin_regions'],
            nr_brillouin_peaks_to_store,
        )

        self.results['brillouin_peak_position_f'] = np.empty(shape_brillouin)
        self.results['brillouin_peak_position_f'][:] = np.nan

        self.results['brillouin_peak_intensity'] = np.empty(shape_brillouin)
        self.results['brillouin_peak_intensity'][:] = np.nan

        self.results['brillouin_peak_offset'] = np.empty(shape_brillouin)
        self.results['brillouin_peak_offset'][:] = np.nan

        self.results['brillouin_shift_f'] = np.empty(shape_brillouin)
        self.results['brillouin_shift_f'][:] = np.nan

        self.results['brillouin_peak_fwhm_f'] = np.empty(shape_brillouin)
        self.results['brillouin_peak_fwhm_f'][:] = np.nan

        # Fit-quality diagnostics, computed directly by the fit itself
        # (see bmlab.fits._fit_noise_and_covariance()/
        # _param_uncertainty()) - unlike brillouin_shift_frame_spread
        # etc. below, these need no separate derived-value pass.
        self.results['brillouin_peak_snr'] = np.empty(shape_brillouin)
        self.results['brillouin_peak_snr'][:] = np.nan

        self.results['brillouin_peak_nrmse'] = np.empty(shape_brillouin)
        self.results['brillouin_peak_nrmse'][:] = np.nan

        self.results['brillouin_peak_center_uncertainty'] = np.empty(
            shape_brillouin)
        self.results['brillouin_peak_center_uncertainty'][:] = np.nan

        # Derived (not fit-direct) quality diagnostic - only
        # controllers.calculate_derived_values() actually computes
        # this, from already-fitted peak positions, but it must exist
        # here too: get_default_quality_thresholds() enables a
        # threshold on it by default, and compute_quality_mask()
        # evaluates every enabled threshold's metric via get_data(),
        # which would otherwise find no array (or, worse, a
        # differently-shaped leftover one from whatever this
        # EvaluationModel's `results` dict held before) for a
        # repetition that hasn't gone through calculate_derived_values
        # yet.
        self.results['brillouin_shift_frame_spread'] = np.empty(
            shape_brillouin)
        self.results['brillouin_shift_frame_spread'][:] = np.nan

        shape_rayleigh = (
            dims['dim_x'],
            dims['dim_y'],
            dims['dim_z'],
            dims['nr_images'],
            dims['nr_rayleigh_regions'],
            1,  # We just add this so it matches the ndims of the
                #  Brillouin array and reshapes are reduced
        )

        self.results['rayleigh_peak_position_f'] = np.empty(shape_rayleigh)
        self.results['rayleigh_peak_position_f'][:] = np.nan

        self.results['rayleigh_peak_intensity'] = np.empty(shape_rayleigh)
        self.results['rayleigh_peak_intensity'][:] = np.nan

        self.results['rayleigh_peak_offset'] = np.empty(shape_rayleigh)
        self.results['rayleigh_peak_offset'][:] = np.nan

        self.results['rayleigh_peak_fwhm_f'] = np.empty(shape_rayleigh)
        self.results['rayleigh_peak_fwhm_f'][:] = np.nan

        self.results['rayleigh_shift'] = np.empty(shape_rayleigh)
        self.results['rayleigh_shift'][:] = np.nan

        self.results['rayleigh_peak_snr'] = np.empty(shape_rayleigh)
        self.results['rayleigh_peak_snr'][:] = np.nan

        self.results['rayleigh_peak_nrmse'] = np.empty(shape_rayleigh)
        self.results['rayleigh_peak_nrmse'][:] = np.nan

        self.results['rayleigh_peak_center_uncertainty'] = np.empty(
            shape_rayleigh)
        self.results['rayleigh_peak_center_uncertainty'][:] = np.nan

        # See brillouin_shift_frame_spread above.
        self.results['rayleigh_shift_frame_spread'] = np.empty(
            shape_rayleigh)
        self.results['rayleigh_shift_frame_spread'][:] = np.nan

    def set_spectra(self, image_key, spectra):
        self.spectra[image_key] = spectra

    def get_spectra(self, image_key):
        spectra = self.spectra.get(image_key)
        if spectra:
            return spectra
        return None

    def set_frequencies(self, image_key, frequencies):
        self.frequencies[image_key] = frequencies

    def get_frequencies(self, image_key):
        frequencies = self.frequencies.get(image_key)
        if frequencies:
            return frequencies
        return None

    def get_fits(self, ind_x, ind_y, ind_z):
        return (self.results['brillouin_peak_position_f'][
               ind_x, ind_y, ind_z, :, :, :],
               self.results['brillouin_peak_fwhm_f'][
               ind_x, ind_y, ind_z, :, :, :],
               self.results['brillouin_peak_intensity'][
               ind_x, ind_y, ind_z, :, :, :],
               self.results['brillouin_peak_offset'][
               ind_x, ind_y, ind_z, :, :, :]), \
               (self.results['rayleigh_peak_position_f'][
                ind_x, ind_y, ind_z, :, :, :],
                self.results['rayleigh_peak_fwhm_f'][
                ind_x, ind_y, ind_z, :, :, :],
                self.results['rayleigh_peak_intensity'][
                ind_x, ind_y, ind_z, :, :, :],
                self.results['rayleigh_peak_offset'][
                ind_x, ind_y, ind_z, :, :, :])

    def get_parameter_keys(self):
        return self.parameters

    def setNrBrillouinPeaks(self, nr_brillouin_peaks):
        self.nr_brillouin_peaks = nr_brillouin_peaks

        self.check_bounds()

    def set_bounds(self, bounds):
        self.bounds_w0 = bounds

        self.check_bounds()

    def check_bounds(self):
        # Check the bounds array for consistency
        # We don't need any bounds for a single-peak fit
        if self.nr_brillouin_peaks == 1:
            self.bounds_w0 = None
            self.bounds_fwhm = None

        # Initialize the bounds if necessary
        if self.nr_brillouin_peaks > 1 and\
                (self.bounds_w0 is None or
                 len(self.bounds_w0) is not self.nr_brillouin_peaks):
            self.bounds_w0 = [['min', 'max'] for _ in
                              range(self.nr_brillouin_peaks)]
        if self.nr_brillouin_peaks > 1 and\
                (self.bounds_fwhm is None or
                 len(self.bounds_fwhm) is not self.nr_brillouin_peaks):
            self.bounds_fwhm = [['0', 'inf'] for _ in
                                range(self.nr_brillouin_peaks)]
