import logging
from collections import OrderedDict

import numpy as np

from bmlab.serializer import Serializer

logger = logging.getLogger(__name__)


class BackgroundModel(Serializer):
    """
    Stores fit results for a repetition's background reference points
    (BrillouinAcquisition's second, independent ROI - see the
    bmlab.file.Background/bmlab.controllers.BackgroundController
    docstrings for what these points are).

    Unlike EvaluationModel, background points don't form a rectangular
    grid (there is no resolution-x/y/z for them, each is addressed by
    a plain sequential key), so results are indexed by a flat point
    axis (in the same order as point_keys) instead of (x, y, z).
    Also unlike EvaluationModel, only a single-peak fit is done - these
    are reference points for a quick sanity check against a known
    sample (e.g. water), not a spatial map that would benefit from a
    multi-peak fit.
    """

    def __init__(self):
        # Background image keys, in the order the results arrays below
        # are indexed by (axis 0) - set once per evaluate() call.
        self.point_keys = []
        self.spectra = {}
        self.frequencies = {}

        self.parameters = self.get_default_parameters()
        self.results = {}
        for key in self.parameters.keys():
            self.results[key] = np.empty((0,))

        # Target/stage position (µm) of each point_keys entry - see
        # bmlab.file.MeasurementData.get_position()/get_stage_position().
        self.positions = {}
        self.stage_positions = {}

    def post_deserialize(self):
        if not hasattr(self, 'point_keys'):
            self.point_keys = []
        if not hasattr(self, 'positions'):
            self.positions = {}
        if not hasattr(self, 'stage_positions'):
            self.stage_positions = {}

    @staticmethod
    def get_default_parameters():
        return OrderedDict({
            'brillouin_shift_f': {
                'unit': 'GHz',
                'symbol': r'$\nu_\mathrm{B}$',
                'label': 'Brillouin frequency shift',
                'scaling': 1e-9,
            },
            'brillouin_peak_fwhm_f': {
                'unit': 'GHz',
                'symbol': r'$\Delta_\mathrm{B}$',
                'label': 'Brillouin peak width',
                'scaling': 1e-9,
            },
            'brillouin_peak_position_f': {
                'unit': 'GHz',
                'symbol': r'$s_\mathrm{B}$',
                'label': 'Brillouin peak position',
                'scaling': 1e-9,
            },
            'brillouin_peak_intensity': {
                'unit': 'a.u.',
                'symbol': r'$I_\mathrm{B}$',
                'label': 'Brillouin peak intensity',
                'scaling': 1,
            },
            'brillouin_peak_offset': {
                'unit': 'a.u.',
                'symbol': r'$I_{0,\mathrm{B}}$',
                'label': 'Brillouin peak offset',
                'scaling': 1,
            },
            'rayleigh_peak_fwhm_f': {
                'unit': 'GHz',
                'symbol': r'$\Delta_\mathrm{R}$',
                'label': 'Rayleigh peak width',
                'scaling': 1e-9,
            },
            'rayleigh_peak_position_f': {
                'unit': 'GHz',
                'symbol': r'$s_\mathrm{R}$',
                'label': 'Rayleigh peak position',
                'scaling': 1e-9,
            },
            'rayleigh_peak_intensity': {
                'unit': 'a.u.',
                'symbol': r'$I_\mathrm{R}$',
                'label': 'Rayleigh peak intensity',
                'scaling': 1,
            },
            'rayleigh_peak_offset': {
                'unit': 'a.u.',
                'symbol': r'$I_{0,\mathrm{R}}$',
                'label': 'Rayleigh peak offset',
                'scaling': 1,
            },
            'intensity': {
                'unit': 'a.u.',
                'symbol': r'$I_\mathrm{total}$',
                'label': 'Intensity',
                'scaling': 1,
            },
            'time': {
                'unit': 's',
                'symbol': r'$t$',
                'label': 'Time',
                'scaling': 1,
            },
            # Fit-quality diagnostics - see
            # EvaluationModel.get_default_parameters() for what each
            # one means, they're computed the same way here.
            'brillouin_peak_snr': {
                'unit': 'a.u.',
                'symbol': r'$\mathrm{SNR}_\mathrm{B}$',
                'label': 'Brillouin peak SNR (quality)',
                'scaling': 1,
            },
            'brillouin_peak_nrmse': {
                'unit': '',
                'symbol': r'$\mathrm{NRMSE}_\mathrm{B}$',
                'label': 'Brillouin peak fit NRMSE (quality)',
                'scaling': 1,
            },
            'brillouin_peak_center_uncertainty': {
                'unit': 'GHz',
                'symbol': r'$\sigma_{\nu_\mathrm{B}}$',
                'label': 'Brillouin peak center uncertainty (quality)',
                'scaling': 1e-9,
            },
            'rayleigh_peak_snr': {
                'unit': 'a.u.',
                'symbol': r'$\mathrm{SNR}_\mathrm{R}$',
                'label': 'Rayleigh peak SNR (quality)',
                'scaling': 1,
            },
            'rayleigh_peak_nrmse': {
                'unit': '',
                'symbol': r'$\mathrm{NRMSE}_\mathrm{R}$',
                'label': 'Rayleigh peak fit NRMSE (quality)',
                'scaling': 1,
            },
            'rayleigh_peak_center_uncertainty': {
                'unit': 'GHz',
                'symbol': r'$\sigma_{\nu_\mathrm{R}}$',
                'label': 'Rayleigh peak center uncertainty (quality)',
                'scaling': 1e-9,
            },
        })

    def invalidate_results(self):
        for key in self.parameters:
            self.results[key][:] = np.nan

    def initialize_results_arrays(self, dims):
        shape_general = (dims['nr_points'], dims['nr_images'])

        self.results['intensity'] = np.full(shape_general, np.nan)
        self.results['time'] = np.full(shape_general, np.nan)

        shape_brillouin = (
            dims['nr_points'],
            dims['nr_images'],
            dims['nr_brillouin_regions'],
        )
        for key in (
                'brillouin_peak_position_f', 'brillouin_peak_fwhm_f',
                'brillouin_peak_intensity', 'brillouin_peak_offset',
                'brillouin_shift_f', 'brillouin_peak_snr',
                'brillouin_peak_nrmse',
                'brillouin_peak_center_uncertainty'):
            self.results[key] = np.full(shape_brillouin, np.nan)

        shape_rayleigh = (
            dims['nr_points'],
            dims['nr_images'],
            dims['nr_rayleigh_regions'],
        )
        for key in (
                'rayleigh_peak_position_f', 'rayleigh_peak_fwhm_f',
                'rayleigh_peak_intensity', 'rayleigh_peak_offset',
                'rayleigh_peak_snr', 'rayleigh_peak_nrmse',
                'rayleigh_peak_center_uncertainty'):
            self.results[key] = np.full(shape_rayleigh, np.nan)

        self.positions = {
            axis: np.full(dims['nr_points'], np.nan)
            for axis in ('x', 'y', 'z')
        }
        self.stage_positions = {
            axis: np.full(dims['nr_points'], np.nan)
            for axis in ('x', 'y', 'z')
        }

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

    def get_parameter_keys(self):
        return self.parameters
