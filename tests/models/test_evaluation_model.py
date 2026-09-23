import numpy as np

from bmlab.models.evaluation_model import EvaluationModel


def test_evaluation_mode_default():
    evm = EvaluationModel()
    assert evm.evaluation_mode == 'single'


def test_initialize_results_arrays_combined_keys():
    """
    'sum'-mode backing store (see EvaluationModel.evaluation_mode):
    initialize_results_arrays() must allocate an all-NaN '_combined'
    array, shaped exactly like its non-combined sibling, for every
    fit-derived Brillouin/Rayleigh quantity - regardless of
    evaluation_mode (they're harmless/unused in 'single' mode).
    """
    evm = EvaluationModel()
    evm.initialize_results_arrays({
        'dim_x': 5,
        'dim_y': 5,
        'dim_z': 5,
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })

    brillouin_keys = (
        'brillouin_peak_position_f', 'brillouin_peak_fwhm_f',
        'brillouin_peak_intensity', 'brillouin_peak_offset',
        'brillouin_peak_snr', 'brillouin_peak_nrmse',
        'brillouin_peak_center_uncertainty', 'brillouin_shift_f',
        'brillouin_shift_f_stokes_anti_stokes')
    rayleigh_keys = (
        'rayleigh_peak_position_f', 'rayleigh_peak_fwhm_f',
        'rayleigh_peak_intensity', 'rayleigh_peak_offset',
        'rayleigh_peak_snr', 'rayleigh_peak_nrmse',
        'rayleigh_peak_center_uncertainty')

    for key in brillouin_keys + rayleigh_keys:
        combined_key = key + '_combined'
        assert combined_key in evm.results
        assert evm.results[combined_key].shape == evm.results[key].shape
        assert np.all(np.isnan(evm.results[combined_key]))

    # Not registered as a separate GUI-selectable parameter.
    for key in brillouin_keys + rayleigh_keys:
        assert (key + '_combined') not in evm.parameters


def test_initialize_results_arrays():
    evm = EvaluationModel()
    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': 5,
        'dim_y': 5,
        'dim_z': 5,
        'nr_images': 2,
        'nr_brillouin_regions': 1,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 1,
    })

    assert evm.results['intensity'].shape == (5, 5, 5, 2, 1, 1)
    assert evm.results['brillouin_peak_position_f'].shape == (5, 5, 5, 2, 1, 1)
    assert evm.results['rayleigh_peak_position_f'].shape == (5, 5, 5, 2, 1, 1)
    assert evm.results['brillouin_shift_f_stokes_anti_stokes'].shape == \
        (5, 5, 5, 2, 1, 1)

    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': 5,
        'dim_y': 5,
        'dim_z': 5,
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })

    assert evm.results['intensity'].shape == (5, 5, 5, 2, 1, 1)
    assert evm.results['brillouin_peak_position_f'].shape == (5, 5, 5, 2, 2, 1)
    assert evm.results['rayleigh_peak_position_f'].shape == (5, 5, 5, 2, 2, 1)

    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': 5,
        'dim_y': 5,
        'dim_z': 5,
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 2,
        'nr_rayleigh_regions': 2,
    })

    assert evm.results['intensity'].shape == (5, 5, 5, 2, 1, 1)
    # If we have more than one Brillouin peak,
    # we store the result of a single peak fit
    # and the result of a multi peak fit, adding up to nr_brillouin_peaks + 1
    assert evm.results['brillouin_peak_position_f'].shape == (5, 5, 5, 2, 2, 3)
    assert evm.results['rayleigh_peak_position_f'].shape == (5, 5, 5, 2, 2, 1)
