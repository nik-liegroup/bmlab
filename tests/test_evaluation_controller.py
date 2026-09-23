import pathlib
import numpy as np

import pytest

from bmlab.controllers import EvaluationController, calculate_derived_values, \
    ExtractionController, CalibrationController, PeakSelectionController
from bmlab.models import CalibrationModel, EvaluationModel, Orientation
from bmlab.models.setup import AVAILABLE_SETUPS


def data_file_path(file_name):
    return pathlib.Path(__file__).parent / 'data' / file_name


def _setup_and_evaluate(evaluation_mode, file_name='Water.h5'):
    """
    Runs the same pipeline as tests/test_run_bmlab_pipeline.py's
    run_pipeline() (real spectra from `file_name`, not a hand-built
    results array), except it sets evm.evaluation_mode before
    evc.evaluate() runs, and returns the controller/model so the test
    can inspect the '_combined' backing arrays directly.
    """
    evc = EvaluationController()
    evc.session.set_file(data_file_path(file_name))
    evc.session.set_current_repetition('0')
    evc.session.set_setup(AVAILABLE_SETUPS[0])
    evc.session.orientation = Orientation(rotation=1, reflection={
        'vertically': False, 'horizontally': False})

    ec = ExtractionController()
    cc = CalibrationController()
    psc = PeakSelectionController()

    ec.find_points_all()
    for calib_key in evc.session.get_calib_keys():
        cc.find_peaks(calib_key)
        cc.calibrate(calib_key)

    psc.add_brillouin_region_frequency((4.0e9, 6.0e9))
    psc.add_brillouin_region_frequency((9.0e9, 11.0e9))
    psc.add_rayleigh_region_frequency((-2.0e9, 2.0e9))
    psc.add_rayleigh_region_frequency((13.0e9, 17.0e9))

    evm = evc.session.evaluation_model()
    evm.evaluation_mode = evaluation_mode
    evc.evaluate()
    return evc, evm


def test_get_key_from_indices():
    resolution = (7, 8, 9)
    assert EvaluationController\
        .get_key_from_indices(resolution, 0, 0, 0) == '0'
    assert EvaluationController\
        .get_key_from_indices(resolution, 0.0, 0.0, 0.0) == '0'
    assert EvaluationController\
        .get_key_from_indices(resolution, 6, 0, 0) == '6'
    assert EvaluationController\
        .get_key_from_indices(resolution, 0, 1, 0) == '7'
    assert EvaluationController\
        .get_key_from_indices(resolution, 1, 1, 0) == '8'
    assert EvaluationController\
        .get_key_from_indices(resolution, 6, 7, 0) == '55'
    assert EvaluationController\
        .get_key_from_indices(resolution, 0, 0, 1) == '56'
    assert EvaluationController\
        .get_key_from_indices(resolution, 0, 1, 1) == '63'
    assert EvaluationController\
        .get_key_from_indices(resolution, 0, 0, 8) == '448'
    assert EvaluationController\
        .get_key_from_indices(resolution, 1, 0, 8) == '449'
    assert EvaluationController\
        .get_key_from_indices(resolution, 1, 1, 8) == '456'
    assert EvaluationController\
        .get_key_from_indices(resolution, 6, 7, 8) == '503'

    # Resolution is not three-dimensional
    with pytest.raises(ValueError) as err:
        EvaluationController \
            .get_key_from_indices((7, 8), 6, 7, 8)
    assert err.typename == 'ValueError'

    # x index is out or range
    with pytest.raises(IndexError) as err:
        EvaluationController \
            .get_key_from_indices((7, 8, 9), 7, 9, 9)
    assert err.typename == 'IndexError'


def test_get_indices_from_key():
    resolution = (7, 8, 9)
    assert EvaluationController\
        .get_indices_from_key(resolution, '0') == (0, 0, 0)
    assert EvaluationController\
        .get_indices_from_key(resolution, '6') == (6, 0, 0)
    assert EvaluationController\
        .get_indices_from_key(resolution, '7') == (0, 1, 0)
    assert EvaluationController\
        .get_indices_from_key(resolution, '8') == (1, 1, 0)
    assert EvaluationController\
        .get_indices_from_key(resolution, '55') == (6, 7, 0)
    assert EvaluationController\
        .get_indices_from_key(resolution, '56') == (0, 0, 1)
    assert EvaluationController\
        .get_indices_from_key(resolution, '63') == (0, 1, 1)
    assert EvaluationController\
        .get_indices_from_key(resolution, '448') == (0, 0, 8)
    assert EvaluationController\
        .get_indices_from_key(resolution, '449') == (1, 0, 8)
    assert EvaluationController\
        .get_indices_from_key(resolution, '456') == (1, 1, 8)
    assert EvaluationController\
        .get_indices_from_key(resolution, '503') == (6, 7, 8)

    # Key is invalid
    with pytest.raises(ValueError) as err:
        EvaluationController \
            .get_indices_from_key((7, 8, 9), '504')
    assert err.typename == 'ValueError'


def test_evaluate():
    evaluationcontroller = EvaluationController()
    evaluationcontroller.evaluate()


def test_calculate_derived_values_equal_region_count():
    evc = EvaluationController()
    evc.session.set_file(data_file_path('Water.h5'))
    evc.session.set_current_repetition('0')
    evm = evc.session.evaluation_model()

    """
    Test calculation in case of same number
    Brillouin and Rayleigh regions
    """
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

    evm.results['brillouin_peak_position_f'][:] = 1
    evm.results['rayleigh_peak_position_f'][:] = 3

    calculate_derived_values()

    assert (evm.results['brillouin_shift_f'] == 2).all()

    evm.results['brillouin_peak_position_f'][:, :, :, :, 0, :] = 1
    evm.results['brillouin_peak_position_f'][:, :, :, :, 1, :] = 4
    evm.results['rayleigh_peak_position_f'][:, :, :, :, 0, :] = -1
    evm.results['rayleigh_peak_position_f'][:, :, :, :, 1, :] = 8

    calculate_derived_values()

    assert (evm.results['brillouin_shift_f'][:, :, :, :, 0, :] == 2).all()
    assert (evm.results['brillouin_shift_f'][:, :, :, :, 1, :] == 4).all()


def test_calculate_derived_values_equal_region_count_nr_peaks_2():
    evc = EvaluationController()
    evc.session.set_file(data_file_path('Water.h5'))
    evc.session.set_current_repetition('0')
    evm = evc.session.evaluation_model()

    """
    Test calculation in case of same number
    Brillouin and Rayleigh regions
    """
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

    evm.results['brillouin_peak_position_f'][:] = 1
    evm.results['rayleigh_peak_position_f'][:] = 3

    calculate_derived_values()

    assert (evm.results['brillouin_shift_f'] == 2).all()

    evm.results['brillouin_peak_position_f'][:, :, :, :, 0, :] = 1
    evm.results['brillouin_peak_position_f'][:, :, :, :, 1, :] = 4
    evm.results['brillouin_peak_position_f'][:, :, :, :, 0, 1] = 2
    evm.results['brillouin_peak_position_f'][:, :, :, :, 1, 1] = 5
    evm.results['rayleigh_peak_position_f'][:, :, :, :, 0, :] = -1
    evm.results['rayleigh_peak_position_f'][:, :, :, :, 1, :] = 8

    calculate_derived_values()

    assert (evm.results['brillouin_shift_f'][:, :, :, :, 0, 0] == 2).all()
    assert (evm.results['brillouin_shift_f'][:, :, :, :, 1, 0] == 4).all()
    assert (evm.results['brillouin_shift_f'][:, :, :, :, 0, 1] == 3).all()
    assert (evm.results['brillouin_shift_f'][:, :, :, :, 1, 1] == 3).all()


def test_calculate_derived_values_stokes_anti_stokes():
    """
    Brillouin shift from the distance between the two Brillouin
    (Stokes/Anti-Stokes) peaks, corrected for the VIPA spectrum's free
    spectral range (FSR): since the Anti-Stokes peak sits near the
    *next* Rayleigh order, distance(Anti-Stokes, Stokes) = FSR - 2 *
    shift, not 2 * shift. Only defined for exactly 2 Brillouin regions,
    and does not depend on this frame's own (possibly noisy) Rayleigh
    peak position - only on the FSR from the calibration fit.
    """
    evc = EvaluationController()
    evc.session.set_file(data_file_path('Water.h5'))
    evc.session.set_current_repetition('0')
    evm = evc.session.evaluation_model()
    cm = evc.session.calibration_model()
    # FSR = 20 (arbitrary units, same as the peak positions below) -
    # a fixed calibration/instrument property, not a per-frame fit.
    cm.vipa_params['0'] = [(0, 0, 0, 20)]

    evm.initialize_results_arrays({
        'dim_x': 5,
        'dim_y': 5,
        'dim_z': 5,
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })

    # Stokes region at 2, Anti-Stokes region at 14 -> distance 12,
    # shift = (FSR - distance) / 2 = (20 - 12) / 2 = 4.
    # Rayleigh peaks are set far off from the true laser line to show
    # the result does not depend on them.
    evm.results['brillouin_peak_position_f'][:, :, :, :, 0, :] = 2
    evm.results['brillouin_peak_position_f'][:, :, :, :, 1, :] = 14
    evm.results['rayleigh_peak_position_f'][:] = 100

    calculate_derived_values()

    key = 'brillouin_shift_f_stokes_anti_stokes'
    assert (evm.results[key][:, :, :, :, 0, :] == 4).all()
    assert (evm.results[key][:, :, :, :, 1, :] == 4).all()


def test_calculate_derived_values_stokes_anti_stokes_wrong_region_count():
    """
    The Stokes/Anti-Stokes shift is only defined for exactly 2 Brillouin
    regions - with any other count, it must be left NaN rather than
    guessing.
    """
    evc = EvaluationController()
    evc.session.set_file(data_file_path('Water.h5'))
    evc.session.set_current_repetition('0')
    evm = evc.session.evaluation_model()
    cm = evc.session.calibration_model()
    cm.vipa_params['0'] = [(0, 0, 0, 20)]

    evm.initialize_results_arrays({
        'dim_x': 5,
        'dim_y': 5,
        'dim_z': 5,
        'nr_images': 2,
        'nr_brillouin_regions': 3,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })

    evm.results['brillouin_peak_position_f'][:] = 1
    evm.results['rayleigh_peak_position_f'][:] = 3

    calculate_derived_values()

    assert np.isnan(
        evm.results['brillouin_shift_f_stokes_anti_stokes']).all()


def test_calculate_derived_values_stokes_anti_stokes_no_calibration():
    """
    Without a fitted calibration (no FSR available), the Stokes/Anti-
    Stokes shift must be left NaN rather than falling back to some
    other assumption.
    """
    evc = EvaluationController()
    evc.session.set_file(data_file_path('Water.h5'))
    evc.session.set_current_repetition('0')
    evm = evc.session.evaluation_model()

    evm.initialize_results_arrays({
        'dim_x': 5,
        'dim_y': 5,
        'dim_z': 5,
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })

    evm.results['brillouin_peak_position_f'][:, :, :, :, 0, :] = 2
    evm.results['brillouin_peak_position_f'][:, :, :, :, 1, :] = 14
    evm.results['rayleigh_peak_position_f'][:] = 100

    calculate_derived_values()

    assert np.isnan(
        evm.results['brillouin_shift_f_stokes_anti_stokes']).all()


def test_calculate_derived_values_quality_metrics():
    """
    brillouin_shift_frame_spread/rayleigh_shift_frame_spread (max-min
    across frames, broadcast back to every frame) - see
    EvaluationModel.get_default_parameters(). The other quality
    metrics (brillouin_peak_snr etc.) are set directly by the fit
    itself, not here - see test_evaluate_sets_quality_diagnostics().
    """
    evc = EvaluationController()
    evc.session.set_file(data_file_path('Water.h5'))
    evc.session.set_current_repetition('0')
    evm = evc.session.evaluation_model()

    evm.initialize_results_arrays({
        'dim_x': 2,
        'dim_y': 2,
        'dim_z': 1,
        'nr_images': 2,
        'nr_brillouin_regions': 1,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 1,
    })

    # Two frames disagreeing by a known amount on both peaks' positions.
    # brillouin_shift_f is |brillouin_peak_position_f -
    # rayleigh_peak_position_f|, so its frame spread combines both:
    # frame 0 shift = |5.0 - 0.0| = 5.0, frame 1 shift = |5.2 - 0.5| =
    # 4.7, spread = 0.3.
    evm.results['brillouin_peak_position_f'][:, :, :, 0, :, :] = 5.0
    evm.results['brillouin_peak_position_f'][:, :, :, 1, :, :] = 5.2
    evm.results['rayleigh_peak_position_f'][:, :, :, 0, :, :] = 0.0
    evm.results['rayleigh_peak_position_f'][:, :, :, 1, :, :] = 0.5

    calculate_derived_values()

    np.testing.assert_allclose(
        evm.results['brillouin_shift_frame_spread'],
        0.3 * np.ones_like(evm.results['brillouin_shift_frame_spread']))
    np.testing.assert_allclose(
        evm.results['rayleigh_shift_frame_spread'],
        0.5 * np.ones_like(evm.results['rayleigh_shift_frame_spread']))

    # Both frames' broadcast-back copies must agree (the spread is a
    # per-point value, not really per-frame).
    np.testing.assert_allclose(
        evm.results['brillouin_shift_frame_spread'][:, :, :, 0],
        evm.results['brillouin_shift_frame_spread'][:, :, :, 1])

    # And it must round-trip through get_data() (peak-index slicing,
    # frame/region averaging, unit scaling) without raising, same as
    # any other registered parameter.
    data, positions, dimensionality, labels = evc.get_data(
        'brillouin_shift_frame_spread', 0)
    np.testing.assert_allclose(data, 0.3 * 1e-9 * np.ones_like(data))


def test_quality_thresholds():
    """
    set_quality_threshold()/compute_quality_mask()/
    apply_quality_thresholds() - the API BMicro's Quality tab drives.
    A point that was never measured ('time' is NaN, e.g. outside the
    ROI) must never count as passing OR failing; among measured
    points, only the enabled threshold(s) decide.
    """
    evc = EvaluationController()
    evc.session.set_file(data_file_path('Water.h5'))
    evc.session.set_current_repetition('0')
    evm = evc.session.evaluation_model()

    evm.initialize_results_arrays({
        'dim_x': 2,
        'dim_y': 2,
        'dim_z': 1,
        'nr_images': 1,
        'nr_brillouin_regions': 1,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 1,
    })
    # This test is about set_quality_threshold()/compute_quality_mask()
    # mechanics, not about what the defaults are (see
    # test_default_quality_thresholds() for that) - start from a clean
    # slate rather than get_default_quality_thresholds(), since this
    # test only ever populates 'brillouin_peak_snr' and every other
    # default-enabled metric would otherwise fail every point (NaN
    # never satisfies a threshold).
    evm.quality_thresholds = {}

    # (0, 0) and (1, 0): measured, good SNR - should pass.
    # (0, 1): measured, bad SNR - should fail.
    # (1, 1): never measured (time stays NaN) - not pass, not fail.
    evm.results['time'][0, 0, 0, 0, 0, 0] = 1.0
    evm.results['time'][1, 0, 0, 0, 0, 0] = 1.0
    evm.results['time'][0, 1, 0, 0, 0, 0] = 1.0
    evm.results['brillouin_peak_snr'][0, 0, 0, 0, 0, 0] = 2.0
    evm.results['brillouin_peak_snr'][1, 0, 0, 0, 0, 0] = 2.0
    evm.results['brillouin_peak_snr'][0, 1, 0, 0, 0, 0] = 0.1

    # No threshold enabled yet - every measured point passes.
    mask, measured = evc.compute_quality_mask()
    assert measured.sum() == 3
    assert mask.sum() == 3

    evc.set_quality_threshold('brillouin_peak_snr', min_value=0.5)
    assert evc.session.evaluation_model()\
        .quality_thresholds['brillouin_peak_snr'] == \
        {'enabled': True, 'min': 0.5, 'max': None}

    mask, measured = evc.compute_quality_mask()
    assert mask[:, :, 0].tolist() == [[True, False], [True, False]]

    evc.apply_quality_thresholds()
    quality_pass = evm.results['quality_pass'][:, :, 0, 0, 0, 0]
    assert quality_pass[0, 0] == 1.0
    assert quality_pass[1, 0] == 1.0
    assert quality_pass[0, 1] == 0.0
    assert np.isnan(quality_pass[1, 1])

    # A disabled threshold is ignored entirely.
    evc.set_quality_threshold('brillouin_peak_snr', enabled=False)
    mask, measured = evc.compute_quality_mask()
    assert mask.sum() == measured.sum() == 3

    # Explicitly clearing a bound (min_value=None) must actually clear
    # it, not be treated as "field not passed, leave it as it was" -
    # this is what unchecking a threshold's checkbox in the GUI does.
    evc.set_quality_threshold(
        'brillouin_peak_snr', enabled=True, min_value=0.5)
    mask, measured = evc.compute_quality_mask()
    assert mask.sum() == 2
    evc.set_quality_threshold('brillouin_peak_snr', min_value=None)
    assert evc.session.evaluation_model()\
        .quality_thresholds['brillouin_peak_snr']['min'] is None
    mask, measured = evc.compute_quality_mask()
    assert mask.sum() == measured.sum() == 3


def test_default_quality_thresholds():
    """
    A fresh EvaluationModel (new file, no prior thresholds saved)
    starts with the data-driven defaults, not an empty dict - and
    compute_quality_mask() can evaluate them (including
    brillouin_shift_frame_spread/rayleigh_shift_frame_spread, which
    only calculate_derived_values() populates - see
    initialize_results_arrays()) without raising, even before that
    ever runs.
    """
    evm = EvaluationModel()
    assert evm.quality_thresholds == \
        EvaluationModel.get_default_quality_thresholds()

    evc = EvaluationController()
    evc.session.set_file(data_file_path('Water.h5'))
    evc.session.set_current_repetition('0')
    evm = evc.session.evaluation_model()
    evm.initialize_results_arrays({
        'dim_x': 2,
        'dim_y': 2,
        'dim_z': 1,
        'nr_images': 1,
        'nr_brillouin_regions': 1,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 1,
    })
    assert evm.quality_thresholds == \
        EvaluationModel.get_default_quality_thresholds()

    evm.results['time'][0, 0, 0, 0, 0, 0] = 1.0
    mask, measured = evc.compute_quality_mask()
    assert measured.sum() == 1
    # Every default-enabled metric is still NaN (nothing fitted this
    # point yet) - NaN never satisfies a threshold, so it fails.
    assert mask.sum() == 0


def test_quality_metric_keys_matches_default_thresholds():
    """
    Regression/drift guard: EvaluationController.QUALITY_METRIC_KEYS
    (which metrics the Quality tab shows a row for) and
    EvaluationModel.get_default_quality_thresholds() (which of those
    get a default enabled threshold) are two independently maintained
    collections - this pins their only allowed difference
    ('brillouin_peak_fwhm_f' deliberately has no default threshold,
    see that method's own docstring), so an edit to one that forgets
    the other fails loudly here instead of silently drifting.
    """
    default_threshold_keys = set(
        EvaluationModel.get_default_quality_thresholds().keys())
    assert set(EvaluationController.QUALITY_METRIC_KEYS) - \
        default_threshold_keys == {'brillouin_peak_fwhm_f'}
    assert default_threshold_keys.issubset(
        EvaluationController.QUALITY_METRIC_KEYS)


def test_calculate_derived_values_different_region_count():
    evc = EvaluationController()
    evc.session.set_file(data_file_path('Water.h5'))
    evc.session.set_current_repetition('0')
    evm = evc.session.evaluation_model()

    """
    Test calculation in case of more Rayleigh than
    Brillouin regions
    """
    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': 5,
        'dim_y': 5,
        'dim_z': 5,
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 3,
    })

    evm.results['brillouin_peak_position_f'][:, :, :, :, 0, :] = 1
    evm.results['brillouin_peak_position_f'][:, :, :, :, 1, :] = 4
    evm.results['rayleigh_peak_position_f'][:, :, :, :, 0, :] = -2
    evm.results['rayleigh_peak_position_f'][:, :, :, :, 1, :] = 9
    evm.results['rayleigh_peak_position_f'][:, :, :, :, 2, :] = 10

    calculate_derived_values()

    assert (evm.results['brillouin_shift_f'][:, :, :, :, 0, :] == 3).all()
    assert (evm.results['brillouin_shift_f'][:, :, :, :, 1, :] == 5).all()


def test_calculate_derived_values_different_region_count_nr_peaks_2():
    evc = EvaluationController()
    evc.session.set_file(data_file_path('Water.h5'))
    evc.session.set_current_repetition('0')
    evm = evc.session.evaluation_model()

    """
    Test calculation in case of more Rayleigh than
    Brillouin regions
    """
    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': 5,
        'dim_y': 5,
        'dim_z': 5,
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 2,
        'nr_rayleigh_regions': 3,
    })

    evm.results['brillouin_peak_position_f'][:, :, :, :, 0, :] = 1
    evm.results['brillouin_peak_position_f'][:, :, :, :, 1, :] = 4
    evm.results['brillouin_peak_position_f'][:, :, :, :, 0, 1] = 2
    evm.results['brillouin_peak_position_f'][:, :, :, :, 1, 1] = 5
    evm.results['rayleigh_peak_position_f'][:, :, :, :, 0, :] = -2
    evm.results['rayleigh_peak_position_f'][:, :, :, :, 1, :] = 9
    evm.results['rayleigh_peak_position_f'][0:4, :, :, :, 2, :] = 10
    evm.results['rayleigh_peak_position_f'][4, :, :, :, 2, :] = 8

    calculate_derived_values()

    assert (evm.results['brillouin_shift_f'][:, :, :, :, 0, 0] == 3).all()
    assert (evm.results['brillouin_shift_f'][0:4, :, :, :, 1, 0] == 5).all()
    assert (evm.results['brillouin_shift_f'][4, :, :, :, 1, 0] == 4).all()
    assert (evm.results['brillouin_shift_f'][:, :, :, :, 0, 1] == 4).all()
    assert (evm.results['brillouin_shift_f'][0:4, :, :, :, 1, 1] == 4).all()
    assert (evm.results['brillouin_shift_f'][4, :, :, :, 1, 1] == 3).all()


def test_get_data_0D(mocker):
    evc = EvaluationController()
    evc.session.set_file(data_file_path('0D.h5'))
    evc.session.set_current_repetition('0')

    evm = evc.session.evaluation_model()

    resolution = evc.session.get_payload_resolution()
    assert resolution == (1, 1, 1)

    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': resolution[0],
        'dim_y': resolution[1],
        'dim_z': resolution[2],
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })
    evm.results['brillouin_shift_f'][:] = 1e9  # [Hz]

    data, positions, dimensionality, labels = evc.get_data('brillouin_shift_f')

    assert data == 1  # [GHz]
    assert positions[0][0, 0, 0] == -1
    assert positions[1][0, 0, 0] == -2
    assert positions[2][0, 0, 0] == -3
    assert dimensionality == 0
    assert labels == [r'$x$ [$\mu$m]', r'$y$ [$\mu$m]', r'$z$ [$\mu$m]']


def test_get_data_no_valid_grid():
    """
    Regression test: EvaluationController.get_data() must not raise
    for a repetition with no valid measurement grid (e.g. an aborted/
    restarted acquisition with resolution attributes but no
    positions-x/y/z datasets) - it should return a clean (None, None,
    None, None) sentinel instead, so callers can bail out gracefully.
    """
    evc = EvaluationController()
    evc.session.set_file(
        data_file_path('aborted_repetition0_no_positions.h5'))
    evc.session.set_current_repetition('0')

    data, positions, dimensionality, labels = \
        evc.get_data('brillouin_shift_f')

    assert data is None
    assert positions is None
    assert dimensionality is None
    assert labels is None


def test_get_data_1D_x(mocker):
    evc = EvaluationController()
    evc.session.set_file(data_file_path('1D-x.h5'))
    evc.session.set_current_repetition('0')

    evm = evc.session.evaluation_model()

    resolution = evc.session.get_payload_resolution()
    assert resolution == (3, 1, 1)

    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': resolution[0],
        'dim_y': resolution[1],
        'dim_z': resolution[2],
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })
    evm.results['brillouin_shift_f'][:] = 1e9  # [Hz]

    data, positions, dimensionality, labels = evc.get_data('brillouin_shift_f')

    positions_x = np.zeros(shape=resolution)
    positions_x[:, 0, 0] = [-1, 0, 1]
    positions_y = np.zeros(shape=resolution)
    positions_y[:] = -2
    positions_z = np.zeros(shape=resolution)
    positions_z[:] = -3

    assert (data == np.ones(shape=resolution)).all()  # [GHz]
    assert (positions[0] == positions_x).all()
    assert (positions[1] == positions_y).all()
    assert (positions[2] == positions_z).all()
    assert dimensionality == 1
    assert labels == [r'$x$ [$\mu$m]', r'$y$ [$\mu$m]', r'$z$ [$\mu$m]']


def test_get_data_1D_y(mocker):
    evc = EvaluationController()
    evc.session.set_file(data_file_path('1D-y.h5'))
    evc.session.set_current_repetition('0')

    evm = evc.session.evaluation_model()

    resolution = evc.session.get_payload_resolution()
    assert resolution == (1, 5, 1)

    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': resolution[0],
        'dim_y': resolution[1],
        'dim_z': resolution[2],
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })
    evm.results['brillouin_shift_f'][:] = 1e9  # [Hz]

    data, positions, dimensionality, labels = evc.get_data('brillouin_shift_f')

    positions_x = np.zeros(shape=resolution)
    positions_x[:] = -1
    positions_y = np.zeros(shape=resolution)
    positions_y[0, :, 0] = [-2, -1, 0, 1, 2]
    positions_z = np.zeros(shape=resolution)
    positions_z[:] = -3

    assert (data == np.ones(shape=resolution)).all()  # [GHz]
    assert (positions[0] == positions_x).all()
    assert (positions[1] == positions_y).all()
    assert (positions[2] == positions_z).all()
    assert dimensionality == 1
    assert labels == [r'$x$ [$\mu$m]', r'$y$ [$\mu$m]', r'$z$ [$\mu$m]']


def test_get_data_1D_z(mocker):
    evc = EvaluationController()
    evc.session.set_file(data_file_path('1D-z.h5'))
    evc.session.set_current_repetition('0')

    evm = evc.session.evaluation_model()

    resolution = evc.session.get_payload_resolution()
    assert resolution == (1, 1, 7)

    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': resolution[0],
        'dim_y': resolution[1],
        'dim_z': resolution[2],
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })
    evm.results['brillouin_shift_f'][:] = 1e9  # [Hz]

    data, positions, dimensionality, labels = evc.get_data('brillouin_shift_f')

    positions_x = np.zeros(shape=resolution)
    positions_x[:] = -1
    positions_y = np.zeros(shape=resolution)
    positions_y[:] = -2
    positions_z = np.zeros(shape=resolution)
    positions_z[0, 0, :] = [-3, -2, -1, 0, 1, 2, 3]

    assert (data == np.ones(shape=resolution)).all()  # [GHz]
    assert (positions[0] == positions_x).all()
    assert (positions[1] == positions_y).all()
    assert (positions[2] == positions_z).all()
    assert dimensionality == 1
    assert labels == [r'$x$ [$\mu$m]', r'$y$ [$\mu$m]', r'$z$ [$\mu$m]']


def test_get_data_2D_xy(mocker):
    evc = EvaluationController()
    evc.session.set_file(data_file_path('2D-xy.h5'))
    evc.session.set_current_repetition('0')

    evm = evc.session.evaluation_model()

    resolution = evc.session.get_payload_resolution()
    assert resolution == (3, 5, 1)

    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': resolution[0],
        'dim_y': resolution[1],
        'dim_z': resolution[2],
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })
    evm.results['brillouin_shift_f'][:] = 1e9  # [Hz]

    data, positions, dimensionality, labels = evc.get_data('brillouin_shift_f')

    data_expected = np.ones(shape=resolution)
    (pos_y, pos_x, pos_z) = np.meshgrid([-2, -1, 0, 1, 2], [-1, 0, 1], [-3])

    assert (data == data_expected).all()
    assert (positions[0] == pos_x).all()
    assert (positions[1] == pos_y).all()
    assert (positions[2] == pos_z).all()
    assert dimensionality == 2
    assert labels == [r'$x$ [$\mu$m]', r'$y$ [$\mu$m]', r'$z$ [$\mu$m]']


def test_get_data_2D_xz(mocker):
    evc = EvaluationController()
    evc.session.set_file(data_file_path('2D-xz.h5'))
    evc.session.set_current_repetition('0')

    evm = evc.session.evaluation_model()

    resolution = evc.session.get_payload_resolution()
    assert resolution == (3, 1, 7)

    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': resolution[0],
        'dim_y': resolution[1],
        'dim_z': resolution[2],
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })
    evm.results['brillouin_shift_f'][:] = 1e9  # [Hz]

    data, positions, dimensionality, labels = evc.get_data('brillouin_shift_f')

    data_expected = np.ones(shape=resolution)
    (pos_y, pos_x, pos_z) = np.meshgrid(
        [-2], [-1, 0, 1], [-3, -2, -1, 0, 1, 2, 3]
    )

    assert (data == data_expected).all()
    assert (positions[0] == pos_x).all()
    assert (positions[1] == pos_y).all()
    assert (positions[2] == pos_z).all()
    assert dimensionality == 2
    assert labels == [r'$x$ [$\mu$m]', r'$y$ [$\mu$m]', r'$z$ [$\mu$m]']


def test_get_data_2D_yz(mocker):
    evc = EvaluationController()
    evc.session.set_file(data_file_path('2D-yz.h5'))
    evc.session.set_current_repetition('0')

    evm = evc.session.evaluation_model()

    resolution = evc.session.get_payload_resolution()
    assert resolution == (1, 5, 7)

    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': resolution[0],
        'dim_y': resolution[1],
        'dim_z': resolution[2],
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })
    evm.results['brillouin_shift_f'][:] = 1e9  # [Hz]

    data, positions, dimensionality, labels = evc.get_data('brillouin_shift_f')

    data_expected = np.ones(shape=resolution)
    (pos_y, pos_x, pos_z) = np.meshgrid(
        [-2, -1, 0, 1, 2], [-1], [-3, -2, -1, 0, 1, 2, 3]
    )

    assert (data == data_expected).all()
    assert (positions[0] == pos_x).all()
    assert (positions[1] == pos_y).all()
    assert (positions[2] == pos_z).all()
    assert dimensionality == 2
    assert labels == [r'$x$ [$\mu$m]', r'$y$ [$\mu$m]', r'$z$ [$\mu$m]']


def test_get_data_3D(mocker):
    evc = EvaluationController()
    evc.session.set_file(data_file_path('3D.h5'))
    evc.session.set_current_repetition('0')

    evm = evc.session.evaluation_model()

    resolution = evc.session.get_payload_resolution()
    assert resolution == (3, 5, 7)

    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': resolution[0],
        'dim_y': resolution[1],
        'dim_z': resolution[2],
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 2,
    })
    evm.results['brillouin_shift_f'][:] = 1e9  # [Hz]

    data, positions, dimensionality, labels = evc.get_data('brillouin_shift_f')

    data_expected = np.ones(resolution)
    (pos_y, pos_x, pos_z) = np.meshgrid(
        [-2, -1, 0, 1, 2],
        [-1, 0, 1],
        [-3, -2, -1, 0, 1, 2, 3]
    )

    assert dimensionality == 3
    assert (data == data_expected).all()  # [GHz]
    assert (positions[0] == pos_x).all()
    assert (positions[1] == pos_y).all()
    assert (positions[2] == pos_z).all()
    assert labels == [r'$x$ [$\mu$m]', r'$y$ [$\mu$m]', r'$z$ [$\mu$m]']


def test_get_data_3D_peak_index():
    evc = EvaluationController()
    evc.session.set_file(data_file_path('3D.h5'))
    evc.session.set_current_repetition('0')

    evm = evc.session.evaluation_model()

    resolution = evc.session.get_payload_resolution()

    # Initialize results array
    evm.initialize_results_arrays({
        'dim_x': resolution[0],
        'dim_y': resolution[1],
        'dim_z': resolution[2],
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_brillouin_peaks': 2,
        'nr_rayleigh_regions': 2,
    })
    evm.results['brillouin_shift_f'][:, :, :, :, :, 0] = 2e9  # [Hz]
    evm.results['brillouin_shift_f'][:, :, :, :, :, 1] = 4e9  # [Hz]
    evm.results['brillouin_shift_f'][:, :, :, :, :, 2] = 6e9  # [Hz]

    evm.results['brillouin_peak_fwhm_f'][:, :, :, :, :, 0] = 1e9  # [Hz]
    evm.results['brillouin_peak_fwhm_f'][:, :, :, :, :, 1] = 1e9  # [Hz]
    evm.results['brillouin_peak_fwhm_f'][:, :, :, :, :, 2] = 2e9  # [Hz]

    evm.results['brillouin_peak_intensity'][:, :, :, :, :, 0] = 1e9  # [Hz]
    evm.results['brillouin_peak_intensity'][:, :, :, :, :, 1] = 2e9  # [Hz]
    evm.results['brillouin_peak_intensity'][:, :, :, :, :, 2] = 3e9  # [Hz]

    # Get first peak
    data, positions, dimensionality, labels =\
        evc.get_data('brillouin_shift_f')
    assert data.shape == resolution
    assert (data == 2 * np.ones(resolution)).all()  # [GHz]

    data, positions, dimensionality, labels =\
        evc.get_data('brillouin_shift_f', 0)
    assert data.shape == resolution
    assert (data == 2 * np.ones(resolution)).all()  # [GHz]

    # Get first peak of multi-peak fit
    data, positions, dimensionality, labels =\
        evc.get_data('brillouin_shift_f', 1)
    assert data.shape == resolution
    assert (data == 4 * np.ones(resolution)).all()  # [GHz]

    # Get second peak of multi-peak fit
    data, positions, dimensionality, labels =\
        evc.get_data('brillouin_shift_f', 2)
    assert data.shape == resolution
    assert (data == 6 * np.ones(resolution)).all()  # [GHz]

    # Get average of multi-peak fits
    data, positions, dimensionality, labels =\
        evc.get_data('brillouin_shift_f', 3)
    assert data.shape == resolution
    assert (data == 5 * np.ones(resolution)).all()  # [GHz]

    # Get weighted average of multi-peak fits
    data, positions, dimensionality, labels =\
        evc.get_data('brillouin_shift_f', 4)
    assert data.shape == resolution
    assert (data == 5.5 * np.ones(resolution)).all()  # [GHz]

    # Get single-peak fit when out of bounds
    data, positions, dimensionality, labels =\
        evc.get_data('brillouin_shift_f', 5)
    assert data.shape == resolution
    assert (data == 2 * np.ones(resolution)).all()  # [GHz]


def test_create_bounds(mocker):
    cm = CalibrationModel()
    evm = EvaluationModel()
    mocker.patch('bmlab.session.Session.calibration_model', return_value=cm)
    mocker.patch('bmlab.session.Session.evaluation_model', return_value=evm)
    evc = EvaluationController()

    # Set the calibration data
    cm.set_frequencies('0', 0,
                       list(1e9 * np.linspace(
                           -1, 16, 681).reshape(1, -1)))
    cm.set_frequencies('1', 1,
                       list(1e9 * np.linspace(
                           -0.975, 16.025, 681).reshape(1, -1)))
    cm.set_frequencies('2', 2,
                       list(1e9 * np.linspace(
                           -0.95, 16.05, 681).reshape(1, -1)))

    # If we don't supply regions and times, we get no bounds
    fit_bounds = EvaluationController().create_bounds(None, None)
    assert fit_bounds is None

    # Test that we get correct bounds for regions
    # that only contain one type of peaks (either Stokes or Anti-Stokes)
    evc.set_bounds([['min', '5'], ['5.5', 'Inf'], ['-inf', 'max']])

    brillouin_regions = [(3.3e9, 7.0e9), (8.0e9, 12.1e9)]
    rayleigh_peaks = 1e9 * np.array([[0.0, 0.1, -0.2], [14.8, 15.0, 15.3]])
    fit_bounds = evc.create_bounds(brillouin_regions, rayleigh_peaks)

    np.testing.assert_allclose([
        [
            [[3.3e9, 5.0e9], [5.5e9, np.inf], [-np.inf, 7.0e9]],
            [[3.3e9, 5.1e9], [5.6e9, np.inf], [-np.inf, 7.0e9]],
            [[3.3e9, 4.8e9], [5.3e9, np.inf], [-np.inf, 7.0e9]]
        ], [
            [[9.8e9, 12.1e9], [-np.inf, 9.3e9], [8.0e9, np.inf]],
            [[10.0e9, 12.1e9], [-np.inf, 9.5e9], [8.0e9, np.inf]],
            [[10.3e9, 12.1e9], [-np.inf, 9.8e9], [8.0e9, np.inf]]
        ]
    ], fit_bounds, atol=1e6)

    # Test that we get the correct bounds for regions
    # that contain both Stokes and Anti-Stokes peaks
    evc.set_bounds([['min', '5'], ['5.5', 'Inf'], ['-inf', 'max'],
                    ['min', '-5'], ['-5.5', 'Inf'], ['-inf', 'max']])

    brillouin_regions = [(2.3e9, 12.1e9), (3.3e9, 12.1e9)]
    rayleigh_peaks = 1e9 * np.array([[0.0, 0.1, -0.2], [14.8, 15.0, 15.3]])
    fit_bounds = evc.create_bounds(brillouin_regions, rayleigh_peaks)

    np.testing.assert_allclose([
        [
            [[2.3e9, 5.0e9], [5.5e9, np.inf], [-np.inf, 12.1e9],
             [9.8e9, 12.1e9], [-np.inf, 9.3e9], [-np.inf, 12.1e9]],
            [[2.3e9, 5.1e9], [5.6e9, np.inf], [-np.inf, 12.1e9],
             [10.0e9, 12.1e9], [-np.inf, 9.5e9], [-np.inf, 12.1e9]],
            [[2.3e9, 4.8e9], [5.3e9, np.inf], [-np.inf, 12.1e9],
             [10.3e9, 12.1e9], [-np.inf, 9.8e9], [-np.inf, 12.1e9]]
        ],
        [
            [[3.3e9, 5.0e9], [5.5e9, np.inf], [-np.inf, 12.1e9],
             [9.8e9, 12.1e9], [-np.inf, 9.3e9], [-np.inf, 12.1e9]],
            [[3.3e9, 5.1e9], [5.6e9, np.inf], [-np.inf, 12.1e9],
             [10.0e9, 12.1e9], [-np.inf, 9.5e9], [-np.inf, 12.1e9]],
            [[3.3e9, 4.8e9], [5.3e9, np.inf], [-np.inf, 12.1e9],
             [10.3e9, 12.1e9], [-np.inf, 9.8e9], [-np.inf, 12.1e9]]
        ]
    ], fit_bounds, atol=1e6)

    # Test real values for a two peak fit
    evc.set_bounds([['3.5', '5'], ['5.0', '7.5']])

    brillouin_regions = [(3.3e9, 7.5e9), (8.0e9, 12.1e9)]
    rayleigh_peaks = 1e9 * np.array([[0.0, 0.1, -0.2], [14.8, 15.0, 15.3]])
    fit_bounds = evc.create_bounds(brillouin_regions, rayleigh_peaks)

    np.testing.assert_allclose([
        [
            [[3.5e9, 5.0e9], [5.0e9, 7.5e9]],
            [[3.6e9, 5.1e9], [5.1e9, 7.6e9]],
            [[3.3e9, 4.8e9], [4.8e9, 7.3e9]]
        ], [
            [[9.8e9, 11.3e9], [7.3e9, 9.8e9]],
            [[10.0e9, 11.5e9], [7.5e9, 10.0e9]],
            [[10.3e9, 11.8e9], [7.8e9, 10.3e9]]
        ]
    ], fit_bounds, atol=1e6)

    # Test real values for a four peak fit over the whole spectrum
    evc.set_bounds([['3.5', '5'], ['5.0', '7.5'],
                    ['-7.5', '-5.0'], ['-5', '-3.5']])

    brillouin_regions = [(2.3e9, 12.1e9), (3.3e9, 12.1e9)]
    rayleigh_peaks = 1e9 * np.array([[0.0, 0.1, -0.2], [14.8, 15.0, 15.3]])
    fit_bounds = evc.create_bounds(brillouin_regions, rayleigh_peaks)

    np.testing.assert_allclose([
        [
            [[3.5e9, 5.0e9], [5.0e9, 7.5e9],
             [7.3e9, 9.8e9], [9.8e9, 11.3e9]],
            [[3.6e9, 5.1e9], [5.1e9, 7.6e9],
             [7.5e9, 10.0e9], [10.0e9, 11.5e9]],
            [[3.3e9, 4.8e9], [4.8e9, 7.3e9],
             [7.8e9, 10.3e9], [10.3e9, 11.8e9]]
        ], [
            [[3.5e9, 5.0e9], [5.0e9, 7.5e9],
             [7.3e9, 9.8e9], [9.8e9, 11.3e9]],
            [[3.6e9, 5.1e9], [5.1e9, 7.6e9],
             [7.5e9, 10.0e9], [10.0e9, 11.5e9]],
            [[3.3e9, 4.8e9], [4.8e9, 7.3e9],
             [7.8e9, 10.3e9], [10.3e9, 11.8e9]]
        ]
    ], fit_bounds, atol=1e6)


def test_create_bounds_fwhm(mocker):
    cm = CalibrationModel()
    evm = EvaluationModel()
    mocker.patch('bmlab.session.Session.calibration_model', return_value=cm)
    mocker.patch('bmlab.session.Session.evaluation_model', return_value=evm)
    evc = EvaluationController()

    # If we don't supply regions and times, we get no bounds
    fit_bounds = EvaluationController().create_bounds_fwhm(None, None)
    assert fit_bounds is None

    # Test that we get correct bounds for regions
    # that only contain one type of peaks (either Stokes or Anti-Stokes)
    evc.set_bounds_fwhm([['min', '-0.9'], ['2.0', 'Inf'], ['-inf', 'max']])

    brillouin_regions = [(3.3e9, 7.0e9), (8.0e9, 12.1e9)]
    rayleigh_peaks = 1e9 * np.array([[0.0, 0.1, -0.2], [14.8, 15.0, 15.3]])
    fit_bounds_fwhm = evc.create_bounds_fwhm(brillouin_regions, rayleigh_peaks)

    np.testing.assert_allclose([
        [
            [[0, 0.9e9], [2.0e9, np.inf], [0, np.inf]],
            [[0, 0.9e9], [2.0e9, np.inf], [0, np.inf]],
            [[0, 0.9e9], [2.0e9, np.inf], [0, np.inf]]
        ], [
            [[0, 0.9e9], [2.0e9, np.inf], [0, np.inf]],
            [[0, 0.9e9], [2.0e9, np.inf], [0, np.inf]],
            [[0, 0.9e9], [2.0e9, np.inf], [0, np.inf]]
        ]
    ], fit_bounds_fwhm, atol=1e6)


def test_evaluation_mode_default_is_single():
    evm = EvaluationModel()
    assert evm.evaluation_mode == 'single'


def test_evaluation_mode_single_preserves_current_behavior():
    """
    (a) Regression-safety: the default 'single' mode must reproduce
    exactly the pre-existing behavior (get_data() is the nanmean,
    across frames, of each frame's own separately-fit peak position),
    unaffected by the new (harmless, unwritten-to) '_combined' backing
    arrays that initialize_results_arrays() now always allocates.
    """
    evc, evm = _setup_and_evaluate('single')

    shift, _, _, _ = evc.get_data('brillouin_shift_f')
    assert shift.size != 0
    np.testing.assert_allclose(shift[~np.isnan(shift)], 5.03, atol=0.05)

    # The combined backing arrays exist but were never written to in
    # 'single' mode - stay all-NaN.
    assert np.all(
        np.isnan(evm.results['brillouin_peak_position_f_combined']))
    assert np.all(np.isnan(evm.results['brillouin_shift_f_combined']))


def test_evaluation_mode_sum_uses_combined_fit():
    """
    (b) 'sum' mode: the combined-fit backing arrays get populated by
    evaluate(), and get_data() transparently redirects to them (same
    physical result as 'single' mode here, since Water.h5's frames are
    all clean measurements of the same peak).
    """
    evc, evm = _setup_and_evaluate('sum')

    assert not np.all(
        np.isnan(evm.results['brillouin_peak_position_f_combined']))
    assert not np.all(
        np.isnan(evm.results['rayleigh_peak_position_f_combined']))

    shift, _, _, _ = evc.get_data('brillouin_shift_f')
    assert shift.size != 0
    np.testing.assert_allclose(shift[~np.isnan(shift)], 5.03, atol=0.05)


def test_evaluation_mode_sum_preserves_frame_spread_qc():
    """
    (c) 'sum' mode must not break the frame-spread QC diagnostic: every
    frame is still fit separately underneath (see evaluate()), so
    brillouin_shift_frame_spread/rayleigh_shift_frame_spread are still
    computed and non-trivial (not all-NaN), exactly as in 'single'
    mode.
    """
    evc, evm = _setup_and_evaluate('sum')

    for key in ('brillouin_shift_frame_spread',
                'rayleigh_shift_frame_spread'):
        data = evm.results[key]
        assert data.size != 0
        finite = data[~np.isnan(data)]
        assert finite.size > 0, f'{key} is all-NaN in sum mode'


def test_evaluation_mode_sum_recovers_true_center_at_least_as_well():
    """
    (b) On noisy synthetic multi-frame data, fitting the sum of the
    frames' spectra (EvaluationController.fit_spectrum_combined())
    should recover the true peak center at least as well as averaging
    the separately-fit per-frame centers (EvaluationController.
    fit_spectra(), the 'single'-mode behavior) - the statistical
    motivation for 'sum' mode (closer to a sufficient statistic for
    the shared peak position under Poisson counting statistics).
    """
    rng = np.random.default_rng(0)
    x = np.linspace(-5, 5, 200)
    true_w0 = 0.3
    true_fwhm = 1.0
    true_intensity = 50.0
    # A comfortably positive offset (rather than e.g. 2.0) keeps the
    # summed-spectrum's edge samples - summing 5 frames' worth of noise
    # raises their variance well above a single frame's - from wandering
    # negative, which would otherwise occasionally violate fit_lorentz's
    # own offset >= 0 initial-guess bound and make this test flaky.
    true_offset = 10.0

    def lorentzian(xx):
        return true_offset + true_intensity * (
            (true_fwhm / 2) ** 2 /
            ((xx - true_w0) ** 2 + (true_fwhm / 2) ** 2))

    n_frames = 5
    noise_sigma = 4.0
    frames = [lorentzian(x) + rng.normal(0, noise_sigma, x.size)
              for _ in range(n_frames)]
    frequencies = [x for _ in range(n_frames)]
    region = (x[0], x[-1])

    # Per-frame fits, naively averaged - the 'single'-mode behavior.
    per_frame_fits = EvaluationController.fit_spectra(
        frames, frequencies, region)
    naive_average = np.nanmean([f[0] for f in per_frame_fits])

    # Sum-then-fit - the 'sum'-mode behavior.
    combined_fit = EvaluationController.fit_spectrum_combined(
        frames, frequencies, region)
    combined_center = combined_fit[0]

    assert abs(combined_center - true_w0) <= \
        abs(naive_average - true_w0) + 0.05


def test_combine_spectra_sums_interpolated_frames():
    """
    _combine_spectra() must: (1) use frequencies[0] as the reference
    axis, (2) correctly interpolate a frame whose frequency axis runs
    in the opposite (decreasing) direction, and (3) not let a single
    NaN pixel in one frame poison the sum at that pixel in every other
    frame (np.nansum-style per-pixel handling).
    """
    from bmlab.controllers import _combine_spectra

    reference = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    frame0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    # Same physical spectrum, but frequency axis decreasing.
    frame1_freq = reference[::-1]
    frame1 = frame0[::-1]
    # A third frame with a NaN at its lowest-frequency pixel - masking
    # it out (rather than e.g. propagating it) shrinks that frame's
    # usable frequency range, so at the reference axis's own lowest
    # point (freq=0, now outside frame2's masked range) frame2
    # contributes nothing (np.interp's left=np.nan) instead of a
    # fabricated/extrapolated value.
    frame2 = np.array([np.nan, 2.0, 3.0, 4.0, 5.0])

    summed, ref_out = _combine_spectra(
        [frame0, frame1, frame2],
        [reference, frame1_freq, reference])

    np.testing.assert_allclose(ref_out, reference)
    # Pixel 0 (freq=0): frame0 = 1, frame1 (interpolated back) = 1,
    # frame2 = NaN (out of its masked range) -> sum of the two valid
    # frames only, not a NaN-poisoned/fabricated-zero result.
    np.testing.assert_allclose(summed[0], 2.0)
    # Every other pixel: all three frames have valid, agreeing data.
    np.testing.assert_allclose(summed[1], 6.0)
    np.testing.assert_allclose(summed[2], 9.0)


def test_post_deserialize_migrates_evaluation_mode_and_combined_keys():
    """
    (d) An old saved session predating evaluation_mode/the '_combined'
    results keys must migrate cleanly: evaluation_mode defaults to
    'single' and every '_combined' key is backfilled as an all-NaN
    array shaped like its non-combined sibling.
    """
    evm = EvaluationModel()
    evm.initialize_results_arrays({
        'dim_x': 2,
        'dim_y': 2,
        'dim_z': 1,
        'nr_images': 2,
        'nr_brillouin_regions': 1,
        'nr_brillouin_peaks': 1,
        'nr_rayleigh_regions': 1,
    })

    # Simulate a pre-0.14.0 session: no evaluation_mode attribute, and
    # none of the '_combined' result keys.
    del evm.evaluation_mode
    combined_keys = [key for key in evm.results if key.endswith('_combined')]
    assert combined_keys, 'no _combined keys to migrate - test is stale'
    for key in combined_keys:
        del evm.results[key]

    evm.post_deserialize()

    assert evm.evaluation_mode == 'single'
    for key in combined_keys:
        assert key in evm.results
        assert evm.results[key].shape == \
            evm.results[key.replace('_combined', '')].shape
        assert np.all(np.isnan(evm.results[key]))
