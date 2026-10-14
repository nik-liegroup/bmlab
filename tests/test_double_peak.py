import numpy as np

from bmlab.double_peak import has_double_peak


def _lorentz(x, w0, fwhm, amplitude):
    return amplitude * (fwhm / 2) ** 2 / ((x - w0) ** 2 + (fwhm / 2) ** 2)


def _spectrum(peaks, noise=2.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(3e9, 7e9, 200)
    y = 100 + sum(_lorentz(x, *peak) for peak in peaks)
    return x, y + rng.normal(0, noise, x.shape)


REGION = (3e9, 7e9)


def test_single_peak_is_not_a_double_peak():
    x, y = _spectrum([(5e9, 0.4e9, 300)])
    assert has_double_peak(REGION, x, y) == 0.0


def test_clearly_separated_second_peak_is_detected():
    x, y = _spectrum([(4.2e9, 0.3e9, 300), (5.8e9, 0.3e9, 200)])
    assert has_double_peak(REGION, x, y) == 1.0


def test_weak_second_peak_is_ignored():
    x, y = _spectrum([(4.2e9, 0.3e9, 300), (5.8e9, 0.3e9, 40)])
    assert has_double_peak(REGION, x, y) == 0.0


def test_unresolved_shoulder_is_not_a_clear_double_peak():
    x, y = _spectrum([(5.0e9, 0.5e9, 300), (5.15e9, 0.5e9, 150)])
    assert has_double_peak(REGION, x, y) == 0.0


def test_pure_noise_is_not_a_double_peak():
    x, y = _spectrum([], noise=5.0)
    assert has_double_peak(REGION, x, y) == 0.0


def test_too_little_data_is_undetermined():
    x = np.linspace(3e9, 7e9, 5)
    assert np.isnan(has_double_peak(REGION, x, np.ones(5)))


def test_evaluation_stores_double_peak_flag_and_quality_uses_it():
    from tests.test_evaluation_controller import _setup_and_evaluate

    evc, evm = _setup_and_evaluate('single')

    flag = evm.results['brillouin_double_peak']
    assert flag.shape == evm.results['brillouin_peak_position_f'].shape
    # Evaluated points carry 0 or 1, never NaN
    fitted = np.isfinite(evm.results['brillouin_peak_position_f'])
    assert np.all(np.isin(flag[fitted], [0.0, 1.0]))

    # A flagged point fails the (default) threshold, others are not
    # affected by it
    flag[0, 0, 0, :, :, :] = 1.0
    mask, measured = evc.compute_quality_mask()
    assert not mask[0, 0, 0]
    assert 'brillouin_double_peak' in evm.quality_thresholds


def test_old_results_without_double_peak_count_as_not_flagged():
    from bmlab.models.evaluation_model import EvaluationModel

    evm = EvaluationModel()
    shape = (2, 1, 1, 1, 1, 1)
    evm.results['brillouin_peak_position_f'] = np.full(shape, np.nan)
    evm.results['brillouin_peak_position_f'][0] = 5e9
    for key in list(evm.results):
        if key not in ('brillouin_peak_position_f', 'time'):
            evm.results[key] = np.full(shape, np.nan)
    evm.results['time'] = np.full((2, 1, 1, 1, 1, 1), 1.0)
    del evm.results['brillouin_double_peak']

    evm.post_deserialize()

    assert evm.results['brillouin_double_peak'][0, 0, 0, 0, 0, 0] == 0.0
    assert np.isnan(evm.results['brillouin_double_peak'][1, 0, 0, 0, 0, 0])
