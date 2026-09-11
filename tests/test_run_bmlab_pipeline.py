import pathlib

import numpy as np

from bmlab.controllers import Controller
from bmlab.models import Orientation
from bmlab.models.setup import AVAILABLE_SETUPS


def run_pipeline(file_name='Water.h5'):

    filepath = pathlib.Path(__file__).parent / 'data' / file_name
    setup = AVAILABLE_SETUPS[0]
    orientation = Orientation(rotation=1, reflection={
            'vertically': False, 'horizontally': False
        })

    # Frequency ranges to evaluate in GHz
    brillouin_regions = [(4.0e9, 6.0e9), (9.0e9, 11.0e9)]
    rayleigh_regions = [(-2.0e9, 2.0e9), (13.0e9, 17.0e9)]

    session = Controller().evaluate(
        filepath,
        setup,
        orientation,
        brillouin_regions,
        rayleigh_regions
    )

    return session


def test_run_pipeline():
    session = run_pipeline()
    evm = session.evaluation_model()
    shift = evm.results['brillouin_shift_f']
    assert shift.size != 0
    np.testing.assert_allclose(shift, 5.03e9, atol=50E6)


def test_run_pipeline_sets_quality_diagnostics():
    """
    Regression test: the full evaluate() pipeline (real spectra, not a
    hand-built results array) must populate brillouin_peak_snr/_nrmse/
    _center_uncertainty and their rayleigh_ equivalents with real,
    finite values - not leave them at their initialize_results_arrays()
    NaN default. See bmlab.fits._fit_noise_and_covariance()/
    _param_uncertainty() and EvaluationController.evaluate().
    """
    session = run_pipeline()
    evm = session.evaluation_model()

    for key in ('brillouin_peak_snr', 'brillouin_peak_nrmse',
                'brillouin_peak_center_uncertainty',
                'rayleigh_peak_snr', 'rayleigh_peak_nrmse',
                'rayleigh_peak_center_uncertainty'):
        data = evm.results[key]
        assert data.size != 0
        finite = data[~np.isnan(data)]
        assert finite.size > 0, f'{key} is all-NaN'
        assert np.all(finite >= 0), f'{key} has a negative value'

    # Water.h5 is a clean, strong single spectrum - SNR should be well
    # above the noise floor.
    assert np.nanmedian(evm.results['brillouin_peak_snr']) > 1


def test_run_pipeline_with_missing_first_image_key():
    """
    Regression test: a surface-following / ROI scan can skip the
    lowest-index grid point (e.g. it was outside the ROI, or the
    surface wasn't found there), so payload image key '0' need not
    exist. Code must use the first *available* image key instead of
    assuming '0' always exists (see EvaluationController.evaluate(),
    Session.set_image_shape(), PeakSelectionView.refresh_plot()).
    Water_sparse.h5 is a copy of Water.h5 with image key '0' deleted.
    """
    session = run_pipeline('Water_sparse.h5')
    assert session.get_image_keys()[0] != '0'
    evm = session.evaluation_model()
    shift = evm.results['brillouin_shift_f']
    assert shift.size != 0
    # The remaining 9 (of 10) measurement points should still fit fine
    valid = shift[~np.isnan(shift)]
    assert valid.size > 0
    np.testing.assert_allclose(valid, 5.03e9, atol=50E6)
