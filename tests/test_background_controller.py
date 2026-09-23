import pathlib
import shutil

import h5py
import numpy as np

from bmlab.controllers import BackgroundController, CalibrationController, \
    ExtractionController, PeakSelectionController
from bmlab.models import Orientation
from bmlab.session import Session


def data_file_path(file_name):
    return pathlib.Path(__file__).parent / 'data' / file_name


def _make_background_fixture(tmp_path):
    """
    Water.h5 already has an (empty) 'background' group (see
    bmlab.file.Background) - it predates the BrillouinAcquisition
    version that first wrote real background points into it. Copies
    payload point '0's own spectrum dataset into background/data/'0'
    so BackgroundController has a real (if synthetic) point to fit -
    since it's a byte-for-byte copy of a known-good payload spectrum,
    the fitted Brillouin shift should land on the same ~5.03 GHz value
    test_run_bmlab_pipeline.py already expects for Water.h5.
    """
    src = data_file_path('Water.h5')
    dst = tmp_path / 'water_with_background.h5'
    shutil.copy(src, dst)

    with h5py.File(dst, 'r+') as h5:
        payload_point = h5['Brillouin/0/payload/data/0']
        background_data = h5['Brillouin/0/background/data']
        h5.copy(payload_point, background_data, name='0')
        # get_channel()/get_position() expect these - not present on a
        # plain payload point copy.
        background_data['0'].attrs['channel'] = \
            np.array([b'Background'], dtype='S')
        background_data['0'].attrs['position_x_um'] = [1.0]
        background_data['0'].attrs['position_y_um'] = [2.0]
        background_data['0'].attrs['position_z_um'] = [3.0]

    return dst


def test_background_controller_evaluate(tmp_path):
    fixture = _make_background_fixture(tmp_path)

    session = Session.get_instance()
    session.set_file(fixture)
    session.set_current_repetition('0')
    # Water.h5's extraction geometry only resolves correctly under this
    # rotation - see test_run_bmlab_pipeline.py's run_pipeline().
    session.orientation = Orientation(rotation=1, reflection={
        'vertically': False, 'horizontally': False
    })

    ExtractionController().find_points_all()

    cc = CalibrationController()
    for calib_key in session.get_calib_keys():
        cc.find_peaks(calib_key)
        cc.calibrate(calib_key)

    psc = PeakSelectionController()
    psc.add_brillouin_region_frequency((4.0e9, 6.0e9))
    psc.add_rayleigh_region_frequency((-2.0e9, 2.0e9))

    bc = BackgroundController()
    bc.evaluate()

    bgm = session.background_model()
    assert bgm.point_keys == ['0']

    shift = bgm.results['brillouin_shift_f']
    assert shift.size != 0
    valid = shift[~np.isnan(shift)]
    assert valid.size > 0
    np.testing.assert_allclose(valid, 5.03e9, atol=50e6)

    assert bgm.positions['x'][0] == 1.0
    assert bgm.positions['y'][0] == 2.0
    assert bgm.positions['z'][0] == 3.0

    data, positions, point_keys = bc.get_data('brillouin_shift_f')
    assert point_keys == ['0']
    # get_data() applies the parameter's own scaling (Hz -> GHz here),
    # same as EvaluationController.get_data().
    np.testing.assert_allclose(data, 5.03, atol=0.05)
    assert positions['x'][0] == 1.0


def test_background_model_serialization_round_trip(tmp_path):
    """
    The background_models dict is saved/loaded exactly like every
    other per-repetition model dict on Session (see
    bmlab.serializer.Serializer's generic dict/Serializer handling) -
    regression test that it actually round-trips, since it's easy for
    a *_models dict to be wired into set_file()/clear() but missed
    somewhere a save/load-affecting migration is added later.
    """
    fixture = _make_background_fixture(tmp_path)

    session = Session.get_instance()
    session.set_file(fixture)
    session.set_current_repetition('0')
    session.orientation = Orientation(rotation=1, reflection={
        'vertically': False, 'horizontally': False
    })

    ExtractionController().find_points_all()
    cc = CalibrationController()
    for calib_key in session.get_calib_keys():
        cc.find_peaks(calib_key)
        cc.calibrate(calib_key)
    psc = PeakSelectionController()
    psc.add_brillouin_region_frequency((4.0e9, 6.0e9))
    psc.add_rayleigh_region_frequency((-2.0e9, 2.0e9))
    BackgroundController().evaluate()

    session.save()
    session.clear()

    session = Session.get_instance()
    session.set_file(fixture)
    session.set_current_repetition('0')

    bgm = session.background_model()
    assert bgm.point_keys == ['0']
    shift = bgm.results['brillouin_shift_f']
    valid = shift[~np.isnan(shift)]
    assert valid.size > 0
    np.testing.assert_allclose(valid, 5.03e9, atol=50e6)
    assert bgm.positions['x'][0] == 1.0


def test_background_calculate_derived_values_stokes_anti_stokes():
    """
    Same FSR-corrected Stokes/Anti-Stokes-distance shift as
    test_evaluation_controller.test_calculate_derived_values_stokes_anti_
    stokes(), for the background-point shape (no peak-index axis).
    """
    session = Session.get_instance()
    session.set_file(data_file_path('Water.h5'))
    session.set_current_repetition('0')
    bgm = session.background_model()
    cm = session.calibration_model()
    cm.vipa_params['0'] = [(0, 0, 0, 20)]

    bgm.initialize_results_arrays({
        'nr_points': 3,
        'nr_images': 2,
        'nr_brillouin_regions': 2,
        'nr_rayleigh_regions': 2,
    })

    # Stokes region at 2, Anti-Stokes region at 14 -> distance 12,
    # shift = (FSR - distance) / 2 = (20 - 12) / 2 = 4.
    bgm.results['brillouin_peak_position_f'][:, :, 0] = 2
    bgm.results['brillouin_peak_position_f'][:, :, 1] = 14
    bgm.results['rayleigh_peak_position_f'][:] = 100

    BackgroundController.calculate_derived_values()

    key = 'brillouin_shift_f_stokes_anti_stokes'
    assert (bgm.results[key] == 4).all()


def test_background_controller_evaluate_without_background_points():
    """
    A repetition whose background group has no points at all (either
    because useBackgroundRoiMask was off, or the file predates the
    feature) must not error - just leave the model empty.
    """
    session = Session.get_instance()
    session.set_file(data_file_path('Water.h5'))
    session.set_current_repetition('0')

    bc = BackgroundController()
    bc.evaluate()

    bgm = session.background_model()
    assert bgm.point_keys == []
