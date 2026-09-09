import pathlib

import pytest

from bmlab.session import Session, get_valid_source, \
    get_source_file_path, get_session_file_path, BmlabInvalidFileError


def data_file_path(file_name):
    return pathlib.Path(__file__).parent / 'data' / file_name


def test_get_source_file_path():
    assert get_source_file_path(data_file_path('EvalData/Water.h5')) \
           == data_file_path('RawData/Water.h5')
    assert get_source_file_path(data_file_path('Water.session.h5'))\
           == data_file_path('Water.h5')


def test_get_session_file_path():
    assert get_session_file_path(data_file_path('RawData/Water.h5')) \
           == data_file_path('EvalData/Water.h5')
    assert get_session_file_path(data_file_path('Water.h5'))\
           == data_file_path('Water.session.h5')


def test_get_valid_source_file():
    # File from BrillouinAcquisition
    assert get_valid_source(data_file_path('Water.h5')) \
           == data_file_path('Water.h5')
    # Session file from bmlab<0.0.14
    assert get_valid_source(data_file_path('1D-x.session.h5'))\
           == data_file_path('1D-x.h5')
    # Session file from bmlab>=0.0.14
    assert get_valid_source(data_file_path('1D-y.session.h5'))\
           == data_file_path('1D-y.h5')

    # Non-existent source data file
    with pytest.raises(FileNotFoundError) as err:
        get_valid_source(data_file_path('Unavailable.h5'))
    assert err.typename == 'FileNotFoundError'
    assert 'Unavailable.h5' in str(err.value.filename)

    # Session file without source data file
    with pytest.raises(FileNotFoundError) as err:
        get_valid_source(data_file_path('lonely.session.h5'))
    assert err.typename == 'FileNotFoundError'
    assert 'lonely.h5' in str(err.value.filename)

    # Session file with invalid source data file
    with pytest.raises(BmlabInvalidFileError) as err:
        get_valid_source(data_file_path('invalid.session.h5'))
    assert err.typename == 'BmlabInvalidFileError'
    assert 'invalid.h5' in str(err.value.filename)

    # Source data file with invalid session file
    with pytest.raises(BmlabInvalidFileError) as err:
        get_valid_source(data_file_path('invalid2.session.h5'))
    assert err.typename == 'BmlabInvalidFileError'
    assert 'invalid2.session.h5' in str(err.value.filename)

    # Invalid source data file
    with pytest.raises(BmlabInvalidFileError) as err:
        get_valid_source(data_file_path('invalid.h5'))
    assert err.typename == 'BmlabInvalidFileError'
    assert 'invalid.h5' in str(err.value.filename)


def test_session_is_singleton():
    session = Session.get_instance()

    with pytest.raises(Exception):
        Session()

    assert session

    id_session = id(session)

    session = Session.get_instance()
    assert id(session) == id_session


def test_session_initializes():
    session = Session.get_instance()
    session.set_file(data_file_path('Water.h5'))

    assert len(session.extraction_models) == 1
    assert len(session.calibration_models) == 1
    assert '0' in session.calibration_models.keys()


def test_clear_session():
    # Arrange; set up session
    session = Session.get_instance()
    session.set_file(data_file_path('Water.h5'))
    session.orientation.rotation = 1
    session.set_current_repetition('0')

    # Act
    session.clear()

    # Assert
    assert session.file is None
    assert session.orientation.rotation == 0
    assert session.current_repetition() is None


def test_session_aborted_repetition():
    session = Session.get_instance()
    session.set_file(data_file_path('aborted_repetition0.h5'))
    session.set_current_repetition('0')
    session.set_current_repetition('1')


def test_session_aborted_repetition_with_no_positions():
    """
    Regression test: selecting a repetition that has resolution
    attributes but no positions-x/y/z datasets at all (a more
    severely aborted/restarted acquisition) must not raise - it
    should behave the same as any other "no valid data" repetition.
    """
    session = Session.get_instance()
    session.set_file(
        data_file_path('aborted_repetition0_no_positions.h5'))

    session.set_current_repetition('0')
    assert session.get_payload_resolution() is None
    assert session.get_payload_positions() is None

    session.set_current_repetition('1')
    assert session.get_payload_resolution() is not None
    assert session.get_payload_positions() is not None


def test_session_get_calib_keys():
    session = Session.get_instance()
    session.set_file(data_file_path('2D-xy.h5'))
    session.set_current_repetition('0')

    keys = session.get_calib_keys()
    assert len(keys) == 1
    keys = session.get_calib_keys(True)
    assert keys == ['1']


def test_session_get_image_keys():
    session = Session.get_instance()
    session.set_file(data_file_path('2D-xy.h5'))
    session.set_current_repetition('0')

    image_keys = session.get_image_keys()
    assert len(image_keys) == 15
    image_keys = session.get_image_keys(True)
    assert image_keys == \
           ['0', '3', '6', '9', '12',
            '1', '4', '7', '10', '13',
            '2', '5', '8', '11', '14']


def test_session_has_no_surface_scan_by_default():
    session = Session.get_instance()
    session.set_file(data_file_path('Water.h5'))
    session.set_current_repetition('0')

    assert not session.has_surface_scan()
    assert session.get_surface_scan_data() is None
    assert not session.has_overview_brightfield()
    assert session.get_overview_brightfield_positions() is None
    assert session.get_overview_brightfield_keys() == []
    assert session.get_overview_brightfield_image('0') is None


def test_session_get_surface_scan_data():
    session = Session.get_instance()
    session.set_file(data_file_path('SurfaceScan.h5'))
    session.set_current_repetition('0')

    assert session.has_surface_scan()
    data = session.get_surface_scan_data()
    assert data['surface_found_mask'].shape == (3, 2)


def test_session_get_overview_brightfield():
    session = Session.get_instance()
    session.set_file(data_file_path('SurfaceScan.h5'))
    session.set_current_repetition('0')

    assert session.has_overview_brightfield()

    keys = session.get_overview_brightfield_keys()
    assert keys == ['0', '1', '2', '3']

    image = session.get_overview_brightfield_image(keys[0])
    assert image.shape == (1, 20, 20)

    positions = session.get_overview_brightfield_positions()
    assert positions['tile_count'] == 2
