import pytest
import pathlib
import datetime

import h5py
import numpy as np

from bmlab.file import BrillouinFile, Payload, \
    BadFileException, is_source_file, is_session_file


def data_file_path(file_name):
    return pathlib.Path(__file__).parent / 'data' / file_name


def test_is_source_file():
    # File from BrillouinAcquisition
    assert is_source_file(data_file_path('Water.h5'))
    # Session file from bmlab<0.0.14
    assert not is_source_file(data_file_path('1D-x.session.h5'))
    # Session file from bmlab>=0.0.14
    assert not is_source_file(data_file_path('1D-y.session.h5'))
    # Non-existing file
    assert not is_source_file(data_file_path('Unavailable.h5'))


def test_is_session_file():
    # Session file from bmlab<0.0.14
    assert is_session_file(data_file_path('1D-x.session.h5'))
    # Session file from bmlab>=0.0.14
    assert is_session_file(data_file_path('1D-y.session.h5'))
    # File from BrillouinAcquisition
    assert not is_session_file(data_file_path('Water.h5'))
    # Non-existing file
    assert not is_session_file(data_file_path('Unavailable.h5'))


def test_file_has_one_repetition():
    bf = BrillouinFile(data_file_path('Water.h5'))
    num_repetitions = bf.repetition_count()
    assert num_repetitions == 1

    bf = BrillouinFile(data_file_path('Water_old.h5'))
    num_repetitions = bf.repetition_count()
    assert num_repetitions == 1


def test_non_existing_file_raises_exception():
    with pytest.raises(OSError):
        BrillouinFile(data_file_path('non_existing_file.h5'))


def test_file_has_comment():
    bf = BrillouinFile(data_file_path('Water.h5'))
    assert bf.comment == 'Brillouin data'


def test_file_has_date():
    bf = BrillouinFile(data_file_path('Water.h5'))
    assert bf.date == datetime.datetime.fromisoformat(
        '2020-11-03T15:20:30.682+01:00')


def test_open_file_with_no_brillouin_data_raises_exception():
    with pytest.raises(BadFileException):
        BrillouinFile(data_file_path('empty_file.h5'))


def test_file_get_repetition_keys():
    bf = BrillouinFile(data_file_path('Water.h5'))
    assert bf.repetition_keys() == ['0']

    assert bf.repetition_keys('Fluorescence') == []

    bf = BrillouinFile(data_file_path('Water_old.h5'))
    assert bf.repetition_keys() == ['0']

    assert bf.repetition_keys('Fluorescence') == []


def test_file_get_repetition_date():
    bf = BrillouinFile(data_file_path('Water.h5'))
    rep = bf.get_repetition('0')
    assert rep.date == datetime.datetime.fromisoformat(
        '2020-11-03T15:20:52.852+01:00')


def test_file_get_resolution():
    bf = BrillouinFile(data_file_path('Water.h5'))
    rep = bf.get_repetition('0')
    assert rep.payload.resolution == (10, 1, 1)


def test_file_get_positions():
    bf = BrillouinFile(data_file_path('Water.h5'))
    rep = bf.get_repetition('0')
    # Test that the shape is correct
    assert rep.payload.positions['x'].shape == (1, 10, 1)
    assert rep.payload.positions['y'].shape == (1, 10, 1)
    assert rep.payload.positions['z'].shape == (1, 10, 1)
    # Test some values
    assert rep.payload.positions['x'][0, 0, 0] == -5652.5
    assert rep.payload.positions['y'][0, 0, 0] == -1963.0
    assert rep.payload.positions['z'][0, 0, 0] == 201.425


def test_file_repetition_has_calibration():
    bf = BrillouinFile(data_file_path('Water.h5'))
    rep = bf.get_repetition('0')
    assert not rep.calibration.is_empty()

    bf = BrillouinFile(data_file_path('Water_old.h5'))
    rep = bf.get_repetition('0')
    assert not rep.calibration.is_empty()


def test_file_payload_image_keys():
    bf = BrillouinFile(data_file_path('Water.h5'))
    rep = bf.get_repetition('0')
    assert rep.payload.image_keys() == [str(k) for k in range(10)]

    bf = BrillouinFile(data_file_path('Water_old.h5'))
    rep = bf.get_repetition('0')
    assert rep.payload.image_keys() == [str(k) for k in range(4)]


def test_file_payload_get_image():
    bf = BrillouinFile(data_file_path('Water.h5'))
    image = bf.get_repetition('0').payload.get_image('0')
    assert image.shape == (2, 400, 400)

    bf = BrillouinFile(data_file_path('Water_old.h5'))
    image = bf.get_repetition('0').payload.get_image('0')
    assert image.shape == (2, 400, 400)


def test_file_payload_get_date():
    bf = BrillouinFile(data_file_path('Water.h5'))
    date = bf.get_repetition('0').payload.get_date('0')
    assert date == datetime.datetime.fromisoformat(
        '2020-11-03T15:21:10.568+01:00')


def test_file_payload_get_time():
    bf = BrillouinFile(data_file_path('Water.h5'))
    time = bf.get_repetition('0').payload.get_time('0')
    assert time == 39.886


def test_file_calibration_image_keys():
    bf = BrillouinFile(data_file_path('Water.h5'))
    rep = bf.get_repetition('0')
    assert rep.calibration.image_keys() == [str(k+1) for k in range(2)]

    bf = BrillouinFile(data_file_path('Water_old.h5'))
    rep = bf.get_repetition('0')
    assert rep.calibration.image_keys() == [str(k+1) for k in range(2)]


def test_file_calibration_get_date():
    bf = BrillouinFile(data_file_path('Water.h5'))
    date = bf.get_repetition('0').calibration.get_date('1')
    assert date == datetime.datetime.fromisoformat(
        '2020-11-03T15:21:07.484+01:00')


def test_file_calibration_get_time():
    bf = BrillouinFile(data_file_path('Water.h5'))
    time = bf.get_repetition('0').calibration.get_time('1')
    assert time == 36.802


def test_file_calibration_get_exposure():
    bf = BrillouinFile(data_file_path('Binning.h5'))
    assert bf.get_repetition('0').calibration.get_exposure('1') == 0.8
    assert bf.get_repetition('0').payload.get_exposure('0') == 0.2


def test_file_calibration_get_binning():
    bf = BrillouinFile(data_file_path('Binning.h5'))
    assert bf.get_repetition('0').calibration.get_binning('1') == '8x8'
    assert bf.get_repetition('0').payload.get_binning('0') == '8x8'

    assert bf.get_repetition('0')\
        .payload.get_binning_factor('0') == 8


def test_file_repetition_count():
    bf = BrillouinFile(data_file_path('Fluorescence.h5'))
    mode = 'Fluorescence'

    rep_keys = bf.repetition_keys(mode)
    assert rep_keys == ['0', '1']

    with pytest.raises(NotImplementedError):
        bf.repetition_count('Raman')


def test_file_get_fluorescence_images():
    bf = BrillouinFile(data_file_path('Fluorescence.h5'))
    mode = 'Fluorescence'

    rep_keys = bf.repetition_keys(mode)
    assert rep_keys == ['0', '1']

    repetition = bf.get_repetition(rep_keys[0], mode)

    image_keys = repetition.payload.image_keys()
    assert image_keys == ['0', '1', '2', '3']

    time = repetition.payload.get_time(image_keys[0])
    assert time == 22.376

    channel = repetition.payload.get_channel(image_keys[0])
    assert channel == 'Blue'

    roi = repetition.payload.get_ROI(image_keys[0])
    assert roi == dict({
        'bottom': 152,
        'height_binned': 700,
        'height_physical': 700,
        'left': 200,
        'right': 202,
        'top': 150,
        'width_binned': 600,
        'width_physical': 600
    })

    image = repetition.payload.get_image(image_keys[0])
    assert image.shape == (1, 700, 600)


def test_file_payload_with_resolution_but_no_positions():
    """
    Regression test: a repetition from a more severely aborted/
    restarted acquisition can have the resolution-x/y/z attributes
    set (the grid was configured) but no positions-x/y/z datasets at
    all (nothing was ever measured, so h5bm never wrote them). This
    must be treated the same as "no valid measurement grid" - not
    silently produce a bogus positions dict (np.array(None) does not
    raise, unlike a missing resolution attribute would).
    """
    bf = BrillouinFile(
        data_file_path('aborted_repetition0_no_positions.h5'))
    rep = bf.get_repetition('0')
    assert rep.payload.resolution is None
    assert rep.payload.positions is None

    # The working repetition is unaffected
    rep = bf.get_repetition('1')
    assert rep.payload.resolution is not None
    assert rep.payload.positions is not None


def test_file_has_no_surface_scan_data_by_default():
    bf = BrillouinFile(data_file_path('Water.h5'))
    rep = bf.get_repetition('0')
    assert not rep.payload.has_surface_scan()
    assert rep.payload.get_surface_scan_data() is None
    assert not rep.payload.has_overview_brightfield()
    assert rep.payload.get_overview_brightfield_positions() is None


def test_file_reconstructs_surface_follow_z_from_sampled_path(tmp_path):
    # Older BrillouinAcquisition versions wrote positions-z as a flat
    # nominal grid (same z for every x, y at a given z-index), discarding
    # the real per-(x, y) surface-following offset - even though that real
    # offset is separately available, in acquisition order, in
    # positions-sampled-x/y/z. Payload should reconstruct the true z from
    # that path rather than exposing the flat nominal grid.
    nx, ny, nz = 2, 2, 3
    x_grid = np.array([0.0, 10.0])
    y_grid = np.array([0.0, 10.0])
    # Deliberately different from any sampled z, so a leftover nominal
    # value is easy to tell apart from a reconstructed one.
    nominal_z = np.array([100.0, 200.0, 300.0])

    # Per-(x, y) column real z, out of ascending z-index order on purpose
    # (acquisition order isn't sorted-by-z-index) - reconstruction sorts
    # within each column, so this must not matter.
    columns = {
        (0.0, 0.0): [1.5, 0.5, 2.5],
        (0.0, 10.0): [2.0, 3.0, 1.0],
        (10.0, 0.0): [0.5, -0.5, 1.5],
        (10.0, 10.0): [1.0, 2.0, 0.0],
    }
    sampled_x, sampled_y, sampled_z = [], [], []
    for (cx, cy), zs in columns.items():
        for z in zs:
            sampled_x.append(cx)
            sampled_y.append(cy)
            sampled_z.append(z)

    def populate(group):
        group.attrs['resolution-x'] = [nx]
        group.attrs['resolution-y'] = [ny]
        group.attrs['resolution-z'] = [nz]

        pos_x = np.tile(x_grid.reshape(1, nx, 1), (nz, 1, ny))
        pos_y = np.tile(y_grid.reshape(1, 1, ny), (nz, nx, 1))
        pos_z = np.tile(nominal_z.reshape(nz, 1, 1), (1, nx, ny))
        group.create_dataset('positions-x', data=pos_x)
        group.create_dataset('positions-y', data=pos_y)
        group.create_dataset('positions-z', data=pos_z)

        group.create_dataset('positions-sampled-x', data=sampled_x)
        group.create_dataset('positions-sampled-y', data=sampled_y)
        group.create_dataset('positions-sampled-z', data=sampled_z)

    payload = _make_payload(
        tmp_path, 'surface_follow_z.h5', populate)

    for ix, cx in enumerate(x_grid):
        for iy, cy in enumerate(y_grid):
            expected = np.sort(columns[(cx, cy)])
            actual = payload.positions['z'][:, ix, iy]
            np.testing.assert_array_equal(actual, expected)
    # None of the reconstructed values should be the old flat nominal.
    assert not np.any(np.isin(payload.positions['z'], nominal_z))


def test_file_falls_back_to_nominal_z_without_sampled_path(tmp_path):
    # Files old enough not to have positions-sampled-* at all (or a
    # column with no matching sampled point) should keep the nominal
    # positions-z untouched rather than erroring.
    nx, ny, nz = 2, 2, 3
    x_grid = np.array([0.0, 10.0])
    y_grid = np.array([0.0, 10.0])
    nominal_z = np.array([100.0, 200.0, 300.0])

    def populate(group):
        group.attrs['resolution-x'] = [nx]
        group.attrs['resolution-y'] = [ny]
        group.attrs['resolution-z'] = [nz]
        pos_x = np.tile(x_grid.reshape(1, nx, 1), (nz, 1, ny))
        pos_y = np.tile(y_grid.reshape(1, 1, ny), (nz, nx, 1))
        pos_z = np.tile(nominal_z.reshape(nz, 1, 1), (1, nx, ny))
        group.create_dataset('positions-x', data=pos_x)
        group.create_dataset('positions-y', data=pos_y)
        group.create_dataset('positions-z', data=pos_z)

    payload = _make_payload(tmp_path, 'no_sampled_path.h5', populate)

    for ix in range(nx):
        for iy in range(ny):
            np.testing.assert_array_equal(
                payload.positions['z'][:, ix, iy], nominal_z)


def test_file_get_surface_scan_data():
    bf = BrillouinFile(data_file_path('SurfaceScan.h5'))
    rep = bf.get_repetition('0')
    assert rep.payload.has_surface_scan()

    data = rep.payload.get_surface_scan_data()
    assert data['surface_found_mask'].shape == (3, 2)
    assert data['surface_found_mask'][0, 0] == 1
    assert data['surface_found_mask'][2, 1] == 2

    assert data['roi_scan_plan_mask'].shape == (3, 2)
    assert data['sampled_mask'].shape == (1, 3, 2)

    assert data['sampled_x'].shape == (4,)
    assert data['sampled_y'].shape == (4,)
    assert data['sampled_z'].shape == (4,)

    assert data['surface_follow_used'] == 1
    assert data['surface_z_offset_um_used'] == 2.5
    assert data['surface_verification_steps_used'] == 3


def test_file_get_overview_brightfield_positions():
    bf = BrillouinFile(data_file_path('SurfaceScan.h5'))
    rep = bf.get_repetition('0')
    assert rep.payload.has_overview_brightfield()

    positions = rep.payload.get_overview_brightfield_positions()
    assert positions['tile_count'] == 2
    assert positions['x'].shape == (2, 2)
    assert positions['y'].shape == (2, 2)
    assert positions['z'].shape == (2, 2)


def _make_payload(tmp_path, name, populate):
    """
    Creates a standalone HDF5 payload group (not a full BrillouinFile) in
    a temp file and wraps it in a Payload, for exercising Payload logic
    against synthetic datasets/attributes without needing a full
    BrillouinAcquisition-generated fixture file.
    """
    h5 = h5py.File(tmp_path / name, 'w')
    group = h5.create_group('payload')
    populate(group)
    return Payload(group, None)


def test_file_get_overview_brightfield_positions_point_count_uniform(
        tmp_path):
    # New-format file: point-count/point-stack-counts with a uniform
    # stack depth of 1 per point (2 points per z-slice, 2 z-slices).
    def populate(group):
        group.create_dataset(
            'positions-overview-brightfield-x', data=np.arange(4.))
        group.create_dataset(
            'positions-overview-brightfield-y', data=np.arange(4.))
        group.create_dataset(
            'positions-overview-brightfield-z', data=np.arange(4.))
        group.create_dataset(
            'positions-overview-brightfield-point-count', data=[2])
        group.create_dataset(
            'positions-overview-brightfield-point-stack-counts',
            data=[1, 1])

    payload = _make_payload(tmp_path, 'uniform.h5', populate)
    assert payload.has_overview_brightfield()

    positions = payload.get_overview_brightfield_positions()
    assert positions['point_count'] == 2
    assert list(positions['point_stack_counts']) == [1, 1]
    assert positions['tile_count'] == 2
    assert positions['x'].shape == (2, 2)


def test_file_get_overview_brightfield_positions_non_uniform_stack(
        tmp_path):
    # New-format file with a non-uniform per-point stack depth: one
    # point kept its full z-stack (depth 3), the other is a single
    # image (depth 1) -> total_per_z = 4, over 2 z-slices = 8 values.
    def populate(group):
        group.create_dataset(
            'positions-overview-brightfield-x', data=np.arange(8.))
        group.create_dataset(
            'positions-overview-brightfield-y', data=np.arange(8.))
        group.create_dataset(
            'positions-overview-brightfield-z', data=np.arange(8.))
        group.create_dataset(
            'positions-overview-brightfield-point-count', data=[2])
        group.create_dataset(
            'positions-overview-brightfield-point-stack-counts',
            data=[1, 3])

    payload = _make_payload(tmp_path, 'nonuniform.h5', populate)

    positions = payload.get_overview_brightfield_positions()
    assert positions['point_count'] == 2
    assert list(positions['point_stack_counts']) == [1, 3]
    # total_per_z = 1 + 3 = 4, reshaped over 2 z-slices
    assert positions['tile_count'] == 4
    assert positions['x'].shape == (2, 4)


def test_file_get_overview_brightfield_positions_old_format_fallback(
        tmp_path):
    # Files from before the point-count/point-stack-counts update only
    # wrote a single tile-count (implying a stack depth of 1 per tile).
    def populate(group):
        group.create_dataset(
            'positions-overview-brightfield-x', data=np.arange(4.))
        group.create_dataset(
            'positions-overview-brightfield-y', data=np.arange(4.))
        group.create_dataset(
            'positions-overview-brightfield-z', data=np.arange(4.))
        group.create_dataset(
            'positions-overview-brightfield-tile-count', data=[2])

    payload = _make_payload(tmp_path, 'oldformat.h5', populate)

    positions = payload.get_overview_brightfield_positions()
    assert positions['point_count'] == 2
    assert list(positions['point_stack_counts']) == [1, 1]
    assert positions['tile_count'] == 2
    assert positions['x'].shape == (2, 2)


def test_file_get_position_and_stage_position(tmp_path):
    def populate(group):
        data = group.create_group('data')
        with_both = data.create_dataset('0', data=np.zeros((1, 2, 2)))
        with_both.attrs['position_x_um'] = [1.0]
        with_both.attrs['position_y_um'] = [2.0]
        with_both.attrs['position_z_um'] = [3.0]
        with_both.attrs['stage_position_x_um'] = [1.1]
        with_both.attrs['stage_position_y_um'] = [2.1]
        with_both.attrs['stage_position_z_um'] = [3.1]

        target_only = data.create_dataset('1', data=np.zeros((1, 2, 2)))
        target_only.attrs['position_x_um'] = [4.0]
        target_only.attrs['position_y_um'] = [5.0]
        target_only.attrs['position_z_um'] = [6.0]

        no_position = data.create_dataset('2', data=np.zeros((1, 2, 2)))
        del no_position  # no position attrs at all

    payload = _make_payload(tmp_path, 'positions.h5', populate)

    assert payload.get_position('0') == {'x': 1.0, 'y': 2.0, 'z': 3.0}
    assert payload.get_stage_position('0') == \
        {'x': 1.1, 'y': 2.1, 'z': 3.1}

    assert payload.get_position('1') == {'x': 4.0, 'y': 5.0, 'z': 6.0}
    assert payload.get_stage_position('1') is None

    assert payload.get_position('2') is None
    assert payload.get_stage_position('2') is None


def test_file_get_overview_brightfield_images():
    bf = BrillouinFile(data_file_path('SurfaceScan.h5'))
    mode = 'Fluorescence'
    rep = bf.get_repetition('0', mode)

    keys = rep.payload.image_keys_by_channel('Brightfield z overview')
    assert keys == ['0', '1', '2', '3']

    # Other channel values yield no matches
    assert rep.payload.image_keys_by_channel('Red') == []

    image = rep.payload.get_image(keys[0])
    assert image.shape == (1, 20, 20)


def test_file_get_scale_calibration():
    bf = BrillouinFile(data_file_path('Fluorescence.h5'))
    mode = 'Fluorescence'

    repetition = bf.get_repetition('0', mode)
    scale_calibration = repetition.payload.get_scale_calibration()

    assert scale_calibration == dict({
        'micrometerToPixX': (-7.95, 9.05),
        'micrometerToPixY': (9.45, 8.65),
        'pixToMicrometerX': (-0.05606325750210643, 0.05865577808023852),
        'pixToMicrometerY': (0.0612482986583706, 0.05152634649037527),
        'positionScanner': (0.0, 0.0),
        'positionStage': (0.0, 0.0),
        'origin': (0.0, 0.0)
    })
