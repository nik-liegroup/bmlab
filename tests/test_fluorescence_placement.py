"""
Placing brightfield images by the middle of the image: the attributes
tools/add_image_centres.py writes into the images taken before
BrillouinAcquisition wrote them itself, and how bmlab reads them
(Payload.get_image_centre_shift(), get_image_centre_raw(),
raw_to_grid_frame(), MeasurementData.get_overview_info()).
"""
from pathlib import Path
import importlib.util
import os
import shutil
import uuid

import h5py
import numpy as np
import pytest

from bmlab.file import BrillouinFile

spec = importlib.util.spec_from_file_location(
    'add_image_centres',
    Path(__file__).parent.parent / 'tools' / 'add_image_centres.py')
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


@pytest.fixture()
def tmp_dir():
    current_path = Path.cwd()
    tmp_dir = Path.cwd() / f"tmp{str(uuid.uuid4())}"
    os.makedirs(tmp_dir, exist_ok=True)
    os.chdir(tmp_dir)
    yield tmp_dir
    os.chdir(current_path)
    try:
        shutil.rmtree(tmp_dir)
    except Exception as e:
        print(e)


def data_file_path(file_name):
    return Path(__file__).parent / 'data' / file_name


def _set_point(group, name, x, y):
    if name not in group:
        group.create_group(name)
    group[name].attrs['x'] = np.array([float(x)])
    group[name].attrs['y'] = np.array([float(y)])


def _put(group, name, value):
    if name in group:
        del group[name]
    group.create_dataset(name, data=np.atleast_1d(np.asarray(value, float)))


def make_file(absolute=True, channel=b'Brillouin'):
    """
    SurfaceScan.h5 (its 4 Fluorescence images are 20 x 20 px overviews
    from before the overview engine, planned position (0, 0)) with the
    scale calibration of their repetition saying: stage at (1000, 2000)
    um, laser marker (scanner) at (240, -330) um, 0.5 um per pixel with
    the image's x along the stage's -y (as on the stand), FOV offset
    (3, -4) um. `channel`: what the images become (b'Brightfield': taken
    in the Fluorescence tab; anything else leaves them overviews).
    """
    path = Path.cwd() / 'S.h5'
    shutil.copy(data_file_path('SurfaceScan.h5'), path)
    with h5py.File(path, 'r+') as f:
        cal = f['Fluorescence/0/payload/scaleCalibration']
        _set_point(cal, 'positionStage', 1000.0, 2000.0)
        _set_point(cal, 'positionScanner', 240.0, -330.0)
        _set_point(cal, 'pixToMicrometerX', 0.0, -0.5)
        _set_point(cal, 'pixToMicrometerY', 0.5, 0.0)
        _set_point(cal, 'origin', 0.0, 0.0)
        _set_point(cal, 'fovOffset', 3.0, -4.0)
        cal.attrs['hasFovOffset'] = np.array([1])
        if channel == b'Brightfield':
            for key in f['Fluorescence/0/payload/data']:
                f['Fluorescence/0/payload/data'][key].attrs['channel'] = \
                    np.array([channel])
        payload = f['Brillouin/0/payload']
        _put(payload, 'positions-grid-coordinates-absolute',
             1 if absolute else 0)
        for axis, value in zip('xyz', (900.0, 1800.0, 50.0)):
            _put(payload, 'positions-absolute-origin-' + axis, value)
    return path


# the middle of a 20 x 20 px image lies (5, -5) um from the calibration
# origin: shift = scanner - (5, -5) + FOV offset
SHIFT = (240.0 - 5.0 + 3.0, -330.0 + 5.0 - 4.0)


def test_images_taken_by_hand_get_the_raw_position_and_the_shift(tmp_dir):
    path = make_file(channel=b'Brightfield')
    tool.main(['--apply', str(path)])
    file = BrillouinFile(path)
    payload = file.get_repetition('0', 'Fluorescence').payload
    assert payload.get_image_centre_shift('0') == pytest.approx(SHIFT)
    # raw = stage + scanner; the middle is raw - shift = stage + 5, -5 - FOV
    centre = payload.get_image_centre_raw('0')
    assert centre == pytest.approx((1002.0, 1999.0))
    # in the frame of the run: minus the absolute origin of the grid
    grid = file.get_repetition('0').payload
    assert grid.raw_to_grid_frame(centre) == pytest.approx((102.0, 199.0))
    file.close()


def test_relative_grid_coordinates_are_the_raw_positions(tmp_dir):
    path = make_file(absolute=False, channel=b'Brightfield')
    tool.main(['--apply', str(path)])
    file = BrillouinFile(path)
    centre = file.get_repetition('0', 'Fluorescence').payload \
        .get_image_centre_raw('0')
    assert file.get_repetition('0').payload.raw_to_grid_frame(centre) == \
        pytest.approx(centre)
    file.close()


def test_a_cropped_roi_enters_with_its_offset_on_the_sensor(tmp_dir):
    path = make_file(channel=b'Brightfield')
    with h5py.File(path, 'r+') as f:
        attrs = f['Fluorescence/0/payload/data/0'].attrs
        # 20 x 20 px cut out 4 binned px from the left edge, binned 2x2
        for name, value in (('ROI_left', 8), ('ROI_top', 0),
                            ('ROI_width_physical', 40),
                            ('ROI_width_binned', 20),
                            ('ROI_height_physical', 40),
                            ('ROI_height_binned', 20)):
            attrs[name] = np.array([value])
    tool.main(['--apply', str(path)])
    file = BrillouinFile(path)
    payload = file.get_repetition('0', 'Fluorescence').payload
    # 4 px further along the image's x: 2 um further along the stage's -y
    assert payload.get_image_centre_raw('0') == \
        pytest.approx((1002.0, 1999.0 - 2.0))
    assert payload.get_image_centre_raw('1') == pytest.approx((1002.0, 1999.0))
    file.close()


def test_a_legacy_overview_is_placed_by_its_shift(tmp_dir):
    path = make_file()
    tool.main(['--apply', str(path)])
    file = BrillouinFile(path)
    payload = file.get_repetition('0', 'Fluorescence').payload
    info = payload.get_overview_info('0')
    # taken with the laser on the planned point (0, 0): the middle of the
    # image is that point minus the shift; no raw position is added
    assert info['image_centre_shift'] == pytest.approx(SHIFT)
    assert (info['anchor']['x'], info['anchor']['y']) == \
        pytest.approx((-SHIFT[0], -SHIFT[1]))
    assert payload.get_image_centre_raw('0') is None
    file.close()


def test_images_with_the_attributes_are_kept(tmp_dir):
    path = make_file()
    with h5py.File(path, 'r+') as f:
        attrs = f['Fluorescence/0/payload/data/0'].attrs
        attrs['image_centre_shift_x_um'] = np.array([7.0])
        attrs['image_centre_shift_y_um'] = np.array([-8.0])
    tool.main(['--apply', str(path)])
    with h5py.File(path, 'r') as f:
        attrs = f['Fluorescence/0/payload/data/0'].attrs
        assert float(attrs['image_centre_shift_x_um'][0]) == 7.0
        assert 'fov_compensation_x_um' not in attrs
        assert 'image_centre_shift_x_um' in \
            f['Fluorescence/0/payload/data/1'].attrs
        assert tool.MARK in f.attrs
    # a second run does nothing
    assert tool.apply_file(str(path), True)[0] == 'already done'


def test_nothing_to_place_with(tmp_dir):
    path = make_file(channel=b'Brightfield')
    with h5py.File(path, 'r+') as f:
        del f['Brillouin/0/payload/positions-grid-coordinates-absolute']
    file = BrillouinFile(path)
    payload = file.get_repetition('0', 'Fluorescence').payload
    assert payload.get_image_centre_shift('0') is None
    assert payload.get_image_centre_raw('0') is None
    assert file.get_repetition('0').payload.raw_to_grid_frame((1.0, 2.0)) \
        is None
    file.close()
