import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from bmlab.export.alignment import calibration_stage_to_pixel
from bmlab.export.overview_brightfield_export import OverviewBrightfieldExport


def make_scene(shape=(1500, 1800), seed=0):
    rng = np.random.default_rng(seed)
    return gaussian_filter(rng.normal(size=shape), 6) * 100 + 128


def crop(scene, row, col, shape=(300, 400)):
    return scene[row:row + shape[0], col:col + shape[1]].copy()


def test_calibration_stage_to_pixel_inverts_the_image_y_axis():
    # The calibration of the 20x objective of this setup: micrometerToPixX
    # = (0, 4.3), micrometerToPixY = (-4.3, 0) in the live view
    # (pixel = x * X + y * Y, ordered (col, row)). The stored image has its
    # rows the other way round, which was measured from the overlap of the
    # overview tiles: stage x -> -4.3 rows, stage y -> -4.3 columns.
    calibration = {'micrometerToPixX': [0.0, 4.3],
                   'micrometerToPixY': [-4.3, 0.0]}
    matrix = calibration_stage_to_pixel(calibration)
    assert np.allclose(matrix, [[0.0, -4.3], [-4.3, 0.0]])
    assert np.allclose(matrix @ [190.6, 0.0], [0.0, -819.58])   # (col, row)
    assert np.allclose(matrix @ [0.0, 238.2], [-1024.26, 0.0])


def test_tiles_placed_from_their_stage_positions_line_up():
    """The 3 x 2 tile grid of an overview, cut from one scene with the
    camera/stage geometry above, is put back together exactly."""
    scene = make_scene((2600, 2600))
    calibration = {'micrometerToPixX': [0.0, 4.0],
                   'micrometerToPixY': [-4.0, 0.0]}
    stage_map = calibration_stage_to_pixel(calibration)
    shape = (300, 400)
    positions = [(x, y) for y in (0.0, 60.0, 120.0) for x in (0.0, 70.0)]
    origin = np.array([1200, 1200])           # (col, row) of tile 0
    tiles = []
    for x, y in positions:
        col, row = origin + (stage_map @ np.array([x, y])).astype(int)
        tiles.append((crop(scene, row, col, shape), x, y))
    reference = positions[0]
    offsets = [stage_map @ (np.array([x, y]) - np.array(reference))
               for _, x, y in tiles]
    min_col = min(o[0] for o in offsets)
    min_row = min(o[1] for o in offsets)
    placed = [(tile, int(round(o[1] - min_row)), int(round(o[0] - min_col)))
              for (tile, _, _), o in zip(tiles, offsets)]
    mosaic = OverviewBrightfieldExport._compose_raw_mosaic(placed)
    top = int(origin[1] + min_row)
    left = int(origin[0] + min_col)
    expected = scene[top:top + mosaic.shape[0], left:left + mosaic.shape[1]]
    covered = ~np.isnan(mosaic)
    assert covered.mean() > 0.6
    assert np.array_equal(mosaic[covered], expected[covered])


def test_compose_raw_mosaic_first_tile_wins():
    a = np.full((10, 10), 1.0)
    b = np.full((10, 10), 2.0)
    canvas = OverviewBrightfieldExport._compose_raw_mosaic(
        [(a, 0, 0), (b, 5, 5)])
    assert canvas.shape == (15, 15)
    assert canvas[7, 7] == 1.0     # overlap: first tile
    assert canvas[12, 12] == 2.0
    assert np.isnan(canvas[0, 12])


# ---- orientation, TIFF writing and image naming ---------------------------
from datetime import datetime, timedelta, timezone  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import tifffile  # noqa: E402

from bmlab.export.alignment import orient_image, warp_local  # noqa: E402
from bmlab.export.naming import (  # noqa: E402
    folder_name, following_brillouin_repetition, timestamp_tag)
from bmlab.export.tiffio import write_tiff  # noqa: E402


@pytest.mark.parametrize('tmatrix', [
    np.matrix([[0.0, 1.0, 0], [-1.0, 0.0, 0], [0, 0, 1]]),
    np.matrix([[0.0, -1.0, 0], [1.0, 0.0, 0], [0, 0, 1]]),
    np.matrix([[-1.0, 0.0, 0], [0.0, 1.0, 0], [0, 0, 1]]),
    np.matrix([[1.0, 0.0, 0], [0.0, -1.0, 0], [0, 0, 1]]),
])
def test_orient_image_matches_warp_local_without_resampling(tmatrix):
    rng = np.random.default_rng(0)
    img = rng.integers(0, 65535, size=(40, 55)).astype(np.uint16)
    oriented, translate = orient_image(img, tmatrix)
    assert oriented.dtype == np.uint16            # pixel type kept
    warped, shape, translate_warp = warp_local(img.astype(float), tmatrix)
    assert np.allclose(translate, translate_warp)
    # same placement; orient_image additionally keeps the last row/column
    assert oriented.shape[0] >= shape[0] and oriented.shape[1] >= shape[1]
    inner = oriented[:shape[0], :shape[1]].astype(float)
    assert np.allclose(inner[1:-1, 1:-1], warped[1:-1, 1:-1], atol=1e-6)
    # nothing is lost: the same set of pixels, just moved
    assert np.array_equal(np.sort(oriented.ravel()), np.sort(img.ravel()))


def test_orient_image_refuses_a_matrix_that_is_not_a_quarter_turn():
    angle = np.deg2rad(30)
    tmatrix = np.matrix([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    with pytest.raises(ValueError):
        orient_image(np.ones((20, 20)), tmatrix)


def test_write_tiff_keeps_pixel_type_and_pages(tmp_path):
    single = (np.arange(12, dtype=np.uint16) * 5000).reshape(3, 4)
    write_tiff(tmp_path / 'one.tif', single)
    back = tifffile.imread(tmp_path / 'one.tif')
    assert back.dtype == np.uint16 and np.array_equal(back, single)

    pages = [single, single + 1, single + 2]
    write_tiff(tmp_path / 'stack.tif', pages)
    back = tifffile.imread(tmp_path / 'stack.tif')
    assert back.shape == (3, 3, 4) and back.dtype == np.uint16
    assert np.array_equal(back[2], single + 2)

    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    rgb[..., 1] = 200
    write_tiff(tmp_path / 'rgb.tif', rgb, rgb=True)
    back = tifffile.imread(tmp_path / 'rgb.tif')
    assert back.shape == (3, 4, 3) and back[0, 0, 1] == 200


def _stub_file(starts):
    return SimpleNamespace(
        repetition_keys=lambda: list(starts),
        get_repetition=lambda key: SimpleNamespace(date=starts[key]))


def test_image_belongs_to_the_brillouin_repetition_acquired_next():
    t0 = datetime(2026, 10, 14, 1, 0, tzinfo=timezone.utc)
    # '0' was an aborted start and no longer exists: the images taken
    # before it move on to the next one that does.
    starts = {'1': t0 + timedelta(minutes=30), '2': t0 + timedelta(hours=2)}
    file = _stub_file(starts)
    assert following_brillouin_repetition(file, t0) == '1'
    assert following_brillouin_repetition(
        file, t0 + timedelta(minutes=29)) == '1'
    # between two acquisitions: the second one
    assert following_brillouin_repetition(
        file, t0 + timedelta(hours=1)) == '2'
    # after the last acquisition, or without a capture time: none
    assert following_brillouin_repetition(
        file, t0 + timedelta(hours=3)) is None
    assert following_brillouin_repetition(file, None) is None
    # numeric order of the keys must not matter, only the time
    starts = {'10': t0 + timedelta(hours=1), '2': t0 + timedelta(hours=2)}
    assert following_brillouin_repetition(_stub_file(starts), t0) == '10'


def test_folder_and_time_stamp_names():
    assert folder_name('FL', '3') == 'FL_BMrep3'
    assert folder_name('Overview', 0) == 'Overview_BMrep0'
    assert folder_name('FL', None) == 'FL_afterBM'
    assert timestamp_tag(datetime(2026, 9, 21, 18, 32, 5)) == '20260921-183205'
    assert timestamp_tag(None) == ''


def test_images_taken_during_a_run_belong_to_the_running_repetition():
    from bmlab.export.naming import running_brillouin_repetition
    t0 = datetime(2026, 10, 14, 1, 0, tzinfo=timezone.utc)
    starts = {'0': t0, '1': t0 + timedelta(hours=1),
              '3': t0 + timedelta(hours=2)}
    file = _stub_file(starts)
    # older software stored 0 for all of them - the time decides
    for minutes, expected in ((5, '0'), (59, '0'), (61, '1'),
                              (130, '3'), (300, '3')):
        assert running_brillouin_repetition(
            file, t0 + timedelta(minutes=minutes), recorded=0) == expected
    # before the first remaining repetition: the one acquired next
    assert running_brillouin_repetition(
        file, t0 - timedelta(minutes=5), recorded=0) == '0'
    # no capture time: the stored number
    assert running_brillouin_repetition(file, None, recorded=3) == '3'
    assert running_brillouin_repetition(file, None) is None


def test_untagged_images_taken_during_a_run_use_the_running_repetition():
    from bmlab.export.naming import running_brillouin_repetition
    t0 = datetime(2026, 7, 30, 11, 0, tzinfo=timezone.utc)
    file = _stub_file({'0': t0})
    assert running_brillouin_repetition(
        file, t0 + timedelta(minutes=8)) == '0'


def test_tmatrix_is_always_an_exact_turn_of_the_camera():
    """
    Regression test: the 10x objective's measured calibration differs by
    0.3 % between the axes (pixel size 2.1466 against 2.14). That is
    measurement noise - the camera is turned by a multiple of 90 degrees
    - so get_tmatrix() returns exactly that turn, images are always
    rotated losslessly and their pixel values never change (resampling
    returned integer images scaled to 0..1, which came out black).
    """
    from bmlab.export.alignment import get_tmatrix
    calibration = {'micrometerToPixX': [0.0, 2.1466],
                   'micrometerToPixY': [-2.14, 0.0]}
    tmatrix = get_tmatrix(calibration)
    assert np.array_equal(np.asarray(tmatrix),
                          [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0]])
    rng = np.random.default_rng(1)
    img = rng.integers(10, 60, size=(30, 40)).astype(np.uint8)
    oriented, _ = orient_image(img, tmatrix)
    assert oriented.dtype == np.uint8
    assert np.array_equal(np.sort(oriented.ravel()), np.sort(img.ravel()))


@pytest.mark.parametrize('x_vector, y_vector', [
    ([0.0, 4.3], [-4.3, 0.0]),
    ([0.0, -4.3], [4.3, 0.0]),
    ([4.3, 0.0], [0.0, 4.3]),
    ([-4.3, 0.0], [0.0, -4.3]),
    ([4.31, 0.02], [-0.03, 4.28]),   # measured, slightly off
])
def test_tmatrix_keeps_orientation_and_sign(x_vector, y_vector):
    from bmlab.export.alignment import get_tmatrix
    tmatrix = np.asarray(get_tmatrix(
        {'micrometerToPixX': x_vector, 'micrometerToPixY': y_vector}))
    linear = tmatrix[:2, :2]
    assert set(np.unique(linear)) <= {-1.0, 0.0, 1.0}
    assert np.abs(linear).sum(axis=0).tolist() == [1.0, 1.0]
    assert np.abs(linear).sum(axis=1).tolist() == [1.0, 1.0]
    assert abs(abs(np.linalg.det(linear)) - 1.0) < 1e-9
    assert tmatrix[2].tolist() == [0.0, 0.0, 1.0]


@pytest.mark.parametrize('x_vector, y_vector', [
    ([0.0, 2.0], [-2.0, 0.0]),     # this setup's camera
    ([0.0, -2.0], [2.0, 0.0]),
    ([2.0, 0.0], [0.0, 2.0]),
    ([-2.0, 0.0], [0.0, -2.0]),
])
def test_transform_matrix_puts_a_stage_point_on_its_pixel(x_vector, y_vector):
    """
    End to end: a marker is put on the raw pixel that a stage point maps
    to (anchor at the image centre, stage_map per um); after the image is
    turned into the stage's axes, the written matrix must send that stage
    point to the marker's pixel.
    """
    from bmlab.export.alignment import (
        get_tmatrix, get_point_to_pixel_matrix)
    calibration = {'micrometerToPixX': x_vector,
                   'micrometerToPixY': y_vector}
    stage_map = calibration_stage_to_pixel(calibration)
    tmatrix = get_tmatrix(calibration)
    height, width = 41, 61                    # centre at (col 30, row 20)
    anchor = (100.0, 50.0)
    for dx, dy in ((5.0, -7.0), (-8.0, 3.0), (0.0, 0.0), (10.0, 6.0)):
        d_col, d_row = stage_map @ np.array([dx, dy])
        col, row = int(round(30 + d_col)), int(round(20 + d_row))
        assert 0 <= col < width and 0 <= row < height
        img = np.zeros((height, width), dtype=np.uint8)
        img[row, col] = 255
        oriented, translate = orient_image(img, tmatrix)
        found_row, found_col = np.unravel_index(
            np.argmax(oriented), oriented.shape)
        matrix = get_point_to_pixel_matrix(
            tmatrix, stage_map, anchor, img.shape, translate)
        predicted = matrix @ np.array([anchor[0] + dx, anchor[1] + dy, 1.0])
        assert predicted[:2] == pytest.approx([found_col, found_row],
                                              abs=1e-9)


def test_transform_matrix_is_none_without_calibration_or_position():
    from bmlab.export.alignment import get_point_to_pixel_matrix
    eye = np.matrix(np.eye(3))
    assert get_point_to_pixel_matrix(None, np.eye(2), (0, 0), (4, 4), eye) \
        is None
    assert get_point_to_pixel_matrix(eye, None, (0, 0), (4, 4), eye) is None
    assert get_point_to_pixel_matrix(eye, np.eye(2), None, (4, 4), eye) \
        is None
