"""
BrillouinAcquisition's overview engine (channel 'Brightfield overview',
per-image metadata, overview events) and its drift correction while
measuring (per-point grid offsets): reading them, the overview export and
the drift columns of the Brillouin export.

The test files predate all of it; the data BrillouinAcquisition writes is
added to copies of them with h5py.
"""
from pathlib import Path
import csv
import os
import shutil
import uuid

import h5py
import numpy as np
import pytest

from bmlab.controllers import ExportController
from bmlab.file import BrillouinFile, OVERVIEW_EVENT_CHANNEL
from bmlab.session import Session


@pytest.fixture()
def tmp_dir():
    current_path = Path.cwd()
    tmp_dir = Path.cwd() / f"tmp{str(uuid.uuid4())}" / 'RawData'
    os.makedirs(tmp_dir, exist_ok=True)
    os.chdir(tmp_dir)
    yield tmp_dir
    os.chdir(current_path)
    try:
        shutil.rmtree(tmp_dir.parent)
    except Exception as e:
        print(e)


def data_file_path(file_name):
    return Path(__file__).parent / 'data' / file_name


def _with_length(values):
    """How BrillouinAcquisition writes the numbers of the resume plan."""
    values = np.asarray(values, dtype=float)
    return np.concatenate([[values.size], values])


def _put(group, name, values):
    group.create_dataset(name, data=np.atleast_1d(np.asarray(values, float)))


DECISIONS = {'reference': 0, 'applied': 1, 'no match': 2}


def add_runtime_drift(path, decisions=('reference', 'applied', 'applied')):
    """
    Drift correction data as BrillouinAcquisition writes it into the
    Brillouin repetition 0 of 2D-xy.h5 (3 x 5 x 1 points). The points are
    measured in reverse address order; the point measured n-th was
    measured with an offset of (0.5 n, -0.25 n) um (all zero for
    decisions that never apply).
    """
    applied = any(d == 'applied' for d in decisions)
    with h5py.File(path, 'r+') as f:
        payload = f['Brillouin/0/payload']
        nz, nx, ny = payload['positions-x'].shape
        count = nx * ny * nz
        address = np.arange(count)[::-1]
        index = (address % nx, (address // nx) % ny, address // (nx * ny))
        for axis, values in zip('xyz', index):
            _put(payload, 'positions-resume-plan-index-' + axis,
                 _with_length(values))
        order = np.arange(count, dtype=float)
        _put(payload, 'positions-grid-offset-per-point-x-um',
             0.5 * order if applied else 0 * order)
        _put(payload, 'positions-grid-offset-per-point-y-um',
             -0.25 * order if applied else 0 * order)
        _put(payload, 'positions-drift-correction-used', 1)
        _put(payload, 'positions-drift-objective-slot-used', 6)
        _put(payload, 'positions-drift-min-ncc-used', 0.3)
        _put(payload, 'positions-drift-max-step-um-used', 1000)
        _put(payload, 'positions-drift-extrapolate-used', 0)
        _put(payload, 'positions-drift-carried-over-x-um', 0)
        _put(payload, 'positions-drift-carried-over-y-um', 0)
        for number, decision in enumerate(decisions):
            prefix = 'positions-overview-event-%04d-' % number
            _put(payload, prefix + 'kind', 0 if number == 0 else 1)
            _put(payload, prefix + 'z-index', max(number - 1, 0))
            _put(payload, prefix + 'point-index', -1 if number == 0 else 5)
            _put(payload, prefix + 'seconds-since-start', 10.0 * number)
            _put(payload, prefix + 'repetition', 0)
            _put(payload, prefix + 'objective-slot', 6)
            _put(payload, prefix + 'objective-captured', 1)
            _put(payload, prefix + 'objective-tiles', 1)
            _put(payload, prefix + 'objective-stack-depth', 2)
            _put(payload, prefix + 'objective-images-saved', 2)
            _put(payload, prefix + 'drift-evaluated', 1)
            _put(payload, prefix + 'drift-decision', DECISIONS[decision])
            _put(payload, prefix + 'drift-step-um', [1.0, -2.0])
            _put(payload, prefix + 'drift-total-um', [1.0 * number, 0.0])
            _put(payload, prefix + 'grid-offset-after-um',
                 [1.0 * number, 0.0])
            _put(payload, prefix + 'drift-mean-ncc', 0.9)
            _put(payload, prefix + 'drift-tiles-compared-used', [1, 1])
        f['Brillouin/0'].attrs['acquisition_state'] = np.array([b'finished'])


def export_brillouin(path, name):
    session = Session.get_instance()
    session.set_file(Path(path))
    session.set_current_repetition('0')
    evm = session.evaluation_model()
    resolution = session.get_payload_resolution()
    shape = (resolution[0], resolution[1], resolution[2], 1, 1, 1)
    evm.results['time'] = np.ones(shape)
    evm.results['intensity'] = np.full(shape, 100.0)

    ec = ExportController()
    config = ec.get_configuration()
    config['fluorescence']['export'] = False
    config['fluorescenceCombined']['export'] = False
    config['surface']['export'] = False
    config['overviewBrightfield']['export'] = False
    ec.export(config)
    session.clear()


def read_csv(path):
    with open(path, newline='') as f:
        lines = f.read().splitlines()
    header = dict(
        row for row in csv.reader(
            [line for line in lines if line.startswith('#')])
        if len(row) >= 2)
    table = [row for row in csv.reader(
        [line for line in lines if not line.startswith('#')])]
    columns = table[0]
    return header, columns, np.array(table[1:], dtype=float)


def read_header(path):
    with open(path, newline='') as f:
        return dict(
            row for row in csv.reader(
                line for line in f if line.startswith('#'))
            if len(row) >= 2)


def test_runtime_drift_is_read(tmp_dir):
    shutil.copy(data_file_path('2D-xy.h5'), Path.cwd() / '2D-xy.h5')
    add_runtime_drift(Path.cwd() / '2D-xy.h5')

    file = BrillouinFile(Path.cwd() / '2D-xy.h5')
    repetition = file.get_repetition('0')
    assert repetition.acquisition_state == 'finished'
    payload = repetition.payload

    events = payload.get_overview_events()
    assert [e['index'] for e in events] == [0, 1, 2]
    assert [e['drift']['decision_name'] for e in events] == \
        ['reference', 'applied', 'applied']
    assert events[0]['kind_name'] == 'before measurement'
    assert events[1]['kind_name'] == 'after z plane'
    assert events[2]['drift']['grid_offset_after_um'] == (2.0, 0.0)
    assert events[1]['objectives'][0]['stack_depth'] == 2

    plan = payload.get_scan_plan()
    # reverse address order, the length in front of the numbers is gone
    assert list(plan['address']) == list(range(14, -1, -1))

    offsets = payload.get_point_drift_offsets()
    assert offsets['x'].shape == payload.positions['x'].shape
    # the point with address 14 was measured first, without offset
    assert offsets['x'].ravel()[14] == 0.0
    assert offsets['x'].ravel()[0] == pytest.approx(0.5 * 14)
    assert offsets['y'].ravel()[0] == pytest.approx(-0.25 * 14)

    drift = payload.get_runtime_drift()
    assert drift['applied'] is True
    assert drift['applied_events'] == 2
    assert drift['max_offset_um'] == pytest.approx(np.hypot(7.0, 3.5))
    assert drift['objective_slot'] == 6
    file.close()


def test_runtime_drift_is_none_without_the_setting(tmp_dir):
    shutil.copy(data_file_path('2D-xy.h5'), Path.cwd() / '2D-xy.h5')
    file = BrillouinFile(Path.cwd() / '2D-xy.h5')
    payload = file.get_repetition('0').payload
    assert payload.get_runtime_drift() is None
    assert payload.get_overview_events() == []
    assert payload.get_scan_plan() is None
    assert payload.get_point_drift_offsets() is None
    assert payload.get_resume_events() == []
    assert payload.get_surface_rescans() == []
    assert file.get_repetition('0').acquisition_state is None
    file.close()


def test_export_runtime_drift_applied(tmp_dir):
    shutil.copy(data_file_path('2D-xy.h5'), Path.cwd() / '2D-xy.h5')
    add_runtime_drift(Path.cwd() / '2D-xy.h5')
    export_brillouin(Path('2D-xy.h5'), '2D-xy')

    export_dir = tmp_dir.parent / 'Export'
    header, columns, data = read_csv(export_dir / '2D-xy_BMrep0_data.csv')
    assert header['#drift_runtime_applied'] == 'True'
    assert header['#drift_runtime_applied_events'] == '2'
    assert header['#drift_runtime_decisions'] == (
        '0:reference;1:applied;2:applied')
    assert header['#acquisition_state'] == 'finished'
    assert header['#drift_correction_used'] == '1.0'
    # no second copy of the data with the points moved: the plain export
    # keeps its columns and x, y (planned = position on the sample)
    assert 'drift_x_um' not in columns
    assert not list(export_dir.glob('*driftcorrected*'))

    # one line per drift step instead
    header = read_header(export_dir / '2D-xy_BMrep0_runtime_drift.csv')
    assert header['#drift_applied'] == 'True'
    assert header['#drift_applied_events'] == '2'
    assert header['#drift_min_ncc'] == '0.3'
    assert header['#drift_objective_slot'] == '6.0'
    with open(export_dir / '2D-xy_BMrep0_runtime_drift.csv',
              newline='') as f:
        rows = list(csv.DictReader(
            line for line in f if not line.startswith('#')))
    assert [r['event'] for r in rows] == ['0', '1', '2']
    assert [r['decision'] for r in rows] == [
        'reference', 'applied', 'applied']
    assert [r['kind'] for r in rows] == [
        'before measurement', 'after z plane', 'after z plane']
    assert rows[2]['step_x_um'] == '1.0' and rows[2]['step_y_um'] == '-2.0'
    assert rows[2]['total_x_um'] == '2.0'
    assert rows[2]['grid_offset_after_x_um'] == '2.0'
    assert rows[2]['mean_ncc'] == '0.9'
    assert (rows[2]['tiles_compared'], rows[2]['tiles_used']) == ('1', '1')
    assert rows[2]['point_index'] == '5'


def test_export_runtime_drift_without_match_is_documented(tmp_dir):
    shutil.copy(data_file_path('2D-xy.h5'), Path.cwd() / '2D-xy.h5')
    add_runtime_drift(Path.cwd() / '2D-xy.h5',
                      decisions=('no match', 'no match'))
    export_brillouin(Path('2D-xy.h5'), '2D-xy')

    export_dir = tmp_dir.parent / 'Export'
    header, _, _ = read_csv(export_dir / '2D-xy_BMrep0_data.csv')
    assert header['#drift_runtime_applied'] == 'False'
    assert header['#drift_runtime_decisions'] == '0:no match;1:no match'
    assert header['#drift_runtime_max_offset_um'] == '0.0'
    # the steps are listed all the same: what was tried and why it failed
    with open(export_dir / '2D-xy_BMrep0_runtime_drift.csv',
              newline='') as f:
        rows = list(csv.DictReader(
            line for line in f if not line.startswith('#')))
    assert [r['decision'] for r in rows] == ['no match', 'no match']
    assert all(r['grid_offset_after_x_um'] == '0.0' or
               r['grid_offset_after_x_um'] == '1.0' for r in rows)


def test_grid_definition_is_in_the_header(tmp_dir):
    shutil.copy(data_file_path('2D-xy.h5'), Path.cwd() / '2D-xy.h5')
    with h5py.File(Path.cwd() / '2D-xy.h5', 'r+') as f:
        payload = f['Brillouin/0/payload']
        _put(payload, 'positions-grid-x-range-um-used', [-1, 1])
        _put(payload, 'positions-grid-y-range-um-used', [-2, 2])
        _put(payload, 'positions-grid-z-range-um-used', [0, 0])
        for axis, value in zip('xyz', (1500.5, -20.25, 3.0)):
            _put(payload, 'positions-absolute-origin-' + axis, value)
        _put(payload, 'positions-grid-coordinates-absolute', 1)
        _put(payload, 'positions-grid-scan-order-used', [0, 1, 2])
        _put(payload, 'positions-roi-polygon-x-um', [0, 10, 10])
        _put(payload, 'positions-roi-polygon-y-um', [0, 0, 5])
    export_brillouin(Path('2D-xy.h5'), '2D-xy')

    header, _, _ = read_csv(tmp_dir.parent / 'Export' / '2D-xy_BMrep0_data.csv')
    assert header['#grid_x_steps'] == '3' and header['#grid_y_steps'] == '5'
    assert header['#grid_x_min_um'] == '-1.0'
    assert header['#grid_x_max_um'] == '1.0'
    assert header['#grid_x_step_um'] == '1.0'
    assert header['#grid_y_step_um'] == '1.0'
    assert header['#grid_z_steps'] == '1'
    assert '#grid_z_step_um' not in header      # a single plane
    assert header['#grid_absolute_origin_x_um'] == '1500.5'
    assert header['#grid_absolute_origin_y_um'] == '-20.25'
    assert header['#grid_scan_loop_level_x'] == '0'
    assert header['#grid_scan_loop_level_z'] == '2'
    assert header['#grid_coordinates_absolute'] == 'True'
    assert header['#roi_polygon_x_um'] == '0.00;10.00;10.00'
    assert header['#roi_polygon_y_um'] == '0.00;0.00;5.00'


# ------------------------------------------------------------------ overviews

SCALE = 10.0   # px/um of the objective that took the overviews (the
#                repetition's own calibration says 20)


def add_overview_engine_images(path):
    """
    The 4 overview images of SurfaceScan.h5 as the overview engine writes
    them: two events of a z stack of 2 at one tile. The second event was
    taken after the grid had been moved by (4, -3) um (the stage target
    carries the drift offset).
    """
    centre_shift = (5.0, -7.0)
    fov = (1.5, 2.5)
    with h5py.File(path, 'r+') as f:
        for index in range(4):
            event, stack = divmod(index, 2)
            offset = (4.0, -3.0) if event == 1 else (0.0, 0.0)
            # the middle of the image is at the planned position (100, 200)
            target = (100.0 + centre_shift[0] + fov[0] + offset[0],
                      200.0 + centre_shift[1] + fov[1] + offset[1])
            attrs = f['Fluorescence/0/payload/data/%d' % index].attrs
            attrs['channel'] = np.array([b'Brightfield overview'])
            attrs['brillouin_repetition_index'] = np.array([0])
            numbers = {
                'overview_event': event, 'overview_kind': event,
                'overview_z_index': event, 'overview_point_index': 5 * event,
                'overview_objective_slot': 6,
                'overview_is_measurement_objective': 0,
                'overview_tile_index': 0, 'overview_tile_count': 1,
                'overview_stack_index': stack, 'overview_stack_count': 2,
                'overview_is_centre_slice': stack,
                'overview_seconds_since_start': 100.0 * event,
                'objective_magnification': 10.0,
                'grid_offset_x_um': offset[0], 'grid_offset_y_um': offset[1],
                'fov_compensation_x_um': fov[0],
                'fov_compensation_y_um': fov[1],
                'image_centre_shift_x_um': centre_shift[0],
                'image_centre_shift_y_um': centre_shift[1],
                'position_x_um': target[0], 'position_y_um': target[1],
                'position_z_um': 10.0 * stack,
                'stage_position_x_um': target[0] + 0.1,
                'stage_position_y_um': target[1] - 0.1,
                'stage_position_z_um': 10.0 * stack,
                'scale_micrometer_to_pix_x_col': 0.0,
                'scale_micrometer_to_pix_x_row': SCALE,
                'scale_micrometer_to_pix_y_col': SCALE,
                'scale_micrometer_to_pix_y_row': 0.0,
                'scale_origin_col': 0.0, 'scale_origin_row': 0.0,
            }
            for name, value in numbers.items():
                attrs[name] = np.array([float(value)])
            attrs['overview_kind_name'] = np.array(
                [b'before measurement' if event == 0 else b'after z plane'])
            attrs['objective_name'] = np.array([b'10x'])
        # what the Brillouin side recorded about the events
        payload = f['Brillouin/0/payload']
        for event, decision in enumerate(('reference', 'applied')):
            prefix = 'positions-overview-event-%04d-' % event
            _put(payload, prefix + 'kind', event)
            _put(payload, prefix + 'drift-evaluated', 1)
            _put(payload, prefix + 'drift-decision', DECISIONS[decision])
            _put(payload, prefix + 'drift-step-um',
                 [4.0 * event, -3.0 * event])
            _put(payload, prefix + 'drift-total-um',
                 [4.0 * event, -3.0 * event])
            _put(payload, prefix + 'grid-offset-after-um',
                 [4.0 * event, -3.0 * event])
            _put(payload, prefix + 'drift-mean-ncc', 0.95)
            _put(payload, prefix + 'drift-tiles-compared-used', [1, 1])
    return centre_shift, fov


def test_overview_info_of_an_image(tmp_dir):
    shutil.copy(data_file_path('SurfaceScan.h5'), Path.cwd() / 'S.h5')
    add_overview_engine_images(Path.cwd() / 'S.h5')

    file = BrillouinFile(Path.cwd() / 'S.h5')
    payload = file.get_repetition('0', 'Fluorescence').payload
    assert payload.image_keys_by_channel(OVERVIEW_EVENT_CHANNEL) == \
        ['0', '1', '2', '3']
    info = payload.get_overview_info('3')
    assert info['event'] == 1
    assert info['kind_name'] == 'after z plane'
    assert info['stack_index'] == 1 and info['is_centre_slice']
    assert info['repetition'] == 0
    assert info['grid_offset'] == (4.0, -3.0)
    # the middle of the image is the planned position again, whatever the
    # shifts and the drift offset were (the read-back stage position is
    # used: 0.1 / -0.1 um off the target)
    assert info['anchor']['x'] == pytest.approx(100.1)
    assert info['anchor']['y'] == pytest.approx(199.9)
    assert info['scale_calibration']['micrometerToPixY'] == (SCALE, 0.0)
    # an image of another channel has none of it
    legacy = BrillouinFile(data_file_path('Fluorescence.h5'))
    for key in legacy.repetition_keys('Fluorescence'):
        p = legacy.get_repetition(key, 'Fluorescence').payload
        assert all(p.get_overview_info(k) is None for k in p.image_keys())
    legacy.close()
    file.close()


def test_export_overview_engine_images(tmp_dir):
    shutil.copy(data_file_path('SurfaceScan.h5'), Path.cwd() / 'S.h5')
    add_overview_engine_images(Path.cwd() / 'S.h5')

    session = Session.get_instance()
    session.set_file(Path('S.h5'))
    session.set_current_repetition('0')
    ec = ExportController()
    config = ec.get_configuration()
    config['fluorescenceCombined']['export'] = False
    config['brillouin']['export'] = False
    config['surface']['export'] = False
    ec.export(config)
    session.clear()

    plots = tmp_dir.parent / 'Plots'
    folder = plots / 'Overview_BMrep0'
    files = sorted(folder.glob('overview_*.tif'),
                   key=lambda p: int(p.name.split('_')[1]))
    # one file per event (its z stack in one file), in event order
    assert [p.name.split('_')[1] for p in files] == ['1', '2']
    # and they are not exported a second time as plain images
    assert not list(plots.glob('FL_*/*verview*'))

    for overview in files:
        matrix = np.loadtxt(
            folder / 'TransformMatrices' / (overview.stem + '_transform.csv'),
            delimiter=',')
        # the planned position (100, 200) is the middle of the 20 x 20
        # image in both events - the second one was taken after the grid
        # had been moved by the drift correction
        column, row, _ = matrix @ np.array([100.1, 199.9, 1.0])
        assert (column, row) == pytest.approx((9.5, 9.5), abs=1e-6)
        # this objective's own calibration (10 px/um), not the
        # repetition's (20 px/um)
        assert np.abs(matrix[:2, :2]).sum(axis=1) == \
            pytest.approx([SCALE, SCALE])

    with open(folder / 'overview_events.csv', newline='') as f:
        rows = list(csv.DictReader(
            line for line in f if not line.startswith('#')))
    assert [r['event'] for r in rows] == ['0', '1']
    assert [r['kind'] for r in rows] == ['before measurement',
                                         'after z plane']
    assert 'drift_decision' not in rows[0]       # that is in the drift file
    assert rows[1]['grid_offset_x_um'] == '4.0'
    assert rows[1]['file'] == files[1].name
    assert rows[0]['images'] == '2' and rows[0]['stack_depth'] == '2'


def test_overview_engine_images_of_a_removed_repetition(tmp_dir):
    """
    The engine records the repetition's key in the file. When that
    repetition was removed afterwards (an aborted start) the images are
    put with the repetition that was running then, or the next one.
    """
    shutil.copy(data_file_path('SurfaceScan.h5'), Path.cwd() / 'S.h5')
    add_overview_engine_images(Path.cwd() / 'S.h5')
    with h5py.File(Path.cwd() / 'S.h5', 'r+') as f:
        for index in range(4):
            f['Fluorescence/0/payload/data/%d' % index].attrs[
                'brillouin_repetition_index'] = np.array([7])

    session = Session.get_instance()
    session.set_file(Path('S.h5'))
    ec = ExportController()
    config = ec.get_configuration()
    for name in ('fluorescence', 'fluorescenceCombined', 'brillouin',
                 'surface'):
        config[name]['export'] = False
    ec.export(config)
    session.clear()

    plots = tmp_dir.parent / 'Plots'
    assert not (plots / 'Overview_BMrep7').exists()
    assert list((plots / 'Overview_BMrep0').glob('overview_*.tif'))


# -------------------------------------------------- resume, rescans, objectives

def test_resume_events_rescans_and_overview_objectives_are_read(tmp_dir):
    shutil.copy(data_file_path('2D-xy.h5'), Path.cwd() / '2D-xy.h5')
    with h5py.File(Path.cwd() / '2D-xy.h5', 'r+') as f:
        payload = f['Brillouin/0/payload']
        # a measurement that was cancelled after 7 of 15 points and
        # continued once (numbers are written with their length in front)
        for name, value in (('point-index', 7), ('points-total', 15),
                            ('resumed-epoch-seconds', 1.7e9),
                            ('was-cancelled', 1),
                            ('incomplete-points-removed', 2)):
            _put(payload, 'positions-resume-event-0000-' + name,
                 _with_length([value]))
        for name, value in (('previous-state', 'stopped'),
                            ('last-point-date', '2024-01-01T23:05:52'),
                            ('resumed-date', '2024-01-01T23:30:00')):
            payload.attrs['resume_event-0000-' + name] = \
                np.array([('s:' + value).encode()])
        # one repeated surface scan
        prefix = 'positions-surface-rescan-0000-'
        _put(payload, prefix + 'success', 1)
        _put(payload, prefix + 'point-index', 221)
        _put(payload, prefix + 'seconds-since-start', 300.0)
        _put(payload, prefix + 'duration-seconds', 42.0)
        _put(payload, prefix + 'grid-offset-um', [1.5, -2.5])
        _put(payload, prefix + 'total-columns-failed-columns', [30, 2])
        _put(payload, prefix + 'points-without-new-surface', 4)
        _put(payload, prefix + 'prior-used', 1)
        _put(payload, prefix + 'reference-threshold', 430.4)
        _put(payload, 'positions-surface-rescan-used', 2)
        # the calibration of the objective that takes the overviews
        _put(payload, 'positions-overview-objective-6-magnification', 10)
        _put(payload, 'positions-overview-objective-6-micrometer-to-pix-x',
             [0.0, 2.15])
        _put(payload, 'positions-overview-objective-6-fov-offset-um',
             [1.0, 119.2, -166.3, 0.4])
        _put(payload, 'positions-overview-objective-6-focus-offset-um',
             [1.0, -5.75, 0.2])
        _put(payload, 'positions-overview-objective-6-is-reference-objective',
             0)

    file = BrillouinFile(Path.cwd() / '2D-xy.h5')
    payload = file.get_repetition('0').payload

    resumed = payload.get_resume_events()
    assert len(resumed) == 1
    assert resumed[0]['point_index'] == 7
    assert resumed[0]['points_total'] == 15
    assert resumed[0]['was_cancelled'] is True
    assert resumed[0]['incomplete_points_removed'] == 2
    assert resumed[0]['previous_state'] == 'stopped'
    assert resumed[0]['resumed_date'] == '2024-01-01T23:30:00'

    rescans = payload.get_surface_rescans()
    assert len(rescans) == 1
    assert rescans[0]['success'] is True
    assert rescans[0]['point_index'] == 221
    assert rescans[0]['grid_offset_um'] == (1.5, -2.5)
    assert (rescans[0]['total_columns'], rescans[0]['failed_columns']) == \
        (30, 2)
    assert rescans[0]['points_without_new_surface'] == 4
    assert rescans[0]['prior_used'] is True
    assert payload.get_acquisition_settings()['surface_rescan_used'] == 2

    objective = payload.get_overview_objectives()[6]
    assert objective['magnification'] == 10
    assert objective['micrometerToPixX'] == (0.0, 2.15)
    assert objective['fov_offset'] == (119.2, -166.3)
    assert objective['has_focus_offset'] is True
    assert objective['focus_offset_um'] == pytest.approx(-5.75)
    assert objective['is_reference_objective'] is False
    file.close()


def test_replaced_reference_is_recorded(tmp_dir):
    shutil.copy(data_file_path('2D-xy.h5'), Path.cwd() / '2D-xy.h5')
    add_runtime_drift(Path.cwd() / '2D-xy.h5',
                      decisions=('reference', 'applied'))
    with h5py.File(Path.cwd() / '2D-xy.h5', 'r+') as f:
        payload = f['Brillouin/0/payload']
        _put(payload, 'positions-overview-event-0000-drift-reference-'
             'replaced', 1)
        _put(payload, 'positions-overview-event-0001-drift-reference-'
             'replaced', 0)

    file = BrillouinFile(Path.cwd() / '2D-xy.h5')
    events = file.get_repetition('0').payload.get_overview_events()
    file.close()
    assert [e['drift']['reference_replaced'] for e in events] == \
        [True, False]

    export_brillouin(Path('2D-xy.h5'), '2D-xy')
    with open(tmp_dir.parent / 'Export' / '2D-xy_BMrep0_runtime_drift.csv',
              newline='') as f:
        rows = list(csv.DictReader(
            line for line in f if not line.startswith('#')))
    assert [r['reference_replaced'] for r in rows] == ['1', '0']
