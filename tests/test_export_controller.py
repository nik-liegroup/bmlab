from pathlib import Path
import csv
import json
import pytest
import shutil
import uuid
import os

import h5py
import numpy as np

from bmlab.session import Session
from bmlab.controllers import ExportController, EvaluationController


@pytest.fixture()
def tmp_dir():
    current_path = Path.cwd()
    tmp_dir = Path.cwd() / f"tmp{str(uuid.uuid4())}" / 'RawData'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir, exist_ok=True)
    os.chdir(tmp_dir)
    yield tmp_dir
    os.chdir(current_path)
    try:
        shutil.rmtree(tmp_dir.parent)
    # Windows sometimes does not correctly close the HDF file. Then we have a
    # file access conflict.
    except Exception as e:
        print(e)


def data_file_path(file_name):
    return Path(__file__).parent / 'data' / file_name


def assert_glob_exists(directory, pattern):
    """
    Asserts at least one file in `directory` matches `pattern` - used
    wherever a filename contains a repetition_or_timestamp_tag()
    timestamp (see bmlab.export.naming), which isn't predictable from
    the fixture alone the way a '_BMrepN' tag is.
    """
    matches = list(directory.glob(pattern))
    assert matches, f"no file matching '{pattern}' in {directory}"
    return matches[0]


def test_export_fluorescence_respects_brillouin_repetition_filter(tmp_dir):
    """
    Regression test: FluorescenceExport/FluorescenceCombinedExport must
    also respect config['brillouin']['repetitions'] for Fluorescence
    repetitions tied to a Brillouin one (see MeasurementData.
    get_brillouin_repetition_index()) - a Fluorescence repetition with
    no such tie (a standalone Fluorescence-tab capture) is exported
    regardless, since there is no repetition choice for it to be
    excluded by.
    """
    shutil.copy(
        data_file_path('Fluorescence.h5'), Path.cwd() / 'Fluorescence.h5')

    with h5py.File(Path.cwd() / 'Fluorescence.h5', 'r+') as f:
        for idx in range(4):
            f[f'Fluorescence/0/payload/data/{idx}'].attrs[
                'brillouin_repetition_index'] = np.array([0])
        # Repetition '1' is left without the attribute - a standalone
        # capture, never excluded by the repetition selection.

    session = Session.get_instance()
    session.set_file(Path('Fluorescence.h5'))

    ec = ExportController()
    config = ec.get_configuration()
    config['brillouin']['export'] = False
    # Exclude Brillouin repetition '0' - repetition 0's Fluorescence
    # images must disappear, repetition 1's must remain.
    config['brillouin']['repetitions'] = []
    ec.export(config)

    session.clear()

    plots_dir = tmp_dir.parent / 'Plots'
    for channel in ('Blue', 'Green', 'Red', 'Brightfield'):
        assert not list(plots_dir.glob(f'{channel}_*_FLrep0.png'))
        assert_glob_exists(plots_dir, f'{channel}_*_FLrep1.png')

    combined_dir = tmp_dir.parent / 'Plots' / 'Bare'
    assert not list(combined_dir.glob('fluorescenceCombined_rgb_*_FLrep0.png'))
    assert_glob_exists(
        combined_dir, 'fluorescenceCombined_rgb_*_FLrep1.png')


def test_export_fluorescence(tmp_dir):
    shutil.copy(
        data_file_path('Fluorescence.h5'), Path.cwd() / 'Fluorescence.h5')

    session = Session.get_instance()
    session.set_file(Path('Fluorescence.h5'))

    ec = ExportController()
    config = ec.get_configuration()
    config['fluorescenceCombined']['export'] = False
    config['brillouin']['export'] = False
    ec.export(config)

    session.clear()

    plots_dir = tmp_dir.parent / 'Plots'
    # Fluorescence.h5 has a scale calibration and no Brillouin
    # repetitions, so only the aligned image is exported (no camera-
    # pixel-space raw image). None of its images carry a
    # brillouin_repetition_index, so each filename gets a capture-
    # timestamp tag (see bmlab.export.naming.repetition_or_timestamp_tag)
    # instead of a '_BMrepN' one.
    for channel in ('Blue', 'Brightfield', 'Green', 'Red'):
        for rep in ('0', '1'):
            assert_glob_exists(plots_dir, f'{channel}_*_FLrep{rep}.png')


def test_export_fluorescence_transform_matrix(tmp_dir):
    """
    Regression test: FluorescenceExport/FluorescenceCombinedExport must
    write a stage-position -> pixel transform matrix for every aligned
    image they export (see bmlab.export.alignment.write_transform_csv()
    - same layout as OverviewBrightfieldExport already writes for its
    own images), so a fluorescence image can actually be placed
    relative to the Brillouin CSV grid or an overview image. Without
    it there is no way to overlay a fluorescence image with anything
    else exported. Fluorescence.h5 itself carries no recorded per-image
    position - this test adds one directly (mimicking what a real
    BrillouinAcquisition capture writes) to exercise that path.
    """
    shutil.copy(
        data_file_path('Fluorescence.h5'), Path.cwd() / 'Fluorescence.h5')

    with h5py.File(Path.cwd() / 'Fluorescence.h5', 'r+') as f:
        for rep in ('0', '1'):
            for idx in range(4):
                ds = f[f'Fluorescence/{rep}/payload/data/{idx}']
                ds.attrs['position_x_um'] = np.array([100.0 + int(rep)])
                ds.attrs['position_y_um'] = np.array([50.0])
                ds.attrs['position_z_um'] = np.array([0.0])

    session = Session.get_instance()
    session.set_file(Path('Fluorescence.h5'))

    ec = ExportController()
    config = ec.get_configuration()
    config['brillouin']['export'] = False
    ec.export(config)

    session.clear()

    plots_dir = tmp_dir.parent / 'Plots'
    transform_dir = plots_dir / 'TransformMatrices'
    assert transform_dir.is_dir()

    def assert_valid_matrix(csv_path):
        assert csv_path.exists()
        with open(csv_path, newline='') as f:
            rows = list(csv.reader(f))
        matrix = np.array(rows, dtype=float)
        assert matrix.shape == (3, 3)
        assert matrix[2].tolist() == [0.0, 0.0, 1.0]
        return matrix

    for channel in ('Blue', 'Green', 'Red', 'Brightfield'):
        png = assert_glob_exists(plots_dir, f'{channel}_*_FLrep0.png')
        assert_valid_matrix(
            transform_dir / (png.stem + '_transform.csv'))

    # Both repetitions were given a different x position (see above) -
    # their transform matrices must actually differ, not just both
    # happen to exist.
    png_rep0 = assert_glob_exists(plots_dir, 'Blue_*_FLrep0.png')
    png_rep1 = assert_glob_exists(plots_dir, 'Blue_*_FLrep1.png')
    matrix_rep0 = assert_valid_matrix(
        transform_dir / (png_rep0.stem + '_transform.csv'))
    matrix_rep1 = assert_valid_matrix(
        transform_dir / (png_rep1.stem + '_transform.csv'))
    assert not np.allclose(matrix_rep0, matrix_rep1)

    combined_transform_dir = \
        tmp_dir.parent / 'Plots' / 'Bare' / 'TransformMatrices'
    assert combined_transform_dir.is_dir()
    combined_png = assert_glob_exists(
        tmp_dir.parent / 'Plots' / 'Bare',
        'fluorescenceCombined_rgb_*_FLrep0.png')
    assert_valid_matrix(
        combined_transform_dir / (combined_png.stem + '_transform.csv'))


def test_export_color_images(tmp_dir):
    shutil.copy(
        data_file_path('ColorImage.h5'), Path.cwd() / 'ColorImage.h5')

    session = Session.get_instance()
    session.set_file(Path('ColorImage.h5'))

    ec = ExportController()
    config = ec.get_configuration()
    config['fluorescenceCombined']['export'] = False
    config['brillouin']['export'] = False
    ec.export(config)

    session.clear()

    plots_dir = tmp_dir.parent / 'Plots'
    assert_glob_exists(plots_dir, 'Brightfield_*_FLrep0.png')


def test_export_color_images_plane(tmp_dir):
    shutil.copy(
        data_file_path('ColorImagePlane.h5'),
        Path.cwd() / 'ColorImagePlane.h5')

    session = Session.get_instance()
    session.set_file(Path('ColorImagePlane.h5'))

    ec = ExportController()
    config = ec.get_configuration()
    config['fluorescenceCombined']['export'] = False
    config['brillouin']['export'] = False
    ec.export(config)

    session.clear()

    plots_dir = tmp_dir.parent / 'Plots'
    assert_glob_exists(plots_dir, 'Brightfield_*_FLrep0.png')


def test_export_fluorescence_combined(tmp_dir):
    shutil.copy(
        data_file_path('Fluorescence.h5'), Path.cwd() / 'Fluorescence.h5')

    session = Session.get_instance()
    session.set_file(Path('Fluorescence.h5'))

    ec = ExportController()
    config = ec.get_configuration()
    config['fluorescence']['export'] = False
    config['brillouin']['export'] = False
    ec.export(config)

    session.clear()

    plots_dir = tmp_dir.parent / 'Plots' / 'Bare'
    # Fluorescence.h5 has a scale calibration and no Brillouin
    # repetitions, so only the stage-aligned combination is exported.
    # None of its images carry a brillouin_repetition_index, so each
    # filename gets a capture-timestamp tag instead of a '_BMrepN' one.
    combinations = [
        '__b', '_g_', '_gb', 'r__', 'r_b', 'rg_', 'rgb',
    ]
    for combination in combinations:
        for rep in ('0', '1'):
            assert_glob_exists(
                plots_dir,
                f'fluorescenceCombined_{combination}_*_FLrep{rep}.png')


def test_export_brillouin_2D(tmp_dir):
    shutil.copy(
        data_file_path('2D-xy.h5'), Path.cwd() / '2D-xy.h5')

    session = Session.get_instance()
    session.set_file(Path('2D-xy.h5'))

    ec = ExportController()
    config = ec.get_configuration()
    config['fluorescence']['export'] = False
    config['fluorescenceCombined']['export'] = False
    ec.export(config)

    session.clear()

    # One combined CSV per repetition (every evaluated quantity) is
    # the only Brillouin export output now - no more per-parameter
    # Plots/Bare and Plots/WithAxis image files.
    assert os.path.exists(
        tmp_dir.parent / 'Export' / '2D-xy_BMrep0_data.csv')


def test_export_brillouin_row_order_matches_grid_address(tmp_dir):
    """
    Regression test: rows must come out in each point's fixed x-fastest
    grid address order (as EvaluationController.get_indices_from_key()
    decodes a linear image key, and as BrillouinAcquisition's own
    H5BM::calculateIndex() assigns one) - NOT an arbitrary raster order,
    and NOT necessarily literal chronological capture order either (the
    physical scan path can visit points in any sequence - see
    BrillouinExport._export_combined_csv()'s docstring). A 'C'-order
    ravel over the (dim_x, dim_y,
    dim_z) results arrays (z fastest) previously shuffled the rows
    relative to that address order, which showed up as an
    apparently-random 'time' column.
    """
    shutil.copy(
        data_file_path('2D-xy.h5'), Path.cwd() / '2D-xy.h5')

    session = Session.get_instance()
    session.set_file(Path('2D-xy.h5'))
    session.set_current_repetition('0')
    evm = session.evaluation_model()

    resolution = session.get_payload_resolution()
    n = resolution[0] * resolution[1] * resolution[2]
    shape = (resolution[0], resolution[1], resolution[2], 1, 1, 1)
    time_data = np.full(shape, np.nan)
    intensity_data = np.full(shape, np.nan)
    # Encode each point's own linear image key as its 'time' value, so
    # the exported row order can be checked directly against it.
    for key in range(n):
        ind_x, ind_y, ind_z = \
            EvaluationController.get_indices_from_key(resolution, key)
        time_data[ind_x, ind_y, ind_z, 0, 0, 0] = key
        intensity_data[ind_x, ind_y, ind_z, 0, 0, 0] = 100.0
    evm.results['time'] = time_data
    evm.results['intensity'] = intensity_data

    ec = ExportController()
    config = ec.get_configuration()
    config['fluorescence']['export'] = False
    config['fluorescenceCombined']['export'] = False
    config['surface']['export'] = False
    config['overviewBrightfield']['export'] = False
    ec.export(config)

    session.clear()

    csv_path = tmp_dir.parent / 'Export' / '2D-xy_BMrep0_data.csv'
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(
            row for row in f if not row.startswith('#')))

    assert len(rows) == n
    assert [float(row['time']) for row in rows] == list(range(n))


def test_export_brillouin_rayleigh_shift_column(tmp_dir):
    """
    Regression test: 'rayleigh_shift' is a derived results key not
    registered in EvaluationModel.parameters (unlike every key
    get_parameter_keys() returns), so EvaluationController.get_data()
    cannot look up a unit scaling for it and raises a KeyError. Once a
    file has actually been evaluated (unlike the other fixtures used
    above, whose sessions were never evaluated, so this key stays
    unset and the crash never triggered), the combined CSV export must
    still include it without raising - see
    BrillouinExport._get_parameter_data().
    """
    shutil.copy(
        data_file_path('2D-xy.h5'), Path.cwd() / '2D-xy.h5')

    session = Session.get_instance()
    session.set_file(Path('2D-xy.h5'))
    session.set_current_repetition('0')
    evm = session.evaluation_model()

    resolution = session.get_payload_resolution()
    shape = (resolution[0], resolution[1], resolution[2], 1, 2, 1)
    evm.results['rayleigh_shift'] = np.random.uniform(-1e7, 1e7, size=shape)
    # 'time'/'intensity' mark a point as actually measured (see
    # BrillouinExport._export_combined_csv) - without them every row
    # here would look unmeasured and get dropped.
    general_shape = (resolution[0], resolution[1], resolution[2], 1, 1, 1)
    evm.results['time'] = np.ones(general_shape)
    evm.results['intensity'] = np.full(general_shape, 100.0)

    ec = ExportController()
    config = ec.get_configuration()
    config['fluorescence']['export'] = False
    config['fluorescenceCombined']['export'] = False
    config['surface']['export'] = False
    config['overviewBrightfield']['export'] = False

    ec.export(config)  # Must not raise.

    session.clear()

    csv_path = tmp_dir.parent / 'Export' / '2D-xy_BMrep0_data.csv'
    assert os.path.exists(csv_path)
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(
            row for row in f if not row.startswith('#')))
    assert 'rayleigh_shift' in rows[0]
    assert not np.isnan(float(rows[0]['rayleigh_shift']))


def test_export_brillouin_acquisition_settings_in_csv(tmp_dir):
    """
    Regression test: the acquisition settings actually used for a
    repetition (calibration schedule, per-point-brightfield timing,
    background ROI, ...) must end up in the combined CSV's '#' header,
    not just be readable via Payload.get_acquisition_settings() - see
    BrillouinExport._export_combined_csv(). 2D-xy.h5 predates every one
    of these settings, so this adds a couple directly to a copy of it
    (mimicking what a real BrillouinAcquisition capture writes) to
    exercise that path; a setting this file still doesn't have (e.g.
    con-calibration-used) must simply be absent from the header, not
    written as an empty/None row.
    """
    shutil.copy(
        data_file_path('2D-xy.h5'), Path.cwd() / '2D-xy.h5')

    with h5py.File(Path.cwd() / '2D-xy.h5', 'r+') as f:
        payload = f['Brillouin/0/payload']
        payload.create_dataset(
            'positions-per-point-brightfield-during-acquisition-used',
            data=np.array([1.0]))
        payload.create_dataset(
            'positions-background-roi-mask-used', data=np.array([0.0]))

    session = Session.get_instance()
    session.set_file(Path('2D-xy.h5'))
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

    csv_path = tmp_dir.parent / 'Export' / '2D-xy_BMrep0_data.csv'
    with open(csv_path, newline='') as f:
        header_rows = dict(
            row for row in csv.reader(f) if row and row[0].startswith('#'))

    assert header_rows[
        '#per_point_brightfield_during_acquisition_used'] == '1.0'
    assert header_rows['#background_roi_mask_used'] == '0.0'
    # con-calibration-used was never added to this file - must not
    # appear as an empty/None row.
    assert '#con_calibration_used' not in header_rows


def test_export_brillouin_drops_only_unmeasured_points(tmp_dir):
    """
    A grid point outside the ROI never gets an image key at all, so
    EVERY column - including 'time'/'intensity', set directly from the
    raw spectrum before any peak fit - stays NaN for it; that row must
    be dropped. A point that WAS measured but whose fit failed only has
    NaN in the fitted columns ('time'/'intensity' are still set) and
    must still show up, fit-failure NaNs and all - it must not be
    conflated with an unmeasured point just because every OTHER column
    happens to be NaN too.
    """
    shutil.copy(
        data_file_path('2D-xy.h5'), Path.cwd() / '2D-xy.h5')

    session = Session.get_instance()
    session.set_file(Path('2D-xy.h5'))
    session.set_current_repetition('0')
    evm = session.evaluation_model()

    resolution = session.get_payload_resolution()  # (3, 5, 1)
    general_shape = (resolution[0], resolution[1], resolution[2], 1, 1, 1)
    fit_shape = (resolution[0], resolution[1], resolution[2], 1, 2, 1)

    time_data = np.full(general_shape, np.nan)
    intensity_data = np.full(general_shape, np.nan)
    shift_data = np.full(fit_shape, np.nan)

    # y-index 2: measured, fit succeeded. y-index 3: measured, fit
    # failed. Every other y-index: outside the ROI, never measured.
    time_data[:, 2, :, :, :, :] = 1.0
    intensity_data[:, 2, :, :, :, :] = 100.0
    shift_data[:, 2, :, :, :, :] = 5.0e9
    time_data[:, 3, :, :, :, :] = 1.0
    intensity_data[:, 3, :, :, :, :] = 100.0

    evm.results['time'] = time_data
    evm.results['intensity'] = intensity_data
    evm.results['brillouin_shift_f'] = shift_data

    ec = ExportController()
    config = ec.get_configuration()
    config['fluorescence']['export'] = False
    config['fluorescenceCombined']['export'] = False
    config['surface']['export'] = False
    config['overviewBrightfield']['export'] = False
    ec.export(config)

    session.clear()

    csv_path = tmp_dir.parent / 'Export' / '2D-xy_BMrep0_data.csv'
    with open(csv_path, newline='') as f:
        rows = list(csv.DictReader(
            row for row in f if not row.startswith('#')))

    # 3 x-values x 2 measured y-values (y=0.0 fit ok, y=1.0 fit failed;
    # 2D-xy.h5's y positions are centered so raw index 2 -> 0.0, 3 -> 1.0)
    # - the 3 unmeasured y-indices (0, 1, 4) contribute no rows.
    assert len(rows) == 6
    by_y = {round(float(row['y']), 1): row for row in rows}
    assert set(by_y) == {0.0, 1.0}
    assert not np.isnan(float(by_y[0.0]['brillouin_shift_f']))
    assert np.isnan(float(by_y[1.0]['brillouin_shift_f']))
    assert not np.isnan(float(by_y[1.0]['time']))


def test_export_brillouin_repetition_filter(tmp_dir):
    """
    config['brillouin']['repetitions'], when not None, restricts export
    to the listed repetition keys - used by BMicro's export dialog to
    let a user pick which repetitions to export.
    """
    shutil.copy(
        data_file_path('2D-xy.h5'), Path.cwd() / '2D-xy.h5')

    session = Session.get_instance()
    session.set_file(Path('2D-xy.h5'))

    ec = ExportController()
    config = ec.get_configuration()
    config['fluorescence']['export'] = False
    config['fluorescenceCombined']['export'] = False
    config['surface']['export'] = False
    config['overviewBrightfield']['export'] = False
    # 2D-xy.h5 only has repetition '0' - excluding it should suppress
    # the export entirely, proving the filter actually took effect.
    config['brillouin']['repetitions'] = []
    ec.export(config)

    session.clear()

    assert not os.path.exists(tmp_dir.parent / 'Export')
    assert not os.path.exists(tmp_dir.parent / 'Plots')


def test_export_repetition_filter_also_restricts_other_exporters(tmp_dir):
    """
    Regression test: config['brillouin']['repetitions'] must restrict
    every exporter, not just the combined CSV (BrillouinExport) -
    SurfaceExport, OverviewBrightfieldExport, FluorescenceExport and
    FluorescenceCombinedExport previously ignored it entirely and
    always exported every repetition's data regardless of the
    selection.
    """
    shutil.copy(
        data_file_path('SurfaceScan.h5'), Path.cwd() / 'SurfaceScan.h5')

    # SurfaceScan.h5 predates brillouin_repetition_index - add it to its
    # overview images directly (mimicking what a real BrillouinAcquisition
    # capture now writes), same as test_export_surface_and_overview_
    # brightfield below.
    with h5py.File(Path.cwd() / 'SurfaceScan.h5', 'r+') as f:
        for idx in range(4):
            ds = f[f'Fluorescence/0/payload/data/{idx}']
            ds.attrs['brillouin_repetition_index'] = np.array([0])

    session = Session.get_instance()
    session.set_file(Path('SurfaceScan.h5'))

    ec = ExportController()
    config = ec.get_configuration()
    config['fluorescenceCombined']['export'] = False
    config['brillouin']['export'] = False
    # SurfaceScan.h5 only has Brillouin repetition '0' - excluding it
    # should suppress the surface/overview-brightfield exports too, not
    # just the (already disabled) combined CSV.
    config['brillouin']['repetitions'] = []
    ec.export(config)

    session.clear()

    plots_dir = tmp_dir.parent / 'Plots'
    assert not os.path.exists(plots_dir / 'surface_BMrep0_z_surface.png')
    assert not os.path.exists(
        plots_dir / 'overviewZStack_0_BMrep0_tiled.tif')
    assert not os.path.exists(
        tmp_dir.parent / 'Export' / 'surface_BMrep0_metrics.json')


def test_export_surface_skipped_when_neither_follow_nor_roi_used(tmp_dir):
    """
    Regression test: BrillouinAcquisition writes the surface-scan
    dataset group unconditionally on every payload since H5BM-v0.0.4,
    whether or not surface following or an ROI restriction was
    actually used for that particular repetition - has_surface_scan()
    (checked by SurfaceExport via get_surface_scan_data()) only
    reflects the file format version, not this run's settings. A run
    that used neither must not produce a surface export at all.
    """
    shutil.copy(
        data_file_path('SurfaceScan.h5'), Path.cwd() / 'SurfaceScan.h5')

    with h5py.File(Path.cwd() / 'SurfaceScan.h5', 'r+') as f:
        g = f['Brillouin/0/payload']
        g['positions-surface-follow-used'][()] = 0
        g['positions-roi-mask-used'][()] = 0

    session = Session.get_instance()
    session.set_file(Path('SurfaceScan.h5'))

    ec = ExportController()
    config = ec.get_configuration()
    config['fluorescence']['export'] = False
    config['fluorescenceCombined']['export'] = False
    config['brillouin']['export'] = False
    config['overviewBrightfield']['export'] = False
    ec.export(config)

    session.clear()

    assert not os.path.exists(tmp_dir.parent / 'Plots')
    assert not os.path.exists(tmp_dir.parent / 'Export')


def test_export_surface_and_overview_brightfield(tmp_dir):
    shutil.copy(
        data_file_path('SurfaceScan.h5'), Path.cwd() / 'SurfaceScan.h5')

    # SurfaceScan.h5 predates brillouin_repetition_index (see
    # MeasurementData.get_brillouin_repetition_index()) - add it to its
    # overview images directly (mimicking what a real BrillouinAcquisition
    # capture now writes) so the export exercises the real, current
    # '_BMrepN' naming instead of falling back to a timestamp tag.
    with h5py.File(Path.cwd() / 'SurfaceScan.h5', 'r+') as f:
        for idx in range(4):
            ds = f[f'Fluorescence/0/payload/data/{idx}']
            ds.attrs['brillouin_repetition_index'] = np.array([0])

    session = Session.get_instance()
    session.set_file(Path('SurfaceScan.h5'))
    session.set_current_repetition('0')

    ec = ExportController()
    config = ec.get_configuration()
    config['fluorescence']['export'] = False
    config['fluorescenceCombined']['export'] = False
    config['brillouin']['export'] = False
    ec.export(config)

    session.clear()

    plots_dir = tmp_dir.parent / 'Plots'
    images = [
        'surface_BMrep0_z_surface.png',
        'surface_BMrep0_3d.png',
        'surface_BMrep0_roi_plan_mask.png',
        # SurfaceScan.h5's overview images belong to Brillouin
        # repetition 0 (see the injected brillouin_repetition_index
        # above), with 2 z-planes x 2 tiles each (point_count=2,
        # point_stack_counts=[1, 1]) - no FLrep in the name (only one
        # overview stack is ever written per Brillouin repetition), one
        # file per z-plane since each plane's own tiles don't belong in
        # one combined stack with another plane's, and each is a tiled
        # mosaic since it has more than one distinct tile position.
        'overviewZStack_0_BMrep0_tiled.tif',
        'overviewZStack_1_BMrep0_tiled.tif',
    ]
    for image in images:
        assert os.path.exists(plots_dir / image)

    # Each tiled mosaic gets a matching matrix mapping an absolute
    # stage position (x, y, um) onto its own pixel coordinates.
    transforms = [
        'overviewZStack_0_BMrep0_tiled_transform.csv',
        'overviewZStack_1_BMrep0_tiled_transform.csv',
    ]
    matrices_dir = plots_dir / 'TransformMatrices'
    for transform in transforms:
        assert os.path.exists(matrices_dir / transform)
        with open(matrices_dir / transform) as f:
            rows = [line.strip().split(',') for line in f if line.strip()]
        assert len(rows) == 3
        assert all(len(row) == 3 for row in rows)

    # SurfaceScan.h5 predates the coarse pre-scan point datasets
    # (positions-surface-prescan-x/y-um etc.) - no plot should be
    # produced for data that was never saved.
    assert not os.path.exists(
        plots_dir / 'surface_BMrep0_prescan_points.png')

    # SurfaceScan.h5 has no drawn ROI polygon saved - only its
    # grid-rasterized 'roi_scan_plan_mask' - so no polygon plot is
    # produced for it.
    assert not os.path.exists(
        plots_dir / 'surface_BMrep0_roi_polygon.png')

    export_dir = tmp_dir.parent / 'Export'
    metrics_file = export_dir / 'surface_BMrep0_metrics.json'
    assert os.path.exists(metrics_file)

    with open(metrics_file) as f:
        metrics = json.load(f)
    assert metrics['surface_found_fraction'] == pytest.approx(0.5)
    assert metrics['surface_interpolated_fraction'] == pytest.approx(1 / 3)
    assert metrics['surface_missing_fraction'] == pytest.approx(1 / 6)
    assert metrics['surface_used_fraction'] == pytest.approx(0.5 + 1 / 3)
    # Restricted to the drawn ROI (5 of the 6 grid points), the point
    # with no surface info at all happens to sit outside the ROI, so
    # roi_surface_missing_fraction is 0 even though the unrestricted
    # surface_missing_fraction above is 1/6.
    assert metrics['roi_surface_found_fraction'] == pytest.approx(0.6)
    assert metrics['roi_surface_interpolated_fraction'] == pytest.approx(0.4)
    assert metrics['roi_surface_missing_fraction'] == pytest.approx(0.0)
    assert metrics['roi_surface_used_fraction'] == pytest.approx(1.0)
    assert metrics['roi_coverage_fraction'] == pytest.approx(0.8)
    assert metrics['surface_follow_used'] == 1
    assert metrics['surface_z_offset_um_used'] == pytest.approx(2.5)


def test_export_fluorescence_with_aborted_brillouin_repetition(tmp_dir):
    """
    Regression test: exports that relate a Fluorescence repetition to
    a Brillouin one (via bmlab.export.timing.get_brillouin_windows)
    must not crash when a Brillouin repetition is aborted/restarted.
    SurfaceScan.h5's Brillouin repetition '1' is exactly that case
    (resolution attributes set, but no positions-x/y/z datasets or
    images at all) - it must be skipped gracefully instead of raising.
    """
    shutil.copy(
        data_file_path('SurfaceScan.h5'), Path.cwd() / 'SurfaceScan.h5')

    session = Session.get_instance()
    session.set_file(Path('SurfaceScan.h5'))
    session.set_current_repetition('0')

    ec = ExportController()
    config = ec.get_configuration()
    config['brillouin']['export'] = False
    config['surface']['export'] = False
    # Must not raise
    ec.export(config)

    session.clear()

    plots_dir = tmp_dir.parent / 'Plots'
    assert os.path.exists(plots_dir)
    assert len(list(plots_dir.iterdir())) > 0


def test_export_surface_disabled(tmp_dir):
    shutil.copy(
        data_file_path('SurfaceScan.h5'), Path.cwd() / 'SurfaceScan.h5')

    session = Session.get_instance()
    session.set_file(Path('SurfaceScan.h5'))
    session.set_current_repetition('0')

    ec = ExportController()
    config = ec.get_configuration()
    config['fluorescence']['export'] = False
    config['fluorescenceCombined']['export'] = False
    config['brillouin']['export'] = False
    config['surface']['export'] = False
    config['overviewBrightfield']['export'] = False
    ec.export(config)

    session.clear()

    plots_dir = tmp_dir.parent / 'Plots'
    assert not os.path.exists(plots_dir)


def test_export_brillouin_3D(tmp_dir):
    shutil.copy(
        data_file_path('3D.h5'), Path.cwd() / '3D.h5')

    session = Session.get_instance()
    session.set_file(Path('3D.h5'))

    ec = ExportController()
    config = ec.get_configuration()
    config['fluorescence']['export'] = False
    config['fluorescenceCombined']['export'] = False
    ec.export(config)

    session.clear()

    # One combined CSV per repetition (every evaluated quantity) is
    # the only Brillouin export output now - no more per-parameter,
    # per-slice Plots/Bare and Plots/WithAxis image files.
    assert os.path.exists(
        tmp_dir.parent / 'Export' / '3D_BMrep0_data.csv')
