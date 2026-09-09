from pathlib import Path
import json
import pytest
import shutil
import uuid
import os

from bmlab.session import Session
from bmlab.controllers import ExportController


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
    # pixel-space raw image, no before/during/after tag).
    images = [
        'Blue_FLrep0.png',
        'Brightfield_FLrep0.png',
        'Green_FLrep0.png',
        'Red_FLrep0.png',
        'Blue_FLrep1.png',
        'Brightfield_FLrep1.png',
        'Green_FLrep1.png',
        'Red_FLrep1.png',
    ]
    for image in images:
        assert os.path.exists(plots_dir / image)


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
    images = [
        'Brightfield_FLrep0.png',
    ]
    for image in images:
        assert os.path.exists(plots_dir / image)


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
    images = [
        'Brightfield_FLrep0.png',
    ]
    for image in images:
        assert os.path.exists(plots_dir / image)


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
    images = [
        'fluorescenceCombined___b_FLrep0.png',
        'fluorescenceCombined__g__FLrep0.png',
        'fluorescenceCombined__gb_FLrep0.png',
        'fluorescenceCombined_r___FLrep0.png',
        'fluorescenceCombined_r_b_FLrep0.png',
        'fluorescenceCombined_rg__FLrep0.png',
        'fluorescenceCombined_rgb_FLrep0.png',
        'fluorescenceCombined___b_FLrep1.png',
        'fluorescenceCombined__g__FLrep1.png',
        'fluorescenceCombined__gb_FLrep1.png',
        'fluorescenceCombined_r___FLrep1.png',
        'fluorescenceCombined_r_b_FLrep1.png',
        'fluorescenceCombined_rg__FLrep1.png',
        'fluorescenceCombined_rgb_FLrep1.png',
    ]
    for image in images:
        assert os.path.exists(plots_dir / image)


def test_export_brillouin_2D(tmp_dir):
    shutil.copy(
        data_file_path('2D-xy.h5'), Path.cwd() / '2D-xy.h5')

    session = Session.get_instance()
    session.set_file(Path('2D-xy.h5'))

    ec = ExportController()
    config = ec.get_configuration()
    config['fluorescence']['export'] = False
    config['fluorescenceCombined']['export'] = False
    config['brillouin']['parameters'] =\
        ['brillouin_shift_f', 'brillouin_peak_intensity']
    ec.export(config)

    session.clear()

    plots_dir = tmp_dir.parent / 'Plots' / 'Bare'
    images = [
        '2D-xy_BMrep0_brillouin_shift_f.png',
        '2D-xy_BMrep0_brillouin_shift_f.tiff',
        '2D-xy_BMrep0_brillouin_peak_intensity.png',
        '2D-xy_BMrep0_brillouin_peak_intensity.tiff',
    ]
    for image in images:
        assert os.path.exists(plots_dir / image)

    plots_dir = tmp_dir.parent / 'Plots' / 'WithAxis'
    images = [
        '2D-xy_BMrep0_brillouin_shift_f.pdf',
        '2D-xy_BMrep0_brillouin_shift_f.png',
        '2D-xy_BMrep0_brillouin_peak_intensity.pdf',
        '2D-xy_BMrep0_brillouin_peak_intensity.png',
    ]
    for image in images:
        assert os.path.exists(plots_dir / image)

    csvs = [
        '2D-xy_BMrep0_brillouin_shift_f.csv',
        '2D-xy_BMrep0_brillouin_peak_intensity.csv',
    ]
    for csv in csvs:
        assert os.path.exists(
            tmp_dir.parent / 'Export' / csv)


def test_export_surface_and_overview_brightfield(tmp_dir):
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
    ec.export(config)

    session.clear()

    plots_dir = tmp_dir.parent / 'Plots'
    images = [
        'surface_BMrep0_z_surface.png',
        'surface_BMrep0_3d.png',
        'surface_BMrep0_roi_plan_mask.png',
        # SurfaceScan.h5's overview images were captured after
        # Brillouin repetition 0 finished, with 2 z-planes x 2 tiles
        # each (point_count=2, point_stack_counts=[1, 1]) - no FLrep
        # in the name (only one overview stack is ever written per
        # Brillouin repetition), one file per z-plane since each
        # plane's own tiles don't belong in one combined stack with
        # another plane's, and each is a tiled mosaic since it has
        # more than one distinct tile position.
        'overviewZStack_0_BMrep0_afterAcq_tiled.tif',
        'overviewZStack_1_BMrep0_afterAcq_tiled.tif',
    ]
    for image in images:
        assert os.path.exists(plots_dir / image)

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
    config['brillouin']['parameters'] =\
        ['brillouin_shift_f', 'brillouin_peak_intensity']
    ec.export(config)

    session.clear()

    slice_count = 3

    plots_dir = tmp_dir.parent / 'Plots' / 'Bare'
    images = [
        '3D_BMrep0_brillouin_shift_f',
        '3D_BMrep0_brillouin_peak_intensity',
    ]
    for image in images:
        # Check that slices are exported as separate PNGs
        for slice_number in range(slice_count):
            assert os.path.exists(
                plots_dir / f'{image}_slice-{slice_number}.png')
        # Check that slices are exported as single stacked TIFF
        assert os.path.exists(
            plots_dir / f'{image}.tiff')

    plots_dir = tmp_dir.parent / 'Plots' / 'WithAxis'
    images = [
        '3D_BMrep0_brillouin_shift_f',
        '3D_BMrep0_brillouin_peak_intensity',
    ]
    file_types = ['pdf', 'png']
    for image in images:
        for file_type in file_types:
            for slice_number in range(slice_count):
                assert os.path.exists(
                    plots_dir / f'{image}_slice-{slice_number}.{file_type}')

    csvs = [
        '3D_BMrep0_brillouin_shift_f',
        '3D_BMrep0_brillouin_peak_intensity',
    ]
    for csv in csvs:
        for slice_number in range(slice_count):
            assert os.path.exists(
                tmp_dir.parent / 'Export' / f'{csv}_slice-{slice_number}.csv')
