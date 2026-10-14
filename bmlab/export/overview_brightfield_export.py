import csv
import logging
import os

import numpy as np

from bmlab import Session
from bmlab.file import OVERVIEW_CHANNELS, FLUORESCENCE_GROUP
from bmlab.export.naming import timestamp_tag, folder_name, \
    running_brillouin_repetition
from bmlab.export.selection import repetition_selected
from bmlab.export.alignment import get_tmatrix, orient_image, \
    get_point_to_pixel_matrix, write_transform_csv, \
    calibration_stage_to_pixel
from bmlab.export.tiffio import write_tiff

logger = logging.getLogger(__name__)

# Two tile/z positions closer than this (um) are treated as "the same
# point" - generous enough to absorb float noise and hysteresis-
# compensation jitter, tight enough to stay well under real tile
# spacing (camera-FOV sized, so at least tens of um).
_POSITION_TOLERANCE_UM = 1.0


class OverviewBrightfieldExport(object):
    """
    Exports the brightfield overview images BrillouinAcquisition takes
    during a Brillouin run (the overview engine, channels OVERVIEW_CHANNELS),
    as TIFF with the pixel type kept.

    Layout, below the experiment's 'Plots' folder: one folder
    Overview_BMrep<N> per Brillouin repetition, holding
    overview_<n>_<YYYYMMDD-HHMMSS>.tif - one file per overview event and
    objective, in event order, n counting the files of that repetition,
    the time stamp being that of the first image. A z stack at one
    position is a multi-page TIFF; several positions (tiles) give the
    merged mosaic, as a multi-page TIFF when each position was imaged at
    several z, and with 'export tiles' the unstitched tiles in a folder
    overview_<n>_<time>_tiles.

    Every image carries its own scale calibration (the repetition's one is
    that of the measuring objective, the overview is usually taken with
    another) and its own stage position. The transform matrices map the
    CSV x, y of the Brillouin export (planned grid positions, i.e.
    already free of the drift the acquisition corrected) onto the pixels
    of that file. An overview_events.csv next to them lists what each
    file is.

    Files written by BrillouinAcquisition versions before the overview
    engine have to be converted once with tools/migrate_legacy_overviews.py
    (their images lack the attributes this export needs; they are
    reported and left out).

    The images the Fluorescence tab takes by hand are exported by
    FluorescenceExport.
    """

    def __init__(self):
        self.session = Session.get_instance()
        self.file = self.session.file
        self.mode = FLUORESCENCE_GROUP
        self.export_merged = True
        self.export_tiles = False
        return

    def export(self, configuration):
        if not self.file:
            return

        config = configuration.get('overviewBrightfield')
        if not config or not (config.get('export')
                              or config.get('exportTiles')):
            return
        # 'export': the merged overviews, 'exportTiles': the unstitched
        # tiles of the tiled ones - either can be chosen without the other.
        self.export_merged = bool(config.get('export'))
        self.export_tiles = bool(config.get('exportTiles'))

        self._export_overview_events(configuration, self._plot_path())

    def _brillouin_repetition_of(self, payload, image_key, info):
        """
        The Brillouin repetition an image of the overview engine was taken
        during. The engine writes the key the repetition had in the file
        then; that is used as long as the repetition is still there. An
        aborted start that was removed from the file afterwards leaves
        its images behind: they belong to the repetition running at that
        time, or to the next one (running_brillouin_repetition()).
        """
        keys = [str(key) for key in self.file.repetition_keys()]
        recorded = info['repetition']
        if recorded is not None and str(recorded) in keys:
            return str(recorded)
        return running_brillouin_repetition(
            self.file, payload.get_date(image_key), None)

    def _export_overview_events(self, configuration, path):
        """
        Exports the overviews of the overview engine (channels
        OVERVIEW_CHANNELS): per Brillouin repetition, event and
        objective one file Overview_BMrep<N>/overview_<n>_<time>.tif, n
        counting in event order. Every image carries its own scale
        calibration (the repetition's one is that of the measuring
        objective, the overview is usually taken with another) and its
        own stage position; the transform matrix maps the CSV x, y onto
        the pixels of the file, see MeasurementData.get_overview_info()
        ('anchor') for what the image is centred on.

        Also writes overview_events.csv into each folder, one line per
        file, with the drift correction's decision for that event.
        """
        members = {}
        without_attributes = 0
        for repetition_key in self.file.repetition_keys(self.mode):
            repetition = self.file.get_repetition(repetition_key, self.mode)
            payload = repetition.payload
            for channel in OVERVIEW_CHANNELS:
                keys = payload.image_keys_by_channel(channel)
                for image_key in keys:
                    info = payload.get_overview_info(image_key)
                    if info is None:
                        without_attributes += 1
                        continue
                    brillouin_key = self._brillouin_repetition_of(
                        payload, image_key, info)
                    members.setdefault(
                        (brillouin_key, info['event'],
                         info['objective_slot']),
                        []).append((repetition, payload, image_key, info))
        if without_attributes:
            logger.warning(
                '%d overview images have no overview-engine attributes '
                '(a file of an older BrillouinAcquisition version) and are '
                'not exported: convert the file with '
                'tools/migrate_legacy_overviews.py', without_attributes)
        if not members:
            return

        def order(group_key):
            brillouin_key, event, slot = group_key
            number = int(brillouin_key) \
                if brillouin_key is not None and str(brillouin_key).isdigit() \
                else -1
            return brillouin_key is None, number, event, slot

        counters = {}
        rows = {}
        for group_key in sorted(members, key=order):
            brillouin_key, event, slot = group_key
            if brillouin_key is not None and not repetition_selected(
                    configuration, brillouin_key):
                continue
            group = sorted(
                members[group_key],
                key=lambda m: (m[3]['tile_index'], m[3]['stack_index']))
            repetition, payload = group[0][0], group[0][1]
            image_keys = [m[2] for m in group]
            infos = [m[3] for m in group]

            folder = path / folder_name('Overview', brillouin_key)
            counters[folder] = counters.get(folder, 0) + 1
            os.makedirs(folder, exist_ok=True)
            stem = f"overview_{counters[folder]}_" \
                f"{timestamp_tag(payload.get_date(image_keys[0]))}"

            scale_calibration = infos[0]['scale_calibration'] \
                or payload.get_scale_calibration()
            tmatrix = get_tmatrix(scale_calibration)
            positions = self._tile_positions(infos)
            self._export_group(
                repetition, image_keys, positions, tmatrix,
                scale_calibration, folder, stem)

            rows.setdefault(folder, []).append(self._event_row(
                f"{stem}.tif" if tmatrix is not None
                else f"{stem}_cameraPixels.tif",
                payload, image_keys, infos))

        for folder, folder_rows in rows.items():
            self._write_events_csv(folder / 'overview_events.csv', folder_rows)

    @staticmethod
    def _tile_positions(infos):
        """
        The position of every image of an overview as _export_group()
        takes it: the anchor of its tile (x, y: the median of the tile's
        z stack - the stage reads back a few tenths of um differently
        from slice to slice, which must not make one tile look like
        several), z of the slice itself. The engine records which tile
        and slice an image is, so the tiles are not guessed from their
        positions here.
        """
        by_tile = {}
        for info in infos:
            if info['anchor'] is not None:
                by_tile.setdefault(info['tile_index'], []).append(
                    (info['anchor']['x'], info['anchor']['y']))
        centre = {
            tile: (float(np.median([p[0] for p in points])),
                   float(np.median([p[1] for p in points])))
            for tile, points in by_tile.items()}
        return [
            None if info['anchor'] is None else {
                'x': centre[info['tile_index']][0],
                'y': centre[info['tile_index']][1],
                'z': info['anchor']['z']}
            for info in infos]

    @staticmethod
    def _event_row(filename, payload, image_keys, infos):
        first = infos[0]
        date = payload.get_date(image_keys[0])
        anchor = first['anchor'] or {}
        return {
            'file': filename,
            'event': first['event'],
            'kind': first['kind_name'],
            'z_index': first['z_index'],
            'point_index': first['point_index'],
            'seconds_since_start': first['seconds_since_start'],
            'time': date.isoformat() if date else '',
            'objective_slot': first['objective_slot'],
            'objective': first['objective_name'],
            'is_measurement_objective': int(first['is_measurement_objective']),
            'images': len(image_keys),
            'tiles': first['tile_count'],
            'stack_depth': first['stack_count'],
            'anchor_x_um': anchor.get('x', ''),
            'anchor_y_um': anchor.get('y', ''),
            'grid_offset_x_um': first['grid_offset'][0],
            'grid_offset_y_um': first['grid_offset'][1],
        }

    @staticmethod
    def _write_events_csv(filename, rows):
        columns = list(rows[0].keys())
        with open(filename, 'w', newline='') as csvfile:
            csvfile.write(
                '#One line per overview file. anchor_*: CSV x, y (planned '
                'grid position) at the middle of the image; grid_offset_*: '
                'drift offset of the grid when the image was taken (what '
                'the drift correction decided at each event is in '
                'Export/*_runtime_drift.csv)\n')
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def _export_group(
            self, repetition, image_keys, positions, tmatrix,
            scale_calibration, folder, stem):
        """
        Exports one group of overview images - the whole repetition's
        worth (one image per outer z-plane) or a single outer z-plane's
        own local sub-stack/tiles - as one merged mosaic if it contains
        more than one distinct tile position, or as the plain (aligned,
        if possible) image or stack otherwise. `positions` is a list of
        {'x', 'y', 'z'} dicts (or None entries), one per image_keys
        entry.
        """
        tiles = None
        if tmatrix is not None and all(p is not None for p in positions):
            tiles = self._group_by_tile(positions)

        if tmatrix is not None and tiles is not None and len(tiles) > 1:
            if self.export_merged:
                self._export_tiled(
                    repetition, image_keys, positions, tiles, tmatrix,
                    scale_calibration, folder / f"{stem}.tif")
            if self.export_tiles:
                self._export_single_tiles(
                    repetition, image_keys, positions, tiles, tmatrix,
                    scale_calibration, folder / f"{stem}_tiles")
        else:
            suffix = '' if tmatrix is not None else '_cameraPixels'
            self._export_stack(
                repetition, image_keys, positions, tmatrix,
                scale_calibration, folder / f"{stem}{suffix}.tif")

    @staticmethod
    def _group_by_tile(positions):
        """
        Groups image indices by (x, y) tile position (within
        _POSITION_TOLERANCE_UM), returning a dict mapping each unique
        (x, y) to the list of image indices captured there.
        """
        tiles = {}
        for idx, pos in enumerate(positions):
            match = next(
                (xy for xy in tiles
                 if abs(xy[0] - pos['x']) < _POSITION_TOLERANCE_UM
                 and abs(xy[1] - pos['y']) < _POSITION_TOLERANCE_UM),
                None)
            key = match if match is not None else (pos['x'], pos['y'])
            tiles.setdefault(key, []).append(idx)
        return tiles

    @staticmethod
    def _read(repetition, image_key):
        """The image as a 2D array, pixel type kept (several frames are
        averaged); None if there is no data."""
        img_data = repetition.payload.get_image(image_key)
        if img_data is None:
            return None
        if img_data.shape[0] == 1:
            return np.asarray(img_data[0])
        mean = np.nanmean(img_data, axis=0)
        if np.issubdtype(img_data.dtype, np.integer):
            mean = np.rint(mean)
        return mean.astype(img_data.dtype)

    @staticmethod
    def _as_pixel_type(image, dtype):
        """Back to the camera's pixel type; NaN (outside the covered
        area) becomes 0."""
        image = np.nan_to_num(image, nan=0.0)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            image = np.clip(np.rint(image), info.min, info.max)
        return image.astype(dtype)

    def _export_stack(
            self, repetition, image_keys, positions, tmatrix,
            scale_calibration, filename):
        """
        A single-xy (or unaligned) image or z-stack: every frame gets the
        same orientation (they share the same footprint, only z
        differs), so the pages come out consistently sized without any
        extra placement logic - or, with no scale calibration, the frames
        are saved exactly as captured. `positions` is the same list
        `_export_group` received - since every frame shares one
        footprint, the first one with a recorded position anchors the
        transform matrix written alongside the image (see
        get_point_to_pixel_matrix()).
        """
        frames = []
        anchor_um, anchor_shape, anchor_translate = None, None, None
        for image_key, position in zip(image_keys, positions):
            img_data = self._read(repetition, image_key)
            if img_data is None:
                continue
            if tmatrix is not None:
                oriented, translate = orient_image(img_data, tmatrix)
                frame = self._as_pixel_type(oriented, img_data.dtype)
                if anchor_um is None and position is not None:
                    anchor_um = (position['x'], position['y'])
                    anchor_shape = img_data.shape
                    anchor_translate = translate
            else:
                frame = img_data
            frames.append(frame)

        if not frames:
            return

        write_tiff(filename, frames[0] if len(frames) == 1 else frames)

        stage_map = calibration_stage_to_pixel(scale_calibration) \
            if scale_calibration is not None else None
        matrix = get_point_to_pixel_matrix(
            tmatrix, stage_map, anchor_um, anchor_shape,
            anchor_translate)
        write_transform_csv(matrix, filename)

    def _export_tiled(
            self, repetition, image_keys, positions, tiles, tmatrix,
            scale_calibration, filename):
        """
        One merged mosaic per distinct z value found among `image_keys`
        (the pages of the file).

        The tiles are first placed side by side in the RAW camera frame -
        where a stage move is just a shift of the image, whatever the
        camera orientation is - and only the finished mosaic is rotated
        once (orient_image()) and saved. Each tile's shift follows from
        its stage position and the scale calibration alone
        (calibration_stage_to_pixel()); the tiles are not matched against
        each other.
        """
        reference_xy = next(iter(tiles))

        z_by_index = [pos['z'] for pos in positions]
        unique_z = []
        for z in z_by_index:
            if not any(abs(z - uz) < _POSITION_TOLERANCE_UM
                       for uz in unique_z):
                unique_z.append(z)
        unique_z.sort()

        def tile_index(tile_xy, z):
            return next(
                (i for i in tiles[tile_xy]
                 if abs(z_by_index[i] - z) < _POSITION_TOLERANCE_UM),
                None)

        pixel_type = []

        def raw_tile(index):
            img_data = self._read(repetition, image_keys[index])
            if img_data is None:
                return None
            if not pixel_type:
                pixel_type.append(img_data.dtype)
            return np.nan_to_num(img_data.astype(float))

        stage_map = calibration_stage_to_pixel(scale_calibration)

        # Stage offset -> raw pixel offset (d_col, d_row) of every tile
        # relative to the reference tile, and the shared canvas.
        offsets = {}
        for tile_xy in tiles:
            delta = np.array([tile_xy[0] - reference_xy[0],
                              tile_xy[1] - reference_xy[1]])
            offsets[tile_xy] = stage_map @ delta
        min_col = min(o[0] for o in offsets.values())
        min_row = min(o[1] for o in offsets.values())

        planes = []
        placement = None
        for z in unique_z:
            placed = []
            for tile_xy in tiles:
                index = tile_index(tile_xy, z)
                if index is None:
                    continue
                tile = raw_tile(index)
                if tile is None:
                    continue
                col = int(round(offsets[tile_xy][0] - min_col))
                row = int(round(offsets[tile_xy][1] - min_row))
                placed.append((tile, row, col))
                if placement is None and tile_xy == reference_xy:
                    placement = (tile.shape, row, col)
            if placed:
                planes.append(placed)
        # One canvas for all planes: an overview stopped part-way has
        # planes with fewer tiles, whose pages would be smaller.
        shape = (max(row + tile.shape[0] for placed in planes
                     for tile, row, _ in placed),
                 max(col + tile.shape[1] for placed in planes
                     for tile, _, col in placed)) if planes else None
        pages = [self._compose_raw_mosaic(placed, shape) for placed in planes]

        if not pages:
            return

        oriented_pages = []
        translate = None
        for page in pages:
            oriented, translate = orient_image(page, tmatrix)
            oriented_pages.append(
                self._as_pixel_type(oriented, pixel_type[0]))
        write_tiff(
            filename,
            oriented_pages[0] if len(oriented_pages) == 1
            else oriented_pages)

        matrix = None
        if placement is not None:
            (anchor_shape, row, col) = placement
            # stage position -> RAW mosaic pixel (col, row): the
            # reference tile's centre sits at the reference stage
            # position, everything else follows from the stage map.
            centre = np.array([col + (anchor_shape[1] - 1) / 2,
                               row + (anchor_shape[0] - 1) / 2])
            raw_matrix = np.eye(3)
            raw_matrix[:2, :2] = stage_map
            raw_matrix[:2, 2] = centre - stage_map @ np.array(reference_xy)
            matrix = np.asarray(translate * tmatrix) @ raw_matrix
        write_transform_csv(matrix, filename)

    def _export_single_tiles(
            self, repetition, image_keys, positions, tiles, tmatrix,
            scale_calibration, folder):
        """
        The tiles of a mosaic as they were captured, one TIFF per tile
        position (a multi-page TIFF if the tile was imaged at several
        z), in folder `folder` (<mosaic name>_tiles). Every tile is only
        turned/mirrored like the mosaic (orient_image()), nothing is
        stitched or blended, and gets its own transform matrix (stage
        position in um -> pixel in that tile file, same convention as
        the mosaic's), so the tiles can be registered one by one, e.g.
        to measure sample drift between two overviews.
        """
        stage_map = calibration_stage_to_pixel(scale_calibration)
        os.makedirs(folder, exist_ok=True)
        for number, (tile_xy, indices) in enumerate(tiles.items(), 1):
            indices = sorted(indices, key=lambda i: positions[i]['z'])
            raw = [self._read(repetition, image_keys[i]) for i in indices]
            raw = [r for r in raw if r is not None]
            if not raw:
                continue
            pages, translate = [], None
            for frame in raw:
                oriented, translate = orient_image(frame, tmatrix)
                pages.append(oriented)
            filename = folder / f"tile_{number}.tif"
            write_tiff(filename, pages[0] if len(pages) == 1 else pages)

            height, width = raw[0].shape[0], raw[0].shape[1]
            centre = np.array([(width - 1) / 2, (height - 1) / 2])
            raw_matrix = np.eye(3)
            raw_matrix[:2, :2] = stage_map
            raw_matrix[:2, 2] = centre - stage_map @ np.array(tile_xy)
            write_transform_csv(
                np.asarray(translate * tmatrix) @ raw_matrix, filename)

    @staticmethod
    def _compose_raw_mosaic(placed, shape=None):
        if shape is None:
            shape = (max(row + tile.shape[0] for tile, row, _ in placed),
                     max(col + tile.shape[1] for tile, _, col in placed))
        canvas = np.full(shape, np.nan)
        for tile, row, col in placed:
            region = canvas[row:row + tile.shape[0], col:col + tile.shape[1]]
            # First tile to reach a pixel wins - simple, deterministic,
            # avoids double-brightening the tiles' overlap margin.
            empty = np.isnan(region)
            region[empty] = tile[empty]
        return canvas

    def _plot_path(self):
        if self.file.path.parent.name == 'RawData':
            path = self.file.path.parents[1] / 'Plots'
        else:
            path = self.file.path.parent
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path
