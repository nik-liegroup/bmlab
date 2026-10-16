import os

import numpy as np

from bmlab import Session
from bmlab.file import OVERVIEW_BRIGHTFIELD_CHANNEL, FLUORESCENCE_GROUP
from bmlab.export.naming import timestamp_tag, folder_name, \
    running_brillouin_repetition
from bmlab.export.selection import repetition_selected
from bmlab.export.alignment import get_tmatrix, orient_image, \
    get_point_to_pixel_matrix, write_transform_csv, \
    calibration_stage_to_pixel
from bmlab.export.tiffio import write_tiff

# Two tile/z positions closer than this (um) are treated as "the same
# point" - generous enough to absorb float noise and hysteresis-
# compensation jitter, tight enough to stay well under real tile
# spacing (camera-FOV sized, so at least tens of um).
_POSITION_TOLERANCE_UM = 1.0


class OverviewBrightfieldExport(object):
    """
    Exports the brightfield overview images BrillouinAcquisition takes
    during a Brillouin run, as TIFF with the pixel type kept.

    Layout, below the experiment's 'Plots' folder: one folder
    Overview_BMrep<N> per Brillouin repetition, holding
    overview_<n>_<YYYYMMDD-HHMMSS>.tif - n counts the overviews of that
    repetition in the order they were taken, the time stamp is that of
    the first image. What a file contains depends on how it was taken:

    - a single image: a plain TIFF,
    - several images at one position (a z stack): a multi-page TIFF,
    - several positions (tiles): the merged mosaic, as a multi-page TIFF
      when each position was imaged at several z.

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

        path = self._plot_path()
        counters = {}

        for repetition_key in self.file.repetition_keys(self.mode):
            repetition = self.file.get_repetition(repetition_key, self.mode)
            payload = repetition.payload
            # Sorted by time so that slice/tile order reflects the order
            # in which the images were taken.
            image_keys = payload.image_keys_by_channel(
                OVERVIEW_BRIGHTFIELD_CHANNEL, sort_by_time=True)
            if not image_keys:
                continue

            # Overview images are only ever taken during a Brillouin run,
            # so they belong to the repetition that was running when the
            # first one was taken - whether or not the file records a
            # repetition number for them (files from older acquisition
            # software have none, and a wrong one when several runs share
            # a file; see running_brillouin_repetition()).
            recorded = payload.get_brillouin_repetition_index(image_keys[0])
            brillouin_repetition_key = running_brillouin_repetition(
                self.file, payload.get_date(image_keys[0]), recorded)
            if brillouin_repetition_key is not None \
                    and not repetition_selected(
                        configuration, brillouin_repetition_key):
                continue
            folder = path / folder_name('Overview', brillouin_repetition_key)

            scale_calibration = payload.get_scale_calibration()
            tmatrix = get_tmatrix(scale_calibration)

            # A single stack only means something if it's one image per
            # outer z-plane (overviewBrightfieldFullStack off, the
            # common case) - each plane's own frame slots naturally into
            # one combined stack. With it on, BrillouinAcquisition
            # captures a *local* sub-stack (and/or several tiles) at
            # every outer z-plane instead; those don't belong in one
            # combined stack together - they're different z-planes'
            # own, independent local z-ranges - so each outer plane gets
            # its own file. overview-brightfield-point-stack-counts
            # (see get_overview_brightfield_positions()) is exactly the
            # planned per-plane image count, so it tells us which case
            # this is without guessing from the data itself. These
            # datasets are written into the *Brillouin* repetition's own
            # payload by runMeasurementPhase(), not the Fluorescence
            # repetition the images themselves live in - hence needing
            # brillouin_repetition_key (already resolved above) to reach
            # them at all.
            overview_positions = None
            if brillouin_repetition_key is not None \
                    and brillouin_repetition_key in \
                    self.file.repetition_keys():
                bm_repetition = \
                    self.file.get_repetition(brillouin_repetition_key)
                overview_positions = \
                    bm_repetition.payload.get_overview_brightfield_positions()
            total_per_z = \
                overview_positions['tile_count'] if overview_positions \
                else 1
            z_steps = len(image_keys) // total_per_z if total_per_z else 0

            if total_per_z <= 1 or z_steps < 1 \
                    or z_steps * total_per_z != len(image_keys):
                # One image per plane (or the actual image count doesn't
                # match the planned shape, e.g. some captures were
                # skipped - safest to fall back to one combined stack
                # rather than guess a grouping that may not hold). Prefer
                # the actual read-back stage position over the target one
                # (see Payload.get_stage_position()'s own docstring:
                # hysteresis compensation/backlash can move these apart)
                # so the transform matrix anchors on where the image was
                # really captured - falling back to the target position
                # for older files that never recorded the read-back
                # value.
                positions = [
                    payload.get_stage_position(key)
                    or payload.get_position(key)
                    for key in image_keys]
                groups = [(image_keys, positions)]
            else:
                groups = []
                for z_index in range(z_steps):
                    keys_at_z = image_keys[
                        z_index * total_per_z:(z_index + 1) * total_per_z]
                    # Prefer the planned per-point positions recorded on
                    # the Brillouin side (reshaped to (z_steps,
                    # total_per_z) by get_overview_brightfield_positions)
                    # over each image's own position attribute, which
                    # was only added in a later BrillouinAcquisition
                    # version and can be entirely absent even when the
                    # planned-position dataset is present.
                    groups.append((keys_at_z, [
                        {'x': overview_positions['x'][z_index, p],
                         'y': overview_positions['y'][z_index, p],
                         'z': overview_positions['z'][z_index, p]}
                        for p in range(total_per_z)]))

            for keys, positions in groups:
                counters[folder] = counters.get(folder, 0) + 1
                os.makedirs(folder, exist_ok=True)
                stem = f"overview_{counters[folder]}_" \
                    f"{timestamp_tag(payload.get_date(keys[0]))}"
                self._export_group(
                    repetition, keys, positions, tmatrix, scale_calibration,
                    folder, stem)

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

        pages = []
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
                pages.append(self._compose_raw_mosaic(placed))

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
    def _compose_raw_mosaic(placed):
        height = max(row + tile.shape[0] for tile, row, _ in placed)
        width = max(col + tile.shape[1] for tile, _, col in placed)
        canvas = np.full((height, width), np.nan)
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
