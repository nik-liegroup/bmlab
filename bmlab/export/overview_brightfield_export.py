import csv
import os

import numpy as np
from PIL import Image

from bmlab import Session
from bmlab.file import OVERVIEW_BRIGHTFIELD_CHANNEL, FLUORESCENCE_GROUP
from bmlab.export.timing import get_brillouin_windows, classify_timing
from bmlab.export.alignment import get_tmatrix, get_pixels_per_um, \
    warp_local, um_offset_to_pixels, get_point_to_pixel_matrix

# Two tile/z positions closer than this (um) are treated as "the same
# point" - generous enough to absorb float noise and hysteresis-
# compensation jitter, tight enough to stay well under real tile
# spacing (camera-FOV sized, so at least tens of um).
_POSITION_TOLERANCE_UM = 1.0


class OverviewBrightfieldExport(object):

    def __init__(self):
        self.session = Session.get_instance()
        self.file = self.session.file
        self.mode = FLUORESCENCE_GROUP
        return

    def export(self, configuration):
        if not self.file:
            return

        config = configuration.get('overviewBrightfield')
        if not config or not config.get('export'):
            return

        brillouin_windows = get_brillouin_windows(self.file)
        fluorescence_repetitions = self.file.repetition_keys(self.mode)

        for repetition_key in fluorescence_repetitions:
            repetition = self.file.get_repetition(repetition_key, self.mode)
            # Sort by time so slice/tile order reflects acquisition order.
            image_keys = repetition.payload.image_keys_by_channel(
                OVERVIEW_BRIGHTFIELD_CHANNEL, sort_by_time=True)
            if not image_keys:
                continue

            dates = [repetition.payload.get_date(key) for key in image_keys]
            timing = classify_timing(dates[0], dates[-1], brillouin_windows)
            # Only one overview stack is ever written per Brillouin
            # repetition (saveOverviewBrightfieldPerZ starts exactly one
            # new Fluorescence repetition per Brillouin repetition), so
            # the Fluorescence repetition index adds no information once
            # the Brillouin repetition it belongs to is known.
            tag = f"_BMrep{timing[0]}_{timing[1]}Acq" if timing else ''

            path = self._plot_path()
            scale_calibration = repetition.payload.get_scale_calibration()
            tmatrix = get_tmatrix(scale_calibration)
            pixels_per_um = get_pixels_per_um(scale_calibration)

            # A single "z-stack" only means something if it's one image
            # per outer z-plane (overviewBrightfieldFullStack off, the
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
            # repetition the images themselves live in - timing[0]
            # (already computed above) is that Brillouin repetition's
            # key, since this is always a 'duringAcq' association for an
            # overview-carrying Fluorescence repetition.
            overview_positions = None
            if timing:
                bm_repetition = self.file.get_repetition(timing[0])
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
                # rather than guess a grouping that may not hold). Only
                # one point per plane here anyway (total_per_z <= 1 in
                # the common case), so per-image position attributes -
                # which may be missing entirely on older files - are
                # not needed for tile placement, only warp_local() is.
                positions = [
                    repetition.payload.get_position(key)
                    for key in image_keys]
                self._export_group(
                    repetition, image_keys, positions, tmatrix,
                    pixels_per_um, path, f"overviewZStack{tag}")
            else:
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
                    positions_at_z = [
                        {'x': overview_positions['x'][z_index, p],
                         'y': overview_positions['y'][z_index, p],
                         'z': overview_positions['z'][z_index, p]}
                        for p in range(total_per_z)
                    ]
                    self._export_group(
                        repetition, keys_at_z, positions_at_z, tmatrix,
                        pixels_per_um, path,
                        f"overviewZStack_{z_index}{tag}")

    def _export_group(
            self, repetition, image_keys, positions, tmatrix,
            pixels_per_um, path, name_stub):
        """
        Exports one group of overview images - either the whole
        repetition's worth (one image per outer z-plane) or a single
        outer z-plane's own local sub-stack/tiles - as a tiled mosaic
        stack if it contains more than one distinct tile position, or
        a plain (aligned, if possible) z-stack otherwise. `positions`
        is a list of {'x', 'y', 'z'} dicts (or None entries), one per
        image_keys entry.
        """
        tiles = None
        if tmatrix is not None and all(p is not None for p in positions):
            tiles = self._group_by_tile(positions)

        if tmatrix is not None and tiles is not None and len(tiles) > 1:
            self._export_tiled(
                repetition, image_keys, positions, tiles, tmatrix,
                pixels_per_um, path / f"{name_stub}_tiled.tif")
        else:
            if tmatrix is not None:
                filename = path / f"{name_stub}.tif"
            else:
                filename = path / f"{name_stub}_cameraPixels.tif"
            self._export_stack(
                repetition, image_keys, positions, tmatrix, pixels_per_um,
                filename)

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

    def _export_stack(
            self, repetition, image_keys, positions, tmatrix,
            pixels_per_um, filename):
        """
        A single-xy (or unaligned) z-stack: every frame gets the same
        warp (they share the same footprint, only z differs), so the
        pages come out consistently sized without any extra placement
        logic - or, with no scale calibration, the frames are saved
        exactly as captured. `positions` is the same list `_export_group`
        received - since every frame shares one footprint, the first
        one with a recorded position anchors the transform matrix
        written alongside the image (see get_point_to_pixel_matrix()).
        """
        frames = []
        anchor_um, anchor_shape, anchor_translate = None, None, None
        for image_key, position in zip(image_keys, positions):
            img_data = repetition.payload.get_image(image_key)
            if img_data is None:
                continue
            img_data = np.nanmean(img_data, axis=0)
            if tmatrix is not None:
                warped, _, translate = warp_local(img_data, tmatrix)
                frame = np.nan_to_num(warped, nan=0.0).astype(np.ubyte)
                if anchor_um is None and position is not None:
                    anchor_um = (position['x'], position['y'])
                    anchor_shape = img_data.shape
                    anchor_translate = translate
            else:
                frame = img_data.astype(np.ubyte)
            frames.append(frame)

        if not frames:
            return

        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(
            filename, save_all=True, append_images=images[1:])

        matrix = get_point_to_pixel_matrix(
            tmatrix, pixels_per_um, anchor_um, anchor_shape,
            anchor_translate)
        self._write_transform_csv(matrix, filename)

    def _export_tiled(
            self, repetition, image_keys, positions, tiles, tmatrix,
            pixels_per_um, filename):
        """
        A tiled mosaic per distinct z value found among `image_keys`:
        every tile is warped individually (rotation/scale-corrected to
        the stage axes) and then placed in a shared canvas at its own
        recorded position, relative to the first tile - so the mosaic
        reflects where each tile was actually captured, not just a
        guessed grid layout.
        """
        reference_xy = next(iter(tiles))

        z_by_index = [pos['z'] for pos in positions]
        unique_z = []
        for z in z_by_index:
            if not any(abs(z - uz) < _POSITION_TOLERANCE_UM
                       for uz in unique_z):
                unique_z.append(z)
        unique_z.sort()

        # The rotation/scale (tmatrix, pixels_per_um) and each tile's own
        # warp_local() translate are the same for every tile (same camera,
        # same calibration) - only the placement offset differs by tile
        # position - so one page's placement geometry (reference tile's
        # own shape/translate, plus the canvas' own min_x/min_y) is enough
        # to build a single transform matrix that is valid for every tile
        # in the mosaic, not just the reference one.
        pages = []
        placement = None
        for z in unique_z:
            warped_tiles = []
            for tile_xy, indices in tiles.items():
                index = next(
                    (i for i in indices
                     if abs(z_by_index[i] - z) < _POSITION_TOLERANCE_UM),
                    None)
                if index is None:
                    continue
                img_data = repetition.payload.get_image(image_keys[index])
                if img_data is None:
                    continue
                img_data = np.nanmean(img_data, axis=0)
                warped, _, translate = warp_local(img_data, tmatrix)
                dx_um = tile_xy[0] - reference_xy[0]
                dy_um = tile_xy[1] - reference_xy[1]
                dx_px, dy_px = um_offset_to_pixels(
                    tmatrix, pixels_per_um, dx_um, dy_um)
                warped_tiles.append((warped, dx_px, dy_px))
                if placement is None and tile_xy == reference_xy:
                    placement = (img_data.shape, translate)
            if warped_tiles:
                canvas, min_x, min_y = self._compose_mosaic(warped_tiles)
                pages.append(canvas)
                if placement is not None and len(placement) == 2:
                    placement = placement + (min_x, min_y)

        if not pages:
            return

        images = [
            Image.fromarray(np.nan_to_num(page, nan=0.0).astype(np.ubyte))
            for page in pages
        ]
        images[0].save(
            filename, save_all=True, append_images=images[1:])

        matrix = None
        if placement is not None and len(placement) == 4:
            (anchor_shape, anchor_translate, min_x, min_y) = placement
            matrix = get_point_to_pixel_matrix(
                tmatrix, pixels_per_um, reference_xy, anchor_shape,
                anchor_translate, placement_offset_px=(-min_x, -min_y))
        self._write_transform_csv(matrix, filename)

    @staticmethod
    def _compose_mosaic(warped_tiles):
        min_x = min(dx for _, dx, _ in warped_tiles)
        min_y = min(dy for _, _, dy in warped_tiles)
        max_x = max(dx + w.shape[1] for w, dx, _ in warped_tiles)
        max_y = max(dy + w.shape[0] for w, _, dy in warped_tiles)

        canvas = np.full(
            (int(np.ceil(max_y - min_y)), int(np.ceil(max_x - min_x))),
            np.nan)
        for warped, dx, dy in warped_tiles:
            x0 = int(round(dx - min_x))
            y0 = int(round(dy - min_y))
            region = canvas[y0:y0 + warped.shape[0], x0:x0 + warped.shape[1]]
            # First tile to reach a pixel wins - simple, deterministic,
            # avoids double-brightening the tiles' overlap margin.
            empty = np.isnan(region)
            region[empty] = warped[:region.shape[0], :region.shape[1]][empty]
        return canvas, min_x, min_y

    @staticmethod
    def _write_transform_csv(matrix, image_filename):
        """
        Writes the 3x3 matrix mapping an absolute stage position (x, y,
        um) to its pixel location in `image_filename`'s image, next to
        it as plain a plain 3-row/3-column CSV (no header) - same
        layout as BrainFusion's AFM loader already expects for its own
        GridInversionMatrix.csv. Writes nothing if `matrix` is None (no
        scale calibration, or no recorded position for this image).
        """
        if matrix is None:
            return
        csv_filename = image_filename.with_name(
            image_filename.stem + '_transform.csv')
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerows(matrix.tolist())

    def _plot_path(self):
        if self.file.path.parent.name == 'RawData':
            path = self.file.path.parents[1] / 'Plots'
        else:
            path = self.file.path.parent
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path
