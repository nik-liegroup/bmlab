import os

import numpy as np

from bmlab import Session
from bmlab.file import OVERVIEW_BRIGHTFIELD_CHANNEL, \
    PER_POINT_BRIGHTFIELD_CHANNEL
from bmlab.export.naming import sanitize_for_filename, timestamp_tag, \
    following_brillouin_repetition, running_brillouin_repetition, \
    folder_name
from bmlab.export.selection import repetition_selected
from bmlab.export.alignment import get_tmatrix, \
    calibration_stage_to_pixel, orient_image, \
    get_point_to_pixel_matrix, write_transform_csv
from bmlab.export.tiffio import write_tiff


class FluorescenceExport(object):
    """
    Exports the images taken in the Fluorescence tab and the per-point
    brightfield images of a Brillouin run as TIFF (pixel type kept) -
    the z overview images of a run are exported by
    OverviewBrightfieldExport.

    Layout, below the experiment's 'Plots' folder:

    - FL_BMrep<N>/<Channel>_<YYYYMMDD-HHMMSS>.tif: the images taken by
      hand. They are not part of a Brillouin measurement, but they are
      taken to prepare one - so they are filed under the Brillouin
      repetition acquired next (see following_brillouin_repetition();
      FL_afterBM if none followed). The channel name is the one the
      acquisition software stored with the image ('Brightfield', 'Red',
      ...); the time stamp is the capture time.
    - PerPoint_BMrep<N>/perpoint_<n>_<YYYYMMDD-HHMMSS>.tif: the
      brightfield images taken at the measured grid points, numbered in
      the order they were taken.
    """

    def __init__(self):
        self.session = Session.get_instance()
        self.file = self.session.file
        self.mode = 'Fluorescence'
        return

    def export(self, configuration):
        if not self.file:
            return

        config = configuration['fluorescence']
        if not config['export']:
            return

        path = self._plot_path()
        written = set()
        per_point_count = {}

        for fluorescence_repetition in self.file.repetition_keys(self.mode):
            repetition = self.file.get_repetition(
                fluorescence_repetition, self.mode)
            payload = repetition.payload
            image_keys = [
                key for key in payload.image_keys(sort_by_time=True)
                if payload.get_channel(key) != OVERVIEW_BRIGHTFIELD_CHANNEL
            ]
            if not image_keys:
                continue

            # All images of one repetition (one click on "Acquire") were
            # taken together: the first one decides where they belong.
            captured = payload.get_date(image_keys[0])
            following = following_brillouin_repetition(self.file, captured)

            scale_calibration = payload.get_scale_calibration()
            tmatrix = get_tmatrix(scale_calibration)
            stage_map = calibration_stage_to_pixel(scale_calibration) \
                if scale_calibration is not None else None

            for image_key in image_keys:
                channel = payload.get_channel(image_key)
                recorded = payload.get_brillouin_repetition_index(image_key)
                if recorded is not None \
                        or channel == PER_POINT_BRIGHTFIELD_CHANNEL:
                    # taken during a run (the per-point channel only
                    # exists during one, even where an older file did
                    # not record the repetition): the one running then
                    brillouin_key = running_brillouin_repetition(
                        self.file, payload.get_date(image_key), recorded)
                else:
                    brillouin_key = following
                if brillouin_key is not None and not repetition_selected(
                        configuration, brillouin_key):
                    continue

                if channel == PER_POINT_BRIGHTFIELD_CHANNEL:
                    folder = path / folder_name('PerPoint', brillouin_key)
                    number = per_point_count.get(folder, 0) + 1
                    per_point_count[folder] = number
                    stem = f"perpoint_{number}_" \
                        f"{timestamp_tag(payload.get_date(image_key))}"
                else:
                    folder = path / folder_name('FL', brillouin_key)
                    stem = f"{sanitize_for_filename(channel) or 'image'}_" \
                        f"{timestamp_tag(payload.get_date(image_key))}"
                os.makedirs(folder, exist_ok=True)
                stem = self._unique(folder, stem, written)

                img_data = self._read(payload, image_key)
                if tmatrix is None:
                    # Without a scale calibration the image cannot be
                    # aligned to the stage axes - the camera-pixel image
                    # is the only export.
                    write_tiff(folder / f"{stem}_cameraPixels.tif",
                               img_data, rgb=self._is_rgb(img_data))
                    continue

                oriented, translate = orient_image(img_data, tmatrix)
                oriented = self._as_pixel_type(oriented, img_data.dtype)
                filename = folder / f"{stem}.tif"
                write_tiff(filename, oriented, rgb=self._is_rgb(oriented))

                # The per-point channel, like the overview images, carries
                # the read-back stage position (it can differ slightly
                # from the target through hysteresis/backlash) - prefer
                # it, fall back to the capture position. The matrix maps
                # [x_um, y_um, 1] onto [col, row, 1] of the image.
                position = payload.get_stage_position(image_key) \
                    or payload.get_position(image_key)
                anchor_um = (position['x'], position['y']) \
                    if position is not None else None
                matrix = get_point_to_pixel_matrix(
                    tmatrix, stage_map, anchor_um, img_data.shape,
                    translate)
                write_transform_csv(matrix, filename)

    @staticmethod
    def _read(payload, image_key):
        """The image as 2D (or rows x cols x 3), pixel type kept; several
        frames are averaged."""
        img_data = payload.get_image(image_key)
        image_class = payload.get_class(image_key)
        if image_class and image_class.casefold() == 'image_grayscale':
            if img_data.shape[0] == 1:
                return np.asarray(img_data[0])
            mean = np.nanmean(img_data, axis=0)
            if np.issubdtype(img_data.dtype, np.integer):
                mean = np.rint(mean)
            return mean.astype(img_data.dtype)
        # Swap dimensions if interlace is "plane"
        image_mode = payload.get_mode(image_key)
        if image_mode and image_mode.casefold() == 'interlace_plane':
            img_data = np.transpose(img_data, (1, 2, 0))
        return np.asarray(img_data)

    @staticmethod
    def _is_rgb(img_data):
        return img_data.ndim == 3 and img_data.shape[2] == 3

    @staticmethod
    def _as_pixel_type(image, dtype):
        """Back to the camera's pixel type (only needed when the image
        had to be resampled, which leaves floats and NaN)."""
        if image.dtype == dtype:
            return image
        image = np.nan_to_num(image, nan=0.0)
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            image = np.clip(np.rint(image), info.min, info.max)
        return image.astype(dtype)

    @staticmethod
    def _unique(folder, stem, written):
        """Two images taken in the same second must not overwrite each
        other."""
        candidate, count = stem, 1
        while (folder / candidate) in written:
            count += 1
            candidate = f"{stem}-{count}"
        written.add(folder / candidate)
        return candidate

    def _plot_path(self):
        if self.file.path.parent.name == 'RawData':
            path = self.file.path.parents[1] / 'Plots'
        else:
            path = self.file.path.parent
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path
