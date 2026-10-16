import os

import numpy as np
import scipy.signal

from bmlab import Session
from bmlab.export.naming import timestamp_tag, \
    following_brillouin_repetition, folder_name
from bmlab.export.selection import repetition_selected
from bmlab.export.alignment import get_tmatrix, \
    calibration_stage_to_pixel, orient_image, \
    get_point_to_pixel_matrix, write_transform_csv
from bmlab.export.tiffio import write_tiff


class FluorescenceCombinedExport(object):
    """
    Colour composites of the red/green/blue images of one Fluorescence
    repetition (one click on "Acquire"), as 8 bit RGB TIFF in the same
    FL_BMrep<N> folder as the single channel images (see
    FluorescenceExport):
    fluorescenceCombined_<combination>_<YYYYMMDD-HHMMSS>.tif, where the
    combination is e.g. 'rgb', 'rg_' or '__b'.
    """

    def __init__(self):
        self.session = Session.get_instance()
        self.file = self.session.file
        self.mode = 'Fluorescence'
        return

    def export(self, configuration):
        if not self.file:
            return

        config = configuration['fluorescenceCombined']
        if not config['export']:
            return

        channels = ['red', 'green', 'blue']
        combinations = [
            'r__',
            '_g_',
            '__b',
            'rg_',
            'r_b',
            '_gb',
            'rgb',
        ]

        if self.file.path.parent.name == 'RawData':
            path = self.file.path.parents[1] / 'Plots'
        else:
            path = self.file.path.parent

        for fluorescence_repetition in self.file.repetition_keys(self.mode):
            repetition = self.file.get_repetition(
                fluorescence_repetition, self.mode)
            payload = repetition.payload
            image_keys = payload.image_keys(sort_by_time=True)
            if not image_keys:
                continue

            captured = payload.get_date(image_keys[0])
            recorded = payload.get_brillouin_repetition_index(image_keys[0])
            brillouin_key = str(recorded) if recorded is not None \
                else following_brillouin_repetition(self.file, captured)
            if brillouin_key is not None and not repetition_selected(
                    configuration, brillouin_key):
                continue

            img_data = payload.get_image(image_keys[0])
            rgb_data = np.zeros((img_data.shape[1], img_data.shape[2], 3))

            scale_calibration = payload.get_scale_calibration()
            tmatrix = get_tmatrix(scale_calibration)
            stage_map = calibration_stage_to_pixel(scale_calibration) \
                if scale_calibration is not None else None

            position = payload.get_position(image_keys[0])
            anchor_um = (position['x'], position['y']) \
                if position is not None else None

            channels_available = []
            for image_key in image_keys:
                channel = payload.get_channel(image_key)
                if channel.casefold() not in channels:
                    continue
                idx = channels.index(channel.casefold())
                channels_available.append(channels[idx][0])
                img_data = np.nanmean(payload.get_image(image_key), axis=0)
                img_data_filtered = scipy.signal.medfilt2d(img_data)
                tmp = abs(img_data - img_data_filtered) >\
                    3 * np.nanstd(img_data)
                img_data[tmp] = img_data_filtered[tmp]
                img_data = img_data - np.nanmin(img_data)
                img_data = 255 * (img_data / np.nanmax(img_data))
                rgb_data[:, :, idx] = img_data

            if not channels_available:
                continue

            folder = path / folder_name('FL', brillouin_key)
            stamp = timestamp_tag(captured)
            for combination in combinations:
                if any(c != '_' and c not in channels_available
                       for c in combination):
                    continue
                # Channels that are not part of the combination stay black
                keep = np.array([c != '_' for c in combination], dtype=float)
                composite = rgb_data * keep

                os.makedirs(folder, exist_ok=True)
                stem = f"fluorescenceCombined_{combination}_{stamp}"
                if tmatrix is None:
                    write_tiff(folder / f"{stem}_cameraPixels.tif",
                               composite.astype(np.uint8), rgb=True)
                    continue

                oriented, translate = orient_image(composite, tmatrix)
                oriented = np.nan_to_num(oriented, nan=0.0)
                filename = folder / f"{stem}.tif"
                write_tiff(filename, np.clip(oriented, 0, 255).astype(
                    np.uint8), rgb=True)

                matrix = get_point_to_pixel_matrix(
                    tmatrix, stage_map, anchor_um, rgb_data.shape,
                    translate)
                write_transform_csv(matrix, filename)
