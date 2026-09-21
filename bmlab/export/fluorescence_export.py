import os
import numpy as np
from PIL import Image

from bmlab import Session
from bmlab.file import OVERVIEW_BRIGHTFIELD_CHANNEL
from bmlab.export.naming import repetition_or_timestamp_tag, \
    sanitize_for_filename
from bmlab.export.selection import repetition_selected
from bmlab.export.alignment import get_tmatrix, get_pixels_per_um, \
    warp_local, get_point_to_pixel_matrix, write_transform_csv


class FluorescenceExport(object):

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

        fluorescence_repetitions = self.file.repetition_keys(self.mode)

        # Loop over all fluorescence repetitions
        for fluorescence_repetition in fluorescence_repetitions:
            # Get the repetition
            repetition = self.file.get_repetition(
                fluorescence_repetition, self.mode)
            # Get the keys for all images in this repetition, excluding
            # the brightfield z-stack/mosaic overview images - those are
            # handled separately by OverviewBrightfieldExport.
            image_keys = [
                key for key in repetition.payload.image_keys()
                if repetition.payload.get_channel(key)
                != OVERVIEW_BRIGHTFIELD_CHANNEL
            ]
            if not image_keys:
                continue

            # Two images sharing the same channel within one repetition
            # would otherwise produce the same filename and silently
            # overwrite each other - number them if that happens.
            channels_seen = [
                repetition.payload.get_channel(key) for key in image_keys]
            channel_counts = {
                channel: channels_seen.count(channel)
                for channel in set(channels_seen)}
            channel_occurrence = {}

            # The Brillouin repetition every image here was captured as
            # part of (see MeasurementData.get_brillouin_repetition_index())
            # - or, for a standalone capture with no such repetition, its
            # own timestamp instead (repetition_or_timestamp_tag()). All
            # images in one Fluorescence repetition share the same
            # relationship in practice, so the first one speaks for the
            # whole repetition, same granularity the removed capture-
            # time-overlap heuristic used.
            keys_by_time = repetition.payload.image_keys(sort_by_time=True)
            brillouin_index = repetition.payload\
                .get_brillouin_repetition_index(keys_by_time[0])
            # Restrict to the selected Brillouin repetitions (see
            # ExportController.get_configuration()'s own
            # 'brillouin'/'repetitions') the same way BrillouinExport
            # does for the CSV - only when these images actually belong
            # to one; a standalone capture with no such repetition is
            # never excluded by that selection.
            if brillouin_index is not None and not repetition_selected(
                    configuration, str(brillouin_index)):
                continue
            tag = repetition_or_timestamp_tag(
                repetition.payload, keys_by_time[0])
            timing_part = f"{tag}_FLrep{fluorescence_repetition}"

            # Get the scale calibration
            scale_calibration = repetition.payload.get_scale_calibration()
            tmatrix = get_tmatrix(scale_calibration)
            pixels_per_um = get_pixels_per_um(scale_calibration)

            # Loop over all images in this repetition
            for image_key in image_keys:
                channel = repetition.payload.get_channel(image_key)
                channel_tag = sanitize_for_filename(channel)
                if channel_counts[channel] > 1:
                    channel_occurrence[channel] = \
                        channel_occurrence.get(channel, -1) + 1
                    channel_tag = \
                        f"{channel_tag}-{channel_occurrence[channel]}"
                img_data = repetition.payload.get_image(image_key)

                # Average all images acquired if grayscale
                image_class = repetition.payload.get_class(image_key)
                if (image_class and image_class.casefold()
                        == 'image_grayscale'):
                    img_data = np.nanmean(img_data, axis=0).astype(np.ubyte)

                # Swap dimensions if interlace is "plane"
                image_mode = repetition.payload.get_mode(image_key)
                if (image_mode and image_mode.casefold()
                        == 'interlace_plane'):
                    img_data = np.transpose(img_data, (1, 2, 0))

                # Construct export path and create it if necessary
                if self.file.path.parent.name == 'RawData':
                    path = self.file.path.parents[1] / 'Plots'
                else:
                    path = self.file.path.parent
                if not os.path.exists(path):
                    os.mkdir(path)

                def build_image(data):
                    image = Image.fromarray(data)
                    if channel.casefold() == 'red':
                        blank = Image.new("L", image.size)
                        image = Image.merge("RGB", (image, blank, blank))
                    if channel.casefold() == 'green':
                        blank = Image.new("L", image.size)
                        image = Image.merge("RGB", (blank, image, blank))
                    if channel.casefold() == 'blue':
                        blank = Image.new("L", image.size)
                        image = Image.merge("RGB", (blank, blank, image))
                    return image

                # Without a scale calibration we cannot align the image
                # to the stage's x-y axes, so the only meaningful export
                # is the raw camera-pixel-space image - it is the sole
                # export in that case. Otherwise, only the stage-aligned
                # image is exported: an unaligned image's pixel
                # coordinates don't correspond to the stage/Brillouin
                # positions stored elsewhere in the file, so it isn't
                # useful for overlaying with the Brillouin map.
                if tmatrix is None:
                    image = build_image(img_data)
                    filename = path / \
                        f"{channel_tag}{timing_part}_cameraPixels.png"
                    image.save(filename)
                    continue

                # Warp the image to align with a standard x-y
                # coordinate system
                image_data_warped, _, translate = \
                    warp_local(img_data, tmatrix)

                # Export image with proper alpha channel
                image_warped = build_image(
                    (255 * image_data_warped).astype(np.ubyte))
                image_alpha = Image.fromarray(
                    np.nanmean(255 * np.logical_not(
                        np.isnan(image_data_warped)),
                               axis=tuple(range(2, image_data_warped.ndim)))
                    .astype(np.ubyte))
                image_warped.putalpha(image_alpha)

                filename = path / f"{channel_tag}{timing_part}.png"
                image_warped.save(filename)

                # Most regular Fluorescence-mode images are a single-
                # point capture with no separate read-back stage
                # position to prefer (see Payload.get_stage_position()'s
                # own docstring) - get_position() is their actual
                # capture position. The "Brightfield per-point" channel
                # is the exception: like the brightfield overview
                # images, it does carry a read-back stage position,
                # which can differ slightly from the target due to
                # hysteresis/backlash, so prefer it here too - falling
                # back to get_position() for every other channel, where
                # get_stage_position() is simply None. Write the same
                # stage-position -> pixel transform matrix
                # OverviewBrightfieldExport writes for its own images,
                # so this image can be placed relative to the Brillouin
                # CSV grid or an overview image using the same
                # [col, row, 1] = M @ [x_um, y_um, 1] mapping.
                position = repetition.payload.get_stage_position(image_key) \
                    or repetition.payload.get_position(image_key)
                anchor_um = (position['x'], position['y']) \
                    if position is not None else None
                matrix = get_point_to_pixel_matrix(
                    tmatrix, pixels_per_um, anchor_um, img_data.shape,
                    translate)
                write_transform_csv(matrix, filename)
