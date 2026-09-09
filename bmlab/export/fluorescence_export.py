import os
import numpy as np
from PIL import Image

from bmlab import Session
from bmlab.file import OVERVIEW_BRIGHTFIELD_CHANNEL
from bmlab.export.timing import get_brillouin_windows, classify_timing, \
    sanitize_for_filename
from bmlab.export.alignment import get_tmatrix, warp_local


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
        brillouin_windows = get_brillouin_windows(self.file)

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

            keys_by_time = repetition.payload.image_keys(sort_by_time=True)
            timing = classify_timing(
                repetition.payload.get_date(keys_by_time[0]),
                repetition.payload.get_date(keys_by_time[-1]),
                brillouin_windows)
            if timing:
                timing_part = f"_{timing[1]}Acq" \
                              f"_FLrep{fluorescence_repetition}" \
                              f"_BMrep{timing[0]}"
            else:
                timing_part = f"_FLrep{fluorescence_repetition}"

            # Get the scale calibration
            scale_calibration = repetition.payload.get_scale_calibration()
            tmatrix = get_tmatrix(scale_calibration)

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
                image_class = (repetition.payload
                               .get_class(image_key).casefold())
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
                image_data_warped, _, _ = warp_local(img_data, tmatrix)

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
