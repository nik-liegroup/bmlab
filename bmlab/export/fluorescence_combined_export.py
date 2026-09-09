import os
import numpy as np
import scipy
from PIL import Image

from bmlab import Session
from bmlab.export.timing import get_brillouin_windows, classify_timing
from bmlab.export.alignment import get_tmatrix, warp_local


class CombinationInvalid(Exception):
    pass


class FluorescenceCombinedExport(object):

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

        fluorescence_repetitions = self.file.repetition_keys(self.mode)
        brillouin_windows = get_brillouin_windows(self.file)

        # Channels that we look for
        channels = ['red', 'green', 'blue']
        # Possible channel combinations
        combinations = [
            'r__',
            '_g_',
            '__b',
            'rg_',
            'r_b',
            '_gb',
            'rgb',
        ]

        # Loop over all fluorescence repetitions
        for fluorescence_repetition in fluorescence_repetitions:
            # Get the repetition
            repetition = self.file.get_repetition(
                fluorescence_repetition, self.mode)
            # Get the keys for all images in this repetition
            image_keys = repetition.payload.image_keys()
            if not image_keys:
                continue

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

            # Read first image of repetition,
            # so we can create an array to store the RGB data
            img_data = repetition.payload.get_image(image_keys[0])
            rgb_data = np.zeros((img_data.shape[1], img_data.shape[2], 3))

            # Get the scale calibration
            scale_calibration = repetition.payload.get_scale_calibration()
            tmatrix = get_tmatrix(scale_calibration)

            channels_available = []
            # Loop over all images in this repetition and
            # construct an array with all channels available
            for image_key in image_keys:
                channel = repetition.payload.get_channel(image_key)

                # If the channel is not red, green or blue, continue
                try:
                    idx = channels.index(channel.casefold())
                    channels_available.append(channels[idx][0])
                    # Average all images acquired
                    img_data = repetition.payload.get_image(image_key)
                    img_data = np.nanmean(img_data, axis=0)
                    # We apply a median filter to remove salt and pepper noise
                    # for pixels whose value is more than 3 sigma different
                    # from the median filtered value.
                    img_data_filtered = scipy.signal.medfilt2d(img_data)
                    tmp = abs(img_data - img_data_filtered) >\
                        3 * np.nanstd(img_data)
                    img_data[tmp] = img_data_filtered[tmp]
                    # We scale the data to max contrast here
                    img_data = img_data - np.nanmin(img_data)
                    img_data = 255 * (img_data / np.nanmax(img_data))
                    rgb_data[:, :, idx] = img_data
                except ValueError:
                    continue

            # If there are no fluorescence channels, continue
            if not channels_available:
                continue

            # Loop over all possible channel combinations
            for combination in combinations:
                # Check if this combination is possible
                # with the available channels
                try:
                    for channel in combination:
                        if channel != '_' \
                                and channel not in channels_available:
                            raise CombinationInvalid
                except CombinationInvalid:
                    continue
                # Create a conversion matrix
                # to drop not needed channels
                channel_matrix = np.matrix([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0]
                ])
                for i, ch in enumerate(combination):
                    if ch == '_':
                        channel_matrix[i, i] = 0
                channel_matrix =\
                    tuple(i[0, 0] for i in
                          channel_matrix.flatten().transpose())

                # Construct export path and create it if necessary
                if self.file.path.parent.name == 'RawData':
                    path = self.file.path.parents[1] / 'Plots' / 'Bare'
                else:
                    path = self.file.path.parent
                if not os.path.exists(path):
                    os.makedirs(path, exist_ok=True)

                # Without a scale calibration we cannot align the image
                # to the stage's x-y axes, so the only meaningful export
                # is the raw camera-pixel-space image - it is the sole
                # export in that case. Otherwise, only the stage-aligned
                # combination is exported (see FluorescenceExport for
                # why the unaligned image isn't useful on its own).
                if tmatrix is None:
                    filename = path / \
                        f"fluorescenceCombined_{combination}" \
                        f"{timing_part}_cameraPixels.png"
                    image = Image.fromarray(rgb_data.astype(np.ubyte))
                    im = image.convert("RGB", channel_matrix)
                    im.save(filename)
                    continue

                # Warp the image to align with a standard x-y
                # coordinate system
                image_data_warped, _, _ = warp_local(rgb_data, tmatrix)

                # Export image with proper alpha channel
                image_warped = Image.fromarray(
                    image_data_warped.astype(np.ubyte))
                image_warped = image_warped.convert("RGB", channel_matrix)
                image_alpha = Image.fromarray(
                    (255 * np.logical_not(
                        np.isnan(np.nanmean(
                            image_data_warped, axis=2)))).astype(np.ubyte))
                image_warped.putalpha(image_alpha)

                filename = path / \
                    f"fluorescenceCombined_{combination}{timing_part}.png"
                image_warped.save(filename)
