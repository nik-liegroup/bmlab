import os

import numpy as np
from PIL import Image

from bmlab import Session
from bmlab.file import OVERVIEW_BRIGHTFIELD_CHANNEL, FLUORESCENCE_GROUP


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

        fluorescence_repetitions = self.file.repetition_keys(self.mode)

        for repetition_key in fluorescence_repetitions:
            repetition = self.file.get_repetition(repetition_key, self.mode)
            # Sort by time so the index in the exported filename reflects
            # acquisition order (z-slice / tile order), and so it doesn't
            # collide across images which all share the same channel name.
            image_keys = repetition.payload.image_keys_by_channel(
                OVERVIEW_BRIGHTFIELD_CHANNEL, sort_by_time=True)

            if not image_keys:
                continue

            path = self._plot_path()

            for index, image_key in enumerate(image_keys):
                img_data = repetition.payload.get_image(image_key)
                if img_data is None:
                    continue
                img_data = np.nanmean(img_data, axis=0).astype(np.ubyte)

                filename = path / f"{self.file.path.stem}" \
                                  f"_FLrep{repetition_key}" \
                                  f"_overviewBrightfield_{index:03d}.png"
                Image.fromarray(img_data).save(filename)

    def _plot_path(self):
        if self.file.path.parent.name == 'RawData':
            path = self.file.path.parents[1] / 'Plots'
        else:
            path = self.file.path.parent
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path
