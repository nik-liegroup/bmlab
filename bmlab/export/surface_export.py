import os
import json

import numpy as np
import matplotlib.pyplot as plt

from bmlab import Session
from bmlab.file import Payload


class SurfaceExport(object):

    def __init__(self):
        self.session = Session.get_instance()
        self.file = self.session.file
        self.mode = 'Brillouin'
        return

    def export(self, configuration):
        if not self.file:
            return

        config = configuration.get('surface')
        if not config or not config.get('export'):
            return

        for repetition_key in self.file.repetition_keys():
            repetition = self.file.get_repetition(repetition_key)
            data = repetition.payload.get_surface_scan_data()
            if data is None:
                continue

            filename_base = \
                f"{self.file.path.stem}_BMrep{repetition_key}_surface"

            plot_path = self._plot_path()
            self._export_heatmap(
                data.get('surface_found_mask'),
                plot_path / f"{filename_base}_found_mask.png",
                'Surface found\n(0=missing, 1=measured, 2=interpolated)',
                vmax=2,
            )
            self._export_heatmap(
                data.get('roi_scan_plan_mask'),
                plot_path / f"{filename_base}_roi_plan_mask.png",
                'ROI scan plan\n(1=inside planned ROI)',
                vmax=1,
            )
            sampled_mask = data.get('sampled_mask')
            sampled_projection = None
            if sampled_mask is not None and sampled_mask.size:
                # Project across z: 1 if sampled at any z-step
                sampled_projection = np.nanmax(sampled_mask, axis=0)
            self._export_heatmap(
                sampled_projection,
                plot_path / f"{filename_base}_sampled_mask.png",
                'Sampled\n(1=sampled at any z-step)',
                vmax=1,
            )

            metrics = self._compute_metrics(data, sampled_projection)

            export_path = self._export_data_path()
            json_filename = export_path / f"{filename_base}_metrics.json"
            with open(json_filename, 'w') as f:
                json.dump(metrics, f, indent=2, default=_json_default)

    def _plot_path(self):
        if self.file.path.parent.name == 'RawData':
            path = self.file.path.parents[1] / 'Plots'
        else:
            path = self.file.path.parent
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path

    def _export_data_path(self):
        if self.file.path.parent.name == 'RawData':
            path = self.file.path.parents[1] / 'Export'
        else:
            path = self.file.path.parent
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        return path

    @staticmethod
    def _export_heatmap(mask, filename, title, vmax):
        if mask is None:
            return
        fig, ax = plt.subplots()
        im = ax.imshow(
            np.rot90(mask), interpolation='nearest', vmin=0, vmax=vmax)
        ax.set_xlabel('$x$ index')
        ax.set_ylabel('$y$ index')
        ax.set_title(title)
        fig.colorbar(im)
        fig.savefig(filename)
        plt.close(fig)

    @staticmethod
    def _compute_metrics(data, sampled_projection):
        metrics = {}

        found_mask = data.get('surface_found_mask')
        if found_mask is not None and found_mask.size:
            total = found_mask.size
            metrics['surface_found_fraction'] = \
                float(np.sum(found_mask == 1)) / total
            metrics['surface_interpolated_fraction'] = \
                float(np.sum(found_mask == 2)) / total
            metrics['surface_missing_fraction'] = \
                float(np.sum(found_mask == 0)) / total

        roi_mask = data.get('roi_scan_plan_mask')
        if roi_mask is not None and sampled_projection is not None \
                and roi_mask.size:
            roi_bool = roi_mask.astype(bool)
            if np.any(roi_bool):
                metrics['roi_coverage_fraction'] = float(
                    np.sum(sampled_projection[roi_bool] > 0)
                ) / float(np.sum(roi_bool))

        for key in Payload.SURFACE_SCAN_SETTINGS:
            value = data.get(key)
            if value is not None:
                metrics[key] = value

        return metrics


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")
