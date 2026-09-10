import os
import json

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers '3d')

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

            filename_base = f"surface_BMrep{repetition_key}"

            plot_path = self._plot_path()

            found_mask = data.get('surface_found_mask')
            positions_z = repetition.payload.positions['z'] \
                if repetition.payload.positions is not None else None
            positions_x = repetition.payload.positions['x'] \
                if repetition.payload.positions is not None else None
            positions_y = repetition.payload.positions['y'] \
                if repetition.payload.positions is not None else None
            prescan_x = data.get('prescan_x')
            prescan_y = data.get('prescan_y')
            roi_polygon_x = data.get('roi_polygon_x')
            roi_polygon_y = data.get('roi_polygon_y')

            # Center x/y on one shared offset (preferring the main
            # grid's own mean, the same "relative sample coordinates"
            # convention the main Brillouin plots use) so the absolute
            # stage position - a large coordinate carrying no
            # information about the scanned area's shape - doesn't
            # show up on any of the plots below, and so the main grid,
            # the coarse pre-scan, and the ROI polygon all shift by the
            # exact same amount and stay mutually consistent rather
            # than each floating to its own independent center.
            x_offset = self._offset(positions_x, prescan_x, roi_polygon_x)
            y_offset = self._offset(positions_y, prescan_y, roi_polygon_y)
            if positions_x is not None:
                positions_x = positions_x - x_offset
            if positions_y is not None:
                positions_y = positions_y - y_offset
            if prescan_x is not None:
                prescan_x = prescan_x - x_offset
            if prescan_y is not None:
                prescan_y = prescan_y - y_offset
            if roi_polygon_x is not None:
                roi_polygon_x = roi_polygon_x - x_offset
            if roi_polygon_y is not None:
                roi_polygon_y = roi_polygon_y - y_offset

            # The coarse pre-scan's own genuinely sampled (x, y) grid -
            # the real, physically probed locations "used in coarse-
            # grain mode to find a surface". Only present on files
            # from a BrillouinAcquisition build that records it (see
            # Payload.SURFACE_SCAN_PATH); skipped entirely otherwise -
            # surface_found_mask/positions-z below are NOT a substitute
            # for this, since they live at the dense scan-plan grid's
            # resolution (often hundreds of points), not the coarse
            # pre-scan's own handful of actually measured columns.
            self._export_prescan_points(
                prescan_x, prescan_y,
                data.get('prescan_found_mask'), data.get('prescan_z'),
                plot_path / f"{filename_base}_prescan_points.png",
            )

            if found_mask is not None and positions_z is not None:
                # BrillouinAcquisition writes one z-target per (x, y)
                # into positions-z (the same value at every z-index of
                # that column's stack, by construction - see
                # Brillouin::runSurfacePreScan()/applySurfaceFollowPlan
                # in BrillouinAcquisition's source). Averaging the
                # column here is not an interpolation - it's reading
                # off that single stored value; nothing is estimated
                # or filled in beyond what BrillouinAcquisition itself
                # already computed and used to run the real scan.
                z_surface = np.nanmean(positions_z, axis=0)

                # found_mask == 2 additionally includes points whose
                # z-target leans on at least one gap-filled coarse
                # cell - still exactly what BrillouinAcquisition
                # itself computed and used to center the real scan,
                # not a bmlab estimate. found_mask == 0 (no surface
                # info at all) is left as a genuine gap - bmlab does
                # not fill it in.
                z_surface_masked = np.where(found_mask > 0, z_surface, np.nan)
                # Centered on the grid's own mean rather than left as
                # the absolute stage z (a large coordinate that carries
                # no information about the surface's shape) - used
                # identically for the 2D map and the 3D plot below, so
                # both show the exact same measurement, just in 2D/3D.
                z_relative = z_surface_masked - np.nanmean(z_surface_masked)
                z_offset = data.get('surface_z_offset_um_used') or 0.0
                title = 'Surface z used for acquisition'
                if z_offset:
                    title += f', +{z_offset:g} $\\mu$m offset'
                self._export_grid_map(
                    positions_x[0, :, :] if positions_x is not None
                    else None,
                    positions_y[0, :, :] if positions_y is not None
                    else None,
                    z_relative,
                    plot_path / f"{filename_base}_z_surface.png",
                    title,
                )

                if positions_x is not None and positions_y is not None:
                    self._export_surface_3d(
                        positions_x[0, :, :], positions_y[0, :, :],
                        z_relative,
                        plot_path / f"{filename_base}_3d.png",
                        title,
                    )

            if positions_x is not None and positions_y is not None:
                self._export_grid_map(
                    positions_x[0, :, :], positions_y[0, :, :],
                    data.get('roi_scan_plan_mask'),
                    plot_path / f"{filename_base}_roi_plan_mask.png",
                    'ROI scan plan',
                    vmin=0, vmax=1,
                )

            self._export_roi_polygon(
                roi_polygon_x, roi_polygon_y,
                plot_path / f"{filename_base}_roi_polygon.png",
            )

            sampled_mask = data.get('sampled_mask')
            sampled_projection = None
            if sampled_mask is not None and sampled_mask.size:
                # Project across z: 1 if sampled at any z-step
                sampled_projection = np.nanmax(sampled_mask, axis=0)

            metrics = self._compute_metrics(data, sampled_projection)

            export_path = self._export_data_path()
            json_filename = export_path / f"{filename_base}_metrics.json"
            with open(json_filename, 'w') as f:
                json.dump(metrics, f, indent=2, default=_json_default)

    @staticmethod
    def _offset(*arrays):
        """
        The mean of the first array that actually has finite values,
        to center a set of related (x or y) coordinate arrays on one
        shared reference point - see the centering comment in
        export(). Falls back to 0.0 (no centering) if none of the
        candidates has any data at all.
        """
        for array in arrays:
            if array is not None and np.size(array) \
                    and np.isfinite(array).any():
                return float(np.nanmean(array))
        return 0.0

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
    def _export_prescan_points(x, y, found_mask, z, filename):
        """
        Plots the coarse surface pre-scan's own genuinely sampled
        (x, y) grid (stage um) - the real, physically probed
        locations, at the pre-scan's own coarse resolution. Cells
        with no drop found at all (found_mask == 0) are left out
        entirely; genuinely measured cells (found_mask == 1) and
        cells filled in from neighboring coarse cells (found_mask
        == 2) are drawn with different markers so the two aren't
        conflated. No dense-grid data of any kind is involved.
        """
        if x is None or y is None or found_mask is None or z is None:
            return
        xx, yy = np.meshgrid(x, y, indexing='ij')
        measured = found_mask == 1
        filled = found_mask == 2
        if not np.any(measured) and not np.any(filled):
            return
        valid = found_mask > 0
        vmin = float(np.nanmin(z[valid]))
        vmax = float(np.nanmax(z[valid]))
        fig = Figure(figsize=(6.4, 5.2))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        sc = None
        if np.any(measured):
            sc = ax.scatter(
                xx[measured], yy[measured], c=z[measured],
                cmap='viridis', vmin=vmin, vmax=vmax,
                s=35, marker='o', label='measured')
        if np.any(filled):
            sc = ax.scatter(
                xx[filled], yy[filled], c=z[filled],
                cmap='viridis', vmin=vmin, vmax=vmax,
                s=35, marker='x', label='gap-filled') or sc
        ax.set_xlabel('$x$ [$\\mu$m]')
        ax.set_ylabel('$y$ [$\\mu$m]')
        ax.set_title('Coarse pre-scan surface points')
        ax.axis('equal')
        # Stage +x/+y point opposite to how the sample appears to move
        # in the BF images - flip both axes so this plot's orientation
        # (up/down and left/right) matches them.
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.legend(loc='best', fontsize='small')
        fig.colorbar(sc)
        fig.tight_layout()
        fig.savefig(filename, bbox_inches='tight')

    @staticmethod
    def _export_grid_map(x, y, z, filename, title, vmin=None, vmax=None):
        """
        Plots a 2D grid quantity `z` at its true (x, y) stage
        positions (um) via pcolormesh - handled directly from the
        actual grid coordinates, so no index-based rot90 flip is
        needed and NaN cells (no data) are simply left blank.
        """
        if x is None or y is None or z is None:
            return
        fig = Figure(figsize=(6.4, 5.2))
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        im = ax.pcolormesh(x, y, z, shading='nearest', vmin=vmin, vmax=vmax)
        ax.set_xlabel('$x$ [$\\mu$m]')
        ax.set_ylabel('$y$ [$\\mu$m]')
        ax.set_title(title)
        ax.set_aspect('equal')
        # Match the BF images' orientation - see _export_prescan_points.
        ax.invert_xaxis()
        ax.invert_yaxis()
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(filename, bbox_inches='tight')

    @staticmethod
    def _export_roi_polygon(x, y, filename):
        """
        Plots the ROI polygon as drawn at acquisition time (before it
        was rasterized into 'roi_scan_plan_mask'), exactly as stored -
        no consistency check against roi_scan_plan_mask. If the two
        don't visually line up, that's real information about the
        file worth seeing, not something to hide.
        """
        if x is None or y is None or x.size == 0:
            return
        fig = Figure()
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        # Close the polygon for display, whether or not it was
        # already stored closed.
        x_closed = np.append(x, x[0])
        y_closed = np.append(y, y[0])
        ax.plot(x_closed, y_closed, '-o', markersize=3)
        ax.set_xlabel('$x$ [$\\mu$m]')
        ax.set_ylabel('$y$ [$\\mu$m]')
        ax.set_title('ROI polygon (as drawn at acquisition time)')
        ax.axis('equal')
        # Match the BF images' orientation - see _export_prescan_points.
        ax.invert_xaxis()
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(filename, bbox_inches='tight')

    @staticmethod
    def _export_surface_3d(x, y, z, filename, title):
        """
        Renders the surface (exactly as BrillouinAcquisition computed
        and used it - no bmlab-side gap-filling, holes stay holes;
        `z` is the same zero-mean-centered measurement as the 2D
        z-surface map, just shown in 3D instead) as an actual 3D
        surface - x/y in stage um, kept at their true relative
        physical proportions so the outline of the scanned area is
        shape-correct. z is deliberately NOT scaled to its own
        physical range: that range is normally tiny compared to the
        x/y extent (a few um of height variation over a scan area
        hundreds of um wide), which would flatten any real surface
        into a barely visible sliver. Instead the z axis is stretched
        to the larger of the x/y extents - as tall as a box aspect
        can make it without exceeding the plot's own x/y footprint -
        so height variation stays visible regardless of scan size.
        """
        if x is None or y is None or z is None or not np.isfinite(z).any():
            return
        fig = Figure()
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111, projection='3d')
        # plot_surface colors each flat quad from its own corner values
        # (by default, the mean of all 4) rather than the colormap's
        # data-wide min/max, so its auto clim can come out far
        # narrower than the data's actual range - explicit vmin/vmax
        # keeps the color scale (and colorbar) honest and matching the
        # 2D map's, even though individual quads are still one flat
        # color rather than exact per-point ones.
        surf = ax.plot_surface(
            x, y, z, cmap='viridis', linewidth=0, antialiased=True,
            vmin=np.nanmin(z), vmax=np.nanmax(z))
        ax.set_xlabel('$x$ [$\\mu$m]')
        ax.set_ylabel('$y$ [$\\mu$m]')
        ax.set_zlabel('$\\Delta z$ [$\\mu$m] (relative to mean)')
        ax.set_title(title)

        # x/y keep their true relative physical proportions; z is
        # exaggerated to the larger of the two (see docstring) - a
        # tiny minimum floor guards against a degenerate (zero-extent)
        # axis, e.g. a 1D scan.
        x_range = max(np.nanmax(x) - np.nanmin(x), 1e-9)
        y_range = max(np.nanmax(y) - np.nanmin(y), 1e-9)
        ax.set_box_aspect((x_range, y_range, max(x_range, y_range)))

        # Match the BF images' orientation - see _export_prescan_points.
        ax.invert_xaxis()
        ax.invert_yaxis()

        # Extra pad, and its own title rather than relying on default
        # placement: fig.colorbar()'s default pad sits right where
        # mpl3d draws the z-axis and its tick labels, so without this
        # the colorbar visually overlaps/hides the z-axis labeling.
        colorbar = fig.colorbar(surf, shrink=0.6, pad=0.15)
        colorbar.ax.set_title('$\\Delta z$ [$\\mu$m]', fontsize='small')
        fig.tight_layout()
        fig.savefig(filename, bbox_inches='tight')

    @staticmethod
    def _compute_metrics(data, sampled_projection):
        metrics = {}

        found_mask = data.get('surface_found_mask')
        roi_mask = data.get('roi_scan_plan_mask')
        if found_mask is not None and found_mask.size:
            # A grid point outside the drawn ROI is *never* considered
            # by the surface pre-scan at all - it isn't "surface not
            # found", it was simply never in scope. Mixed into an
            # unrestricted fraction, a small ROI on a large grid makes
            # coverage look far worse than it is. So: report both the
            # full-grid fractions (name unchanged, for continuity) and
            # the same fractions restricted to the drawn ROI, which is
            # what "was the surface found/used" usually really means.
            roi_bool = roi_mask.astype(bool) if roi_mask is not None \
                else np.ones_like(found_mask, dtype=bool)

            def _fractions(mask, prefix):
                total = mask.size
                if not total:
                    return
                # Directly, independently measured at that point.
                metrics[f'{prefix}surface_found_fraction'] = \
                    float(np.sum(mask == 1)) / total
                # Filled in by the acquisition software from
                # neighbors - still used to center the actual scan,
                # but not itself an independent measurement.
                metrics[f'{prefix}surface_interpolated_fraction'] = \
                    float(np.sum(mask == 2)) / total
                # No surface info at all - the scan used a nominal,
                # uncorrected z there instead.
                metrics[f'{prefix}surface_missing_fraction'] = \
                    float(np.sum(mask == 0)) / total
                # found==1 or found==2: every point the acquisition
                # had *some* surface estimate for and actually used,
                # as opposed to surface_found_fraction which is only
                # the subset that was independently measured. This is
                # often what "was the surface followed here" means.
                metrics[f'{prefix}surface_used_fraction'] = \
                    float(np.sum(mask > 0)) / total

            _fractions(found_mask, '')
            if np.any(roi_bool):
                _fractions(found_mask[roi_bool], 'roi_')

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
