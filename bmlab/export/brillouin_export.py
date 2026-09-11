import csv
import logging
import os
import warnings
import numpy as np

from bmlab import Session

logger = logging.getLogger(__name__)


class BrillouinExport(object):

    def __init__(self, evc):
        self.session = Session.get_instance()
        self.file = self.session.file
        self.mode = 'Brillouin'
        self.evc = evc
        return

    def export(self, configuration):
        if not self.file:
            return

        config = configuration['brillouin']
        if not config['export']:
            return

        brillouin_repetitions = self.file.repetition_keys()
        selected_repetitions = config.get('repetitions')
        if selected_repetitions is not None:
            brillouin_repetitions = [
                repetition for repetition in brillouin_repetitions
                if repetition in selected_repetitions]

        for brillouin_repetition in brillouin_repetitions:
            self.session.set_current_repetition(brillouin_repetition)

            # Get a list of all parameters available
            parameters = self.session.evaluation_model().get_parameter_keys()
            nr_brillouin_peaks =\
                self.session.evaluation_model().nr_brillouin_peaks

            # dimensionality (like positions) doesn't depend on which
            # parameter we ask for, so a single call tells us whether
            # this repetition has a valid measurement grid at all (it
            # is None for an aborted/restarted acquisition that never
            # wrote any positions) - skip it entirely if not, instead
            # of re-discovering that once per parameter below.
            _, _, dimensionality, _ =\
                self.evc.get_data(next(iter(parameters)), 0)
            if dimensionality is None or dimensionality < 2:
                continue

            self._export_combined_csv(
                brillouin_repetition, parameters, nr_brillouin_peaks)

    def _get_parameter_data(self, parameter_key, brillouin_peak_index, raw):
        """
        Like `self.evc.get_data()`, but also works for 'rayleigh_shift' -
        a derived results key that (unlike every key in
        get_parameter_keys()) was never registered in
        EvaluationModel.parameters, so EvaluationController.get_data()
        cannot look up its unit scaling and raises a KeyError for it.
        `raw` is the already-fetched evm.results[parameter_key].
        """
        evm = self.session.evaluation_model()
        if parameter_key in evm.parameters:
            data, positions, _, _ = self.evc.get_data(
                parameter_key, brillouin_peak_index)
            return data, positions

        positions = list(self.session.get_payload_positions().values())
        sliced = raw[:, :, :, :, :, 0] if raw.ndim >= 6 else raw
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore', message='Mean of empty slice')
            data = np.nanmean(sliced, axis=tuple(range(3, sliced.ndim)))
        # rayleigh_shift is a difference of two rayleigh_peak_position_f
        # values, so it shares that key's Hz -> GHz scaling.
        data = evm.parameters['rayleigh_peak_position_f']['scaling'] * data
        return data, positions

    def _export_combined_csv(
            self, brillouin_repetition, parameters, nr_brillouin_peaks):
        """
        Writes every evaluated quantity for `brillouin_repetition` into a
        single CSV, one row per actually-measured grid point (points
        outside the ROI - or otherwise never fit, NaN in every column -
        are dropped, not written as empty rows) - replacing the old
        one-file-per-parameter-per-slice layout. Columns are the raw,
        un-centered stage position (x, y, z, in um - the same coordinate
        system the transform matrix written by OverviewBrightfieldExport
        maps into image pixels, so the two files agree on what "x, y"
        means) followed by every key in `parameters` (plus the derived
        'rayleigh_shift'), split into one column per Brillouin peak fit
        only for the keys that actually vary by peak.

        Rows are raveled 'F' (x fastest, then y, then z), matching each
        point's fixed grid address - the same x-fastest linear-index
        formula both EvaluationController.get_indices_from_key() and
        BrillouinAcquisition's own H5BM::calculateIndex() use to key a
        point's spectrum, independent of when it was actually captured.
        This is NOT necessarily literal chronological acquisition order:
        the physical scan path (BrillouinAcquisition's ScanPlanner /
        m_orderedPositions) can visit grid points in any sequence - e.g.
        a snake pattern - while each spectrum still files under its
        fixed x-fastest address regardless of when it was captured. Row
        order here is a deterministic spatial (grid-address) order, not
        a timeline; the 'time' column is each point's real timestamp if
        chronological order is what's actually wanted.
        """
        evm = self.session.evaluation_model()

        columns = []
        x = y = z = None
        for parameter_key in list(parameters) + ['rayleigh_shift']:
            raw = evm.results.get(parameter_key)
            # 'rayleigh_shift' is a derived, migration-only key: older
            # sessions that were never re-evaluated after it was
            # introduced won't have it, unlike the keys in `parameters`
            # (always present, see EvaluationModel.__init__).
            if raw is None:
                continue
            nr_peaks_stored = raw.shape[5] if raw.ndim >= 6 else 1

            if nr_brillouin_peaks > 1 and nr_peaks_stored > 1:
                peak_variants = [(0, '_peak-single')]
                peak_variants += [(i, f'_peak-{i}')
                                  for i in range(1, nr_brillouin_peaks + 1)]
                peak_variants += [
                    (nr_brillouin_peaks + 1, '_peak-average'),
                    (nr_brillouin_peaks + 2, '_peak-average-weighted'),
                ]
            else:
                peak_variants = [(0, '')]

            for brillouin_peak_index, postfix in peak_variants:
                data, positions = self._get_parameter_data(
                    parameter_key, brillouin_peak_index, raw)
                if x is None:
                    # Positions don't depend on parameter_key/peak_index -
                    # grab the (un-centered, absolute stage um) grid once.
                    x, y, z = positions
                if data.size != x.size:
                    # A repetition whose results were never (re-)computed
                    # for this key at the current grid size - e.g. never
                    # evaluated at all (still EvaluationModel.__init__'s
                    # np.empty((0,)) placeholder - normally caught by
                    # EvaluationController.get_data()'s own size==0 guard,
                    # but 'rayleigh_shift' bypasses get_data() entirely,
                    # see _get_parameter_data()), or evaluated with a
                    # region/peak count of 0 (nr_brillouin_regions=0 makes
                    # brillouin_peak_position_f's own shape - and anything
                    # derived from it - collapse to size 0 too). Either
                    # way there's nothing real for this column at this
                    # repetition's points - fill it with NaN at the
                    # correct shape instead of crashing when it's later
                    # zipped against the (real, grid-sized) position
                    # columns.
                    logger.warning(
                        "%s repetition %s: '%s%s' has %d values, expected "
                        "%d (grid size) - exporting it as all-NaN",
                        self.file.path, brillouin_repetition, parameter_key,
                        postfix, data.size, x.size)
                    data = np.full(x.shape, np.nan)
                # 'F' (x fastest, then y, then z) matches each point's
                # fixed grid address (see this method's docstring) -
                # NOT necessarily the real chronological capture order.
                columns.append(
                    (f"{parameter_key}{postfix}", data.ravel(order='F')))

        if self.file.path.parent.name == 'RawData':
            csv_path = self.file.path.parents[1] / 'Export'
        else:
            csv_path = self.file.path.parent
        if not os.path.exists(csv_path):
            os.makedirs(csv_path, exist_ok=True)
        csv_filename = csv_path / \
            f"{self.file.path.stem}_BMrep{brillouin_repetition}_data.csv"

        scale_calibration = self.file.get_repetition(
            brillouin_repetition).payload.get_scale_calibration()

        # Grid points outside the measured ROI have no image key at all,
        # so EvaluationController.evaluate()'s per-point loop never
        # visits their index and every column - including 'time' and
        # 'intensity', which are set directly from the raw spectrum,
        # before any peak fitting - stays at its NaN fill value. A
        # point that WAS measured but whose fit genuinely failed only
        # has NaN peak-fit columns; 'time'/'intensity' are still set,
        # and that row must still show up (with NaN fit results) rather
        # than being dropped like an unmeasured one. So "measured" is
        # defined via those two raw columns specifically, not via "any
        # column is non-NaN" (which would incorrectly keep no row
        # distinction at all between the two cases).
        measured_columns = [col for name, col in columns
                            if name in ('time', 'intensity')]
        if measured_columns:
            valid = np.zeros(x.size, dtype=bool)
            for col in measured_columns:
                valid |= ~np.isnan(col)
        else:
            # 'time'/'intensity' weren't selected for export for some
            # reason - fall back to "at least one column has a value".
            valid = np.zeros(x.size, dtype=bool)
            for _, col in columns:
                valid |= ~np.isnan(col)

        # Coordinate system (not written into the CSV itself - kept
        # here so it's clear without cluttering the file): x, y, z are
        # the absolute stage position (um) each point was measured at,
        # exactly as BrillouinAcquisition wrote it (positions-x/y/z) -
        # raw, not centered/rotated on the measurement grid. A
        # '<image>_transform.csv' written by OverviewBrightfieldExport
        # next to an overview image maps these same (x, y) values onto
        # that image's own pixel coordinates:
        # [col, row, 1] = M @ [x, y, 1].
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['#source_file', self.file.path.name])
            writer.writerow(['#file_version',
                             getattr(self.file, 'file_version_string', None)])
            writer.writerow(['#repetition', brillouin_repetition])
            writer.writerow(['#nr_brillouin_peaks', nr_brillouin_peaks])
            writer.writerow(['#dim_x', x.shape[0]])
            writer.writerow(['#dim_y', x.shape[1]])
            writer.writerow(['#dim_z', x.shape[2]])
            writer.writerow(['#points_measured', int(valid.sum())])
            writer.writerow(['#date', self.file.date])
            comment = getattr(self.file, 'comment', None)
            if comment is not None:
                writer.writerow(['#comment', comment])

            setup = self.session.setup
            if setup is not None:
                writer.writerow(['#setup', setup.name])
                writer.writerow(['#setup_temperature_K', setup.temperature])
            orientation = self.session.orientation
            # Only affects how raw camera images are reoriented for
            # extraction/calibration - the x, y, z grid above comes
            # straight from the stage positions and is unaffected by it.
            writer.writerow(
                ['#image_orientation_rotation', orientation.rotation])
            writer.writerow([
                '#image_orientation_reflection_vertically',
                orientation.reflection['vertically']])
            writer.writerow([
                '#image_orientation_reflection_horizontally',
                orientation.reflection['horizontally']])

            psm = self.session.peak_selection_model()
            if psm is not None:
                for i, region in enumerate(psm.get_brillouin_regions()):
                    writer.writerow(
                        [f'#brillouin_region_{i}_hz', f'{region}'])
                for i, region in enumerate(psm.get_rayleigh_regions()):
                    writer.writerow(
                        [f'#rayleigh_region_{i}_hz', f'{region}'])

            # The quality thresholds behind the 'quality_pass' column
            # (see EvaluationController.apply_quality_thresholds()) -
            # written in the same display units (e.g. GHz) get_data()
            # returns, matching that column and every quality metric
            # column's own values. A metric with no row here had no
            # threshold set at export time.
            for metric_key, threshold in evm.quality_thresholds.items():
                writer.writerow(
                    [f'#quality_threshold_{metric_key}_enabled',
                     threshold.get('enabled', True)])
                writer.writerow(
                    [f'#quality_threshold_{metric_key}_min',
                     threshold.get('min')])
                writer.writerow(
                    [f'#quality_threshold_{metric_key}_max',
                     threshold.get('max')])

            if scale_calibration is not None:
                for key, value in scale_calibration.items():
                    writer.writerow([f'#{key}_x', value[0]])
                    writer.writerow([f'#{key}_y', value[1]])

            header = ['x', 'y', 'z'] + [name for name, _ in columns]
            writer.writerow(header)

            x_flat, y_flat, z_flat = (
                x.ravel(order='F')[valid], y.ravel(order='F')[valid],
                z.ravel(order='F')[valid])
            rows = zip(x_flat, y_flat, z_flat,
                       *(col[valid] for _, col in columns))
            for row in rows:
                writer.writerow(row)
