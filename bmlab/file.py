"""
Module for interacting with files containing Brillouin microscopy
data.

NOTE: Ideally, the users of bmlab should not have to know about
the file format in which the data are stored. So, if possible,
do not expose HDF objects to the outside (like BMicro).
"""

import datetime
import re

import numpy as np
import h5py
from pathlib import Path

from packaging import version

BRILLOUIN_GROUP = 'Brillouin'
FLUORESCENCE_GROUP = 'Fluorescence'

# Channel name of the overview of the measuring objective that is taken
# after every z plane. Like the other overviews it is stored as a regular
# Fluorescence-mode image, distinguished only by this channel attribute
# (no dedicated HDF5 group exists for them), and carries the same
# attributes (see OVERVIEW_EVENT_CHANNEL).
OVERVIEW_BRIGHTFIELD_CHANNEL = 'Brightfield z overview'

# Channel name of the other brightfield overviews of the overview engine
# (files of older BrillouinAcquisition versions are converted to it by
# tools/migrate_legacy_overviews.py): one "event" is taken before
# the measurement, after every z plane and/or after the measurement,
# with one or several objectives; every image carries its own metadata
# (overview_event, overview_kind_name, overview_tile_index,
# overview_stack_index, grid_offset_x_um, image_centre_shift_x_um, ...)
# and its own scale calibration, see MeasurementData.get_overview_info().
OVERVIEW_EVENT_CHANNEL = 'Brightfield overview'

# All overview images; everything that treats "the overviews" (export,
# exclusion from the plain Fluorescence export) uses this.
OVERVIEW_CHANNELS = (OVERVIEW_BRIGHTFIELD_CHANNEL, OVERVIEW_EVENT_CHANNEL)

# Channel name for a plain single-shot brightfield snapshot, as taken
# e.g. right before a scan starts (its own single-image Fluorescence
# repetition, not an overview event) - exported by FluorescenceExport
# with a .tif and transform matrix like the other snapshots.
BRIGHTFIELD_CHANNEL = 'Brightfield'

# Channel name for a brightfield image captured alongside an
# individual Brillouin measurement point (one per point, or one per
# every Nth point - see Brillouin::capturePerPointBrightfieldImage()),
# as opposed to the per-z-plane overviews (OVERVIEW_CHANNELS). Both its position and stage-position attributes are
# already converted into the same grid frame as the Brillouin
# measurement grid's own positions-x/y/z, exactly like the overview
# channels above.
PER_POINT_BRIGHTFIELD_CHANNEL = 'Brightfield per-point'


def _get_text_attribute(group, name):
    """A text attribute as written by BrillouinAcquisition (a one-element
    array of bytes) as str, None if it is missing."""
    try:
        value = group.attrs.get(name)
        if value is None:
            return None
        value = np.ravel(value)[0]
        return value.decode('utf-8') if hasattr(value, 'decode')             else str(value)
    except BaseException:
        return None


def _get_datetime(time_stamp):
    """ Convert the time stamp in the HDF file to Python datetime """
    try:
        return datetime.datetime.fromisoformat(time_stamp)
    except Exception:
        return None


def is_source_file(path):
    try:
        with h5py.File(path, "r") as h5:
            file_format = h5.attrs.get('version')
            if file_format is not None:
                # Is this a file from BrillouinAcquisition?
                # Then we get an ndarray with a bytes object inside
                # (because the version attribute is stored
                # as 'cset=H5T_CSET_ASCII').
                if isinstance(file_format, np.ndarray)\
                        and isinstance(file_format[0], bytes):
                    file_format = file_format[0].decode('ascii')
                # This is from BrillouinAcquisition!
                if file_format.startswith('H5BM'):
                    return True
        return False
    except Exception:
        return False


def is_session_file(path):
    try:
        path = Path(path)
        with h5py.File(path, "r") as h5:
            file_format = h5.attrs.get('version')
            if file_format is not None:
                # This is a bmlab session file created
                # with bmlab>=0.0.14
                if isinstance(file_format, str)\
                        and file_format.startswith('bmlab'):
                    return True

            # This is a bmlab session file created with bmlab<0.0.14
            if 'session' in h5.keys()\
                    and h5['session'].attrs.get('type').startswith('bmlab'):
                return True
        return False
    except Exception:
        return False


class BrillouinFile(object):

    def __init__(self, path):
        """
        Load a HDF file with Brillouin microscopy data.

        Parameters
        ----------
        path : Path
            path of the file to load

        Raises
        ------
        OSError
            when trying to open non-existing or bad file
        """
        self.path = Path(path).resolve()
        self.file = None
        # Repetitions are built once per file; building one reads its
        # positions and reconstructs the surface-follow z values, which
        # is far too slow to repeat on every access.
        self._repetition_cache = {}
        self.file = h5py.File(self.path, 'r')
        self.file_version_string = self.file.attrs.get('version')[
            0].decode('utf-8')
        if not self.file_version_string.startswith('H5BM'):
            raise BadFileException('File does not contain any Brillouin data')
        self.file_version = self.file_version_string[-5:]
        if version.parse(self.file_version) < version.parse("0.0.4"):
            raise BadFileException(
                'File format %s is not supported (H5BM-v0.0.4 or newer '
                'is)' % self.file_version_string)
        if BRILLOUIN_GROUP not in self.file:
            raise BadFileException('File does not contain any Brillouin data')
        self.Brillouin_group = self.file[BRILLOUIN_GROUP]
        # Fluorescence group is optional
        self.Fluorescence_group = None
        if FLUORESCENCE_GROUP in self.file:
            self.Fluorescence_group = self.file[FLUORESCENCE_GROUP]
        # Comments are optional,
        # e.g. for files containing only fluorescence data
        comment = self.file.attrs.get('comment')
        if comment is not None:
            self.comment = comment[0].decode('utf-8')
        self.date = _get_datetime(
            self.file.attrs.get('date')[0].decode('utf-8'))

    def __del__(self):
        """
        Destructor. Closes hdf file when object runs out of scope.
        """
        self.close()

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass

    @staticmethod
    def checkMode(mode):
        modes = [BRILLOUIN_GROUP, FLUORESCENCE_GROUP]
        if mode not in modes:
            raise NotImplementedError(
                'The mode "{}" is not supported.'.format(mode))

    def repetition_count(self, mode=BRILLOUIN_GROUP):
        """
        Get the number of repetitions in the data file.

        Returns
        -------
        out : int
            Number of repetitions in the data file
        """
        self.checkMode(mode)
        return len(self.repetition_keys(mode))

    def repetition_keys(self, mode=BRILLOUIN_GROUP):
        """
        Returns list of keys for the various repetitions in the file.

        Returns
        -------
        out: list of str
        """
        self.checkMode(mode)
        mode_group = self.Brillouin_group
        if mode == FLUORESCENCE_GROUP:
            mode_group = self.Fluorescence_group
        if mode_group is None:
            return []
        return list(mode_group.keys())

    def get_repetition(self, repetition_key, mode=BRILLOUIN_GROUP):
        """
        Get a repetition from the data file based on given key.

        Parameters
        ----------
        repetition_key : str
            key to identify the repetition in the Brillouin group
        mode: str
            the mode to look at

        Returns
        -------
        out : Repetition
            the repetition
        """
        self.checkMode(mode)
        mode_group = self.Brillouin_group
        if mode == FLUORESCENCE_GROUP:
            mode_group = self.Fluorescence_group
        # Check that we have a group for this mode
        if mode_group is None:
            return None
        cache_key = (mode, repetition_key)
        repetition = self._repetition_cache.get(cache_key)
        if repetition is None:
            repetition = Repetition(mode_group.get(repetition_key), self)
            self._repetition_cache[cache_key] = repetition
        return repetition


class Repetition(object):

    def __init__(self, repetition_group, file):
        """
        Creates a repetition from the corresponding group of a HDF file.

        Parameters
        ----------
        repetition_group : HDF group
            The HDF group representing a Repetition. Consists of payload,
            calibration and background.
        """
        self.date = _get_datetime(
            repetition_group.attrs.get('date')[0].decode('utf-8'))
        # How the repetition ended, as written by BrillouinAcquisition
        # versions with Continue/resume: 'preparing', 'surface-scan',
        # 'surface-scan-stopped', 'measuring', 'stopped' or 'finished'.
        # None for files that predate it.
        self.acquisition_state = _get_text_attribute(
            repetition_group, 'acquisition_state')
        self.acquisition_state_time = _get_text_attribute(
            repetition_group, 'acquisition_state_time')
        self.payload = Payload(repetition_group.get('payload'), self)
        calibration_group = repetition_group.get('calibration')
        # Having a calibration is optional for a repetition
        if calibration_group is not None:
            self.calibration = Calibration(calibration_group, self)
        background_group = repetition_group.get('background')
        # Having background reference points is optional for a repetition -
        # only present when useBackgroundRoiMask was enabled for that run.
        if background_group is not None:
            self.background = Background(background_group, self)
        self.file = file


class MeasurementData(object):

    def __init__(self, payload_group, repetition):
        """
        Creates a payload representation from the corresponding group of a
        HDF file.

        Parameters
        ----------
        payload_group : HDF group
            The payload of a repetition, basically a set of images

        """
        self.repetition = repetition
        self.group = payload_group
        if payload_group is not None:
            self.data = payload_group.get('data')
        else:
            self.data = None

    def image_keys(self, sort_by_time=False):
        """
        Returns the keys of the images stored in the payload,
        optionally sorted by time.

        Parameters
        ----------
        sort_by_time : bool

        Returns
        -------
        out: list of str
            Keys of images in payload.
        """
        if self.data:
            keys = list(self.data.keys())
            if not sort_by_time:
                keys_int = [int(key) for key in keys]
                return [i for _, i in sorted(zip(keys_int, keys))]

            dates = []
            for key in keys:
                dates.append(self.get_date(key))

            return [i for _, i in sorted(zip(dates, keys))]
        return []

    def get_image(self, image_key):
        """
        Returns the image from the calibration for given key.

        Parameters
        ----------
        image_key: str
            Key for the image.

        Returns
        -------
        out: numpy.ndarray
            Array representing the image.
        """
        if self.data is None:
            return None
        imgs = self.data.get(image_key)
        if imgs is None:
            return None
        # Newer BrillouinAcquisition files may pad a point's frame stack
        # with unused trailing slots (adaptive frame extension: the frame
        # axis is sized for camera.frameCount + maxAdditionalFrames for
        # every point, regardless of how many frames were actually taken
        # at this point). The `frames_actual` attribute, when present,
        # records how many leading frames along that axis are real; we
        # trim the padded/ragged trailing slots here, before any caller
        # (e.g. ImageController.extract_spectra) ever sees the array, so
        # downstream frame counts (len(imgs), np.arange(len(imgs)), ...)
        # reflect only the real frames. Older files, or files where the
        # feature wasn't used, have no such attribute and are returned
        # unchanged.
        try:
            frames_actual = int(imgs.attrs.get('frames_actual')[0])
        except Exception:
            frames_actual = None
        array = np.array(imgs)
        if frames_actual is not None and 0 <= frames_actual < array.shape[0]:
            array = array[:frames_actual]
        return array

    def get_image_count(self, image_key):
        if self.data is None:
            return 0
        imgs = self.data.get(image_key)
        if imgs is None:
            return 0
        try:
            frames_actual = int(imgs.attrs.get('frames_actual')[0])
        except Exception:
            frames_actual = None
        if frames_actual is not None and 0 <= frames_actual < imgs.shape[0]:
            return frames_actual
        return imgs.shape[0]

    def get_date(self, image_key):
        """"
        Returns the date of an image
        with the given key
        """
        try:
            return _get_datetime(
                self.data.get(image_key).attrs.get('date')[0].decode('utf-8'))
        except Exception:
            return ''

    def get_time(self, image_key):
        try:
            # Get date of the calibration
            date = self.get_date(image_key)
            # Get the reference date
            ref = self.repetition.file.date
            # return the difference in seconds
            return (date - ref).total_seconds()
        except Exception:
            return None

    def is_empty(self):
        return self.data is None or len(self.data) == 0

    def get_exposure(self, image_key):
        """"
        Returns the exposure time of a payload image
        with the given key
        """
        try:
            return self.data.get(image_key).attrs\
                .get('exposure')[0]
        except Exception:
            # the test fixtures of the first file format version have none
            return 0.5

    def get_binning(self, image_key):
        """"
        Returns the binning of a payload image
        with the given key
        """
        try:
            return self.data.get(image_key).attrs\
                .get('binning')[0].decode('utf-8')
        except Exception:
            # the test fixtures of the first file format version have none
            return '1x1'

    def get_binning_factor(self, image_key):
        binning = self.get_binning(image_key)
        # We only check the values possible
        # with BrillouinAcquisition
        if binning == '2x2':
            return 2
        elif binning == '4x4':
            return 4
        elif binning == '8x8':
            return 8
        return 1

    def get_channel(self, image_key):
        """"
        Returns the channel of a payload image
        with the given key
        """
        try:
            return self.data.get(image_key).attrs\
                .get('channel')[0].decode('utf-8')
        except Exception:
            return None

    def get_brillouin_repetition_index(self, image_key):
        """
        Returns the Brillouin repetition this Fluorescence-group image
        (a per-z overview or per-point brightfield capture - see
        OVERVIEW_BRIGHTFIELD_CHANNEL/PER_POINT_BRIGHTFIELD_CHANNEL) was
        captured as part of, or None if it has none - either an older
        file that predates this attribute, or a genuinely standalone
        Fluorescence-tab "Acquire" capture, which was never part of any
        Brillouin measurement to begin with.
        """
        try:
            return int(self.data.get(image_key).attrs
                       .get('brillouin_repetition_index')[0])
        except Exception:
            return None

    def image_keys_by_channel(self, channel, sort_by_time=False):
        """
        Returns the keys of the images in the payload whose channel
        attribute matches the given value, optionally sorted by time.

        Parameters
        ----------
        channel : str
        sort_by_time : bool

        Returns
        -------
        out: list of str
        """
        keys = self.image_keys(sort_by_time=sort_by_time)
        return [key for key in keys if self.get_channel(key) == channel]

    def get_class(self, image_key):
        """"
        Returns the image class of a payload image
        with the given key
        """
        try:
            return self.data.get(image_key).attrs\
                .get('IMAGE_SUBCLASS')[0].decode('utf-8')
        except Exception:
            return None

    def get_mode(self, image_key):
        """"
        Returns the interlace mode of a payload image
        with the given key
        """
        try:
            return self.data.get(image_key).attrs\
                .get('INTERLACE_MODE')[0].decode('utf-8')
        except Exception:
            return None

    def get_ROI(self, image_key):
        """
        Returns the region of interest of a payload image
        with the given key
        Parameters
        ----------
        image_key

        Returns
        -------

        """
        # Older measurements didn't save this information,
        # so we wrap it in a try/except
        try:
            attributes = ['left', 'right',
                          'bottom', 'top',
                          'width_physical', 'height_physical',
                          'width_binned', 'height_binned']
            roi = dict()
            for attribute in attributes:
                roi[attribute] =\
                    self.data.get(image_key).attrs.get('ROI_' + attribute)[0]

            return roi
        except BaseException:
            return None

    def get_position(self, image_key):
        """
        Returns the xyz position (in µm) that a Fluorescence-group image
        was targeted at, or None if not recorded (older files, or an
        image outside the Fluorescence group).

        For regular Fluorescence-mode images this is the actual capture
        position (single-point capture, no separate stage position is
        recorded). For brightfield overview images it is the *intended*
        target, which can differ slightly from the actual stage position
        - see get_stage_position().
        """
        try:
            attrs = self.data.get(image_key).attrs
            return {
                axis: attrs.get('position_' + axis + '_um')[0]
                for axis in ('x', 'y', 'z')
            }
        except BaseException:
            return None

    def get_stage_position(self, image_key):
        """
        Returns the actual stage xyz position (in µm) read back at
        capture time, or None if not recorded. Only present for
        brightfield overview images, where the stage moves away and
        back and so can differ slightly from the target position
        returned by get_position() due to hysteresis compensation and
        backlash.
        """
        try:
            attrs = self.data.get(image_key).attrs
            return {
                axis: attrs.get('stage_position_' + axis + '_um')[0]
                for axis in ('x', 'y', 'z')
            }
        except BaseException:
            return None

    # The kinds of overview moments (OVERVIEW_MOMENT_KIND in
    # BrillouinAcquisition); the number is stored as 'overview_kind'.
    OVERVIEW_KIND_NAMES = {
        0: 'before measurement', 1: 'after z plane',
        2: 'after repetition', 3: 'after measurement'}

    def _attribute(self, image_key, name):
        """One attribute of an image as float or str; None if missing."""
        try:
            value = self.data.get(image_key).attrs.get(name)
            if value is None:
                return None
            value = np.ravel(value)[0]
            if isinstance(value, (bytes, np.bytes_)):
                return value.decode('utf-8')
            return float(value)
        except BaseException:
            return None

    def _attribute_pair(self, image_key, name_x, name_y, default=None):
        x = self._attribute(image_key, name_x)
        y = self._attribute(image_key, name_y)
        if x is None or y is None:
            return default
        return (x, y)

    def get_overview_info(self, image_key):
        """
        Returns what the overview engine of BrillouinAcquisition wrote
        with an image of the OVERVIEW_EVENT_CHANNEL, or None if the image
        has none of it (any other channel, older files).

        Returns
        -------
        out: dict
            'event': number of the overview event (one event = one round
                of overviews, with one or several objectives),
            'kind', 'kind_name': before measurement / after z plane /
                after repetition / after measurement,
            'z_index', 'point_index' (-1 before the first point),
            'objective_slot', 'objective_name', 'magnification',
            'is_measurement_objective', 'tile_index', 'tile_count',
            'stack_index', 'stack_count', 'is_centre_slice',
            'seconds_since_start', 'repetition' (key of the Brillouin
            repetition it was taken during),
            'position', 'stage_position': dicts with x, y, z in um in the
                frame of the Brillouin positions-x/y/z (the target of the
                LASER, shifts below included),
            'grid_offset': (x, y), the drift offset the grid had been
                moved by when the image was taken (0 without drift
                correction), 'fov_compensation' and
                'image_centre_shift': (x, y), see below,
            'scale_calibration': the scale calibration of the objective
                that took the image (the keys needed to place it),
            'anchor': dict with x, y, z: the CSV coordinate (planned grid
                position, drift offset taken off) of the middle of the
                image.

        The stored target is not the middle of the image: the overview
        engine puts the centre of gravity of the grid points there and
        moves the target by the scanner offset, the position of the
        image centre against the calibration origin (image_centre_shift)
        and the FOV offset of the objective against the measuring one
        (fov_compensation), and the drift offset is added to every stage
        target. The anchor takes all of that off again (checked against
        the centre of gravity of the measured points: within 0.1 um).
        """
        event = self._attribute(image_key, 'overview_event')
        if event is None:
            return None

        def number(name, default=None):
            value = self._attribute(image_key, name)
            return default if value is None else value

        position = self.get_position(image_key)
        stage_position = self.get_stage_position(image_key)
        grid_offset = self._attribute_pair(
            image_key, 'grid_offset_x_um', 'grid_offset_y_um', (0.0, 0.0))
        fov = self._attribute_pair(
            image_key, 'fov_compensation_x_um', 'fov_compensation_y_um',
            (0.0, 0.0))
        shift = self._attribute_pair(
            image_key, 'image_centre_shift_x_um', 'image_centre_shift_y_um',
            (0.0, 0.0))
        # where the lens was: the read-back position if there is one
        measured = stage_position or position
        anchor = None
        if measured is not None:
            anchor = {
                'x': measured['x'] - shift[0] - fov[0] - grid_offset[0],
                'y': measured['y'] - shift[1] - fov[1] - grid_offset[1],
                # (the plane it was aimed at: stable for grouping the
                # pages of a mosaic, unlike the read-back)
                'z': (position or measured)['z']}

        scale = None
        x_pair = self._attribute_pair(
            image_key, 'scale_micrometer_to_pix_x_col',
            'scale_micrometer_to_pix_x_row')
        y_pair = self._attribute_pair(
            image_key, 'scale_micrometer_to_pix_y_col',
            'scale_micrometer_to_pix_y_row')
        if x_pair is not None and y_pair is not None:
            scale = {
                'micrometerToPixX': x_pair,
                'micrometerToPixY': y_pair,
                'origin': self._attribute_pair(
                    image_key, 'scale_origin_col', 'scale_origin_row'),
                'objectiveName': self._attribute(image_key, 'objective_name'),
                'magnification': number('objective_magnification')}

        kind = number('overview_kind')
        kind = None if kind is None else int(kind)
        return {
            'event': int(event),
            'kind': kind,
            'kind_name': self._attribute(image_key, 'overview_kind_name')
            or self.OVERVIEW_KIND_NAMES.get(kind),
            'z_index': int(number('overview_z_index', 0)),
            'point_index': int(number('overview_point_index', -1)),
            'objective_slot': int(number('overview_objective_slot', -1)),
            'objective_name': self._attribute(image_key, 'objective_name'),
            'magnification': number('objective_magnification'),
            'is_measurement_objective': bool(
                number('overview_is_measurement_objective', 0)),
            'tile_index': int(number('overview_tile_index', 0)),
            'tile_count': int(number('overview_tile_count', 1)),
            'stack_index': int(number('overview_stack_index', 0)),
            'stack_count': int(number('overview_stack_count', 1)),
            'is_centre_slice': bool(number('overview_is_centre_slice', 0)),
            'seconds_since_start': number('overview_seconds_since_start'),
            'repetition': self.get_brillouin_repetition_index(image_key),
            'position': position,
            'stage_position': stage_position,
            'grid_offset': grid_offset,
            'fov_compensation': fov,
            'image_centre_shift': shift,
            'scale_calibration': scale,
            'anchor': anchor,
        }


class Payload(MeasurementData):

    # Datasets written by BrillouinAcquisition's surface-following scan
    # (positions-<name> under the Brillouin payload group). Masks use
    # the C-order shape they were written with, so a mask array can be
    # indexed directly as mask[x_index, y_index] (2D masks) or
    # mask[z_index, x_index, y_index] (3D masks) without transposing.
    SURFACE_SCAN_MASKS = {
        'surface_found_mask': 'positions-surface-found-mask',
        'sampled_mask': 'positions-sampled-mask',
        'roi_scan_plan_mask': 'positions-roi-scan-plan-mask',
    }
    SURFACE_SCAN_PATH = {
        'sampled_x': 'positions-sampled-x',
        'sampled_y': 'positions-sampled-y',
        'sampled_z': 'positions-sampled-z',
        # The ROI polygon drawn at acquisition time (stage um),
        # independent of its grid-discretized 'roi_scan_plan_mask'.
        'roi_polygon_x': 'positions-roi-polygon-x-um',
        'roi_polygon_y': 'positions-roi-polygon-y-um',
        # The coarse (binned) x/y sample grid runSurfacePreScan() itself
        # actually probed, before any interpolation onto the dense
        # scan-plan grid - the genuinely sparse "real coordinates used
        # in coarse-grain mode to find a surface". Only present on
        # files saved by a BrillouinAcquisition build that records this
        # (see Brillouin::m_surfacePreScanXUm/YUm) - None otherwise, in
        # which case no coarse pre-scan plot should be produced (the
        # dense surface-found-mask is not a substitute - it lives at
        # the dense grid's own resolution, not the coarse pre-scan's).
        'prescan_x': 'positions-surface-prescan-x-um',
        'prescan_y': 'positions-surface-prescan-y-um',
    }
    # 2D, shape (len(prescan_x), len(prescan_y)) - same 0/1/2 "found"
    # convention as SURFACE_SCAN_MASKS['surface_found_mask'], at the
    # coarse pre-scan's own resolution.
    SURFACE_PRESCAN_MASKS = {
        'prescan_found_mask': 'positions-surface-prescan-found-mask',
        'prescan_z': 'positions-surface-prescan-z-um',
        'prescan_metric': 'positions-surface-prescan-metric',
    }
    # The full (z, metric) drop curve behind each pre-scan column,
    # not just prescan_z/prescan_metric's single found/last-measured
    # value - every sample actually taken while searching that column
    # (seed, rewind, forward walk, verification), in measurement
    # order. 'curve_z'/'curve_metric' are rank-3, shape (len(
    # prescan_x), len(prescan_y), max_samples), NaN-padded past each
    # column's own 'curve_sample_counts' (a column's real samples
    # always come first). Only present alongside prescan_x/prescan_y,
    # and only if at least one sample was ever recorded.
    SURFACE_SCAN_CURVES = {
        'curve_z': 'positions-surface-curve-z-um',
        'curve_metric': 'positions-surface-curve-metric',
        'curve_sample_counts': 'positions-surface-curve-sample-counts',
    }
    # Extra boundary points that found a surface, requested outside
    # the regular coarse pre-scan grid - a flat list (not a grid,
    # unlike SURFACE_PRESCAN_MASKS), in the same grid frame as
    # roi_polygon_x/y. Absent entirely if none were requested or none
    # of the requested ones found a surface.
    SURFACE_SCAN_BOUNDARY = {
        'boundary_x': 'positions-surface-prescan-boundary-x-um',
        'boundary_y': 'positions-surface-prescan-boundary-y-um',
        'boundary_z': 'positions-surface-prescan-boundary-z-um',
    }
    SURFACE_SCAN_SETTINGS = {
        'surface_follow_used': 'positions-surface-follow-used',
        'surface_drop_fraction_used':
            'positions-surface-drop-fraction-used',
        'surface_medium_reference_value_used':
            'positions-surface-medium-reference-value-used',
        'surface_z_offset_um_used': 'positions-surface-z-offset-um-used',
        'surface_follow_half_range_um_used':
            'positions-surface-follow-half-range-um-used',
        'surface_max_rewind_um_used':
            'positions-surface-max-rewind-um-used',
        'surface_verification_steps_used':
            'positions-surface-verification-steps-used',
        'surface_verification_frame_average_used':
            'positions-surface-verification-frame-average-used',
        'surface_verification_tolerance_fraction_used':
            'positions-surface-verification-tolerance-fraction-used',
        'roi_mask_used': 'positions-roi-mask-used',
        'grid_coordinates_absolute':
            'positions-grid-coordinates-absolute',
        # The medium-reference-derived drop threshold runSurfacePreScan()
        # actually used - (1 - surface_drop_fraction_used) *
        # surface_medium_reference_value_used, saved directly so it
        # doesn't need reconstructing from those two fields.
        'surface_reference_threshold':
            'positions-surface-reference-threshold-computed',
    }

    def __init__(self, payload_group, repetition):
        """
        Creates a payload representation from the corresponding group of a
        HDF file.

        Parameters
        ----------
        payload_group : HDF group
            The payload of a repetition, basically a set of images

        """
        super(Payload, self).__init__(payload_group, repetition)
        # Only Brillouin payloads have a resolution and positions.
        # A repetition from an aborted/restarted acquisition can have
        # the resolution attributes set but no positions-x/y/z
        # datasets at all (nothing was ever measured). h5py's .get()
        # then returns None, and np.array(None) does NOT raise - it
        # silently produces a bogus 0-d object array - so we have to
        # check for that explicitly instead of relying on the except
        # below to catch it.
        try:
            self.resolution = tuple(int(payload_group.attrs.get(
                'resolution-%s' % axis)[0]) for axis in ['x', 'y', 'z'])
            positions_x = payload_group.get('positions-x')
            positions_y = payload_group.get('positions-y')
            positions_z = payload_group.get('positions-z')
            if positions_x is None \
                    or positions_y is None \
                    or positions_z is None:
                raise BadFileException(
                    'Payload has no positions-x/y/z datasets')
            self.positions = {
                'x': np.array(positions_x),
                'y': np.array(positions_y),
                'z': np.array(positions_z),
            }
            self._reconstruct_surface_follow_z()
        except BaseException:
            self.resolution = None
            self.positions = None

    def _reconstruct_surface_follow_z(self):
        """
        BrillouinAcquisition versions before the fix for the surface-follow
        z metadata bug wrote positions-z as a flat nominal grid (the same z
        for every x, y at a given z-index), discarding the real,
        surface-corrected z actually used at each (x, y) - even though that
        real value is available separately, in acquisition order, as
        positions-sampled-x/y/z (always written on every acquisition, not
        only surface-following ones - see get_surface_scan_data()).

        Reconstructs the true per-point z from that path here: each sampled
        point is matched back to its (x_index, y_index) grid column by its
        (x, y) position (unaffected by the bug, so still a reliable key),
        then z values within a column are assigned by ascending sort - z
        increases monotonically with z-index by construction (the grid is
        stepped through in order), so this recovers the correct z even
        without the original acquisition-order-to-grid-index mapping
        (which isn't itself stored in the file).

        Falls back silently to the nominal positions-z already in
        self.positions['z'] for any (x, y) column with no matching sampled
        points, and for files old enough not to have positions-sampled-*
        at all.
        """
        sampled_x_ds = self.group.get('positions-sampled-x')
        sampled_y_ds = self.group.get('positions-sampled-y')
        sampled_z_ds = self.group.get('positions-sampled-z')
        if sampled_x_ds is None \
                or sampled_y_ds is None \
                or sampled_z_ds is None:
            return
        sampled_x = np.array(sampled_x_ds)
        sampled_y = np.array(sampled_y_ds)
        sampled_z = np.array(sampled_z_ds)
        if sampled_x.size == 0:
            return

        nz, nx, ny = self.positions['z'].shape
        x_grid = self.positions['x'][0, :, 0]
        y_grid = self.positions['y'][0, 0, :]

        # Explicit, tiny absolute tolerance rather than np.isclose()'s
        # default (which also scales with the values' own magnitude via
        # rtol) - in absolute-grid-coordinates mode, x/y can carry a
        # large stage-origin offset baked in, where the default rtol
        # would loosen to a fraction of a micron and risk conflating
        # two genuinely distinct, closely-spaced grid columns. 1e-3 um
        # is far below any real grid spacing yet far above float64
        # rounding noise at typical stage-position magnitudes.
        tolerance = 1e-3

        for ix in range(nx):
            x_match = np.isclose(
                sampled_x, x_grid[ix], atol=tolerance, rtol=0)
            if not np.any(x_match):
                continue
            for iy in range(ny):
                mask = x_match & np.isclose(
                    sampled_y, y_grid[iy], atol=tolerance, rtol=0)
                if not np.any(mask):
                    continue
                zs = np.sort(sampled_z[mask])
                n = min(zs.size, nz)
                self.positions['z'][:n, ix, iy] = zs[:n]

    def has_surface_scan(self):
        """
        Returns whether this payload contains data from a
        surface-following scan (BrillouinAcquisition >= the
        "major surface scanning update", H5BM-v0.0.4, no version
        bump so this can only be detected from dataset presence).
        """
        return self.group is not None and \
            self.group.get(
                self.SURFACE_SCAN_MASKS['surface_found_mask']) is not None

    def get_surface_scan_data(self):
        """
        Returns a dict with the surface-scan masks, the actually
        sampled path and the scan settings that were used, or None
        if this payload does not contain surface-scan data.

        Returns
        -------
        out: dict or None
            'surface_found_mask': ndarray[x, y], 0=not found,
                1=found by measurement, 2=gap-filled/interpolated
            'sampled_mask': ndarray[z, x, y], 1 if that grid point
                was actually sampled
            'roi_scan_plan_mask': ndarray[x, y], 1 if inside the
                planned ROI polygon
            'sampled_x'/'sampled_y'/'sampled_z': ndarray, the actually
                sampled path in acquisition order
            'roi_polygon_x'/'roi_polygon_y': ndarray, the ROI polygon
                drawn at acquisition time (stage um) - the un-
                discretized shape 'roi_scan_plan_mask' was rasterized
                from
            'prescan_x'/'prescan_y': ndarray, the coarse (binned) x/y
                sample grid the surface pre-scan itself actually
                probed (stage um) - only present on newer files (see
                SURFACE_SCAN_PATH)
            'prescan_found_mask'/'prescan_z'/'prescan_metric':
                ndarray[len(prescan_x), len(prescan_y)], the pre-
                scan's own result at its own (coarse) resolution -
                only present alongside prescan_x/prescan_y
            'curve_z'/'curve_metric': ndarray[len(prescan_x),
                len(prescan_y), max_samples], the full drop curve
                sampled at each pre-scan column, NaN-padded past
                'curve_sample_counts' - only present alongside
                prescan_x/prescan_y, and only on files new enough to
                record it (see SURFACE_SCAN_CURVES)
            'curve_sample_counts': ndarray[len(prescan_x),
                len(prescan_y)], the real (non-padding) sample count
                for each column's curve_z/curve_metric row
            'boundary_x'/'boundary_y'/'boundary_z': ndarray, extra
                boundary points that found a surface, requested
                outside the regular coarse pre-scan grid - a flat
                list, not a grid (see SURFACE_SCAN_BOUNDARY)
            plus the scalar settings listed in SURFACE_SCAN_SETTINGS
        """
        if not self.has_surface_scan():
            return None
        data = {}
        for key, dataset_name in self.SURFACE_SCAN_MASKS.items():
            ds = self.group.get(dataset_name)
            data[key] = np.array(ds) if ds is not None else None
        for key, dataset_name in self.SURFACE_SCAN_PATH.items():
            ds = self.group.get(dataset_name)
            data[key] = np.array(ds) if ds is not None else None
        for key, dataset_name in self.SURFACE_SCAN_BOUNDARY.items():
            ds = self.group.get(dataset_name)
            data[key] = np.array(ds) if ds is not None else None
        # Only meaningful together with prescan_x/prescan_y (absent on
        # older files) - skip entirely rather than returning arrays
        # with no coordinates to plot them against.
        if data.get('prescan_x') is not None \
                and data.get('prescan_y') is not None:
            for key, dataset_name in self.SURFACE_PRESCAN_MASKS.items():
                ds = self.group.get(dataset_name)
                data[key] = np.array(ds) if ds is not None else None
            for key, dataset_name in self.SURFACE_SCAN_CURVES.items():
                ds = self.group.get(dataset_name)
                data[key] = np.array(ds) if ds is not None else None
        for key, dataset_name in self.SURFACE_SCAN_SETTINGS.items():
            ds = self.group.get(dataset_name)
            data[key] = np.array(ds).flatten()[0] if ds is not None else None
        return data

    def get_surface_follow_summary(self):
        """
        Returns the surface-follow settings of this repetition and how
        well the surface was found, as scalars (what the combined CSV
        export writes into its header), or None if this payload has no
        surface-scan data.

        Returns
        -------
        out: dict or None
            The scalar settings of SURFACE_SCAN_SETTINGS (each None if
            the file predates it), plus
            'surface_planned_columns': columns inside the planned ROI,
            'surface_planned_columns_with_surface': of those, the ones
                that have a surface z (found by measurement or filled
                in, mask value 1 or 2) - the others have none,
            and, only if the file carries the coarse pre-scan:
            'surface_prescan_columns_in_roi': pre-scan columns whose
                nearest grid column lies in the planned ROI,
            'surface_prescan_found': of those, found by measurement
                (mask value 1),
            'surface_prescan_filled_in': filled in (value 2),
            'surface_prescan_not_found': not found (value 0).
            The surface is only measured on the coarse pre-scan
            columns; the dense grid's other columns get their surface
            z by interpolation, so the dense mask alone says little
            about how many columns found a surface.
        """
        data = self.get_surface_scan_data()
        if data is None:
            return None
        summary = {key: data.get(key) for key in self.SURFACE_SCAN_SETTINGS}
        found, plan = data.get('surface_found_mask'), \
            data.get('roi_scan_plan_mask')
        if found is None or plan is None:
            return summary
        planned = np.asarray(plan) > 0
        summary['surface_planned_columns'] = int(planned.sum())
        summary['surface_planned_columns_with_surface'] = int(
            (np.asarray(found)[planned] > 0).sum())

        prescan = data.get('prescan_found_mask')
        if prescan is None or self.positions is None:
            return summary
        grid_x, grid_y = self.positions['x'][0], self.positions['y'][0]
        values = []
        for i, x in enumerate(np.ravel(data['prescan_x'])):
            for j, y in enumerate(np.ravel(data['prescan_y'])):
                nearest = np.unravel_index(
                    np.argmin((grid_x - x) ** 2 + (grid_y - y) ** 2),
                    grid_x.shape)
                if planned[nearest]:
                    values.append(np.asarray(prescan)[i, j])
        values = np.array(values)
        summary['surface_prescan_columns_in_roi'] = int(values.size)
        summary['surface_prescan_found'] = int((values == 1).sum())
        summary['surface_prescan_filled_in'] = int((values == 2).sum())
        summary['surface_prescan_not_found'] = int((values == 0).sum())
        return summary

    # Background ROI polygon (BrillouinAcquisition's second, independent
    # ROI - see the Background class for the reference points it
    # captured). The drawn shape itself, not those points.
    BACKGROUND_ROI = {
        'polygon_x': 'positions-background-roi-polygon-x-um',
        'polygon_y': 'positions-background-roi-polygon-y-um',
    }

    def get_background_roi(self):
        """
        Returns the background ROI polygon drawn at acquisition time,
        or None if this file predates the feature entirely (no
        'background-roi-mask-used' dataset at all).

        Returns
        -------
        out: dict or None
            'mask_used': bool, whether the background ROI was enabled
                for this run.
            'polygon_x'/'polygon_y': ndarray or None, the drawn
                polygon's vertices (stage um, same grid-plan frame as
                get_surface_scan_data()'s own 'roi_polygon_x'/'y') -
                None if the ROI was never drawn (mask_used False) or
                had fewer than 3 vertices.
        """
        if self.group is None:
            return None
        mask_used_ds = self.group.get('positions-background-roi-mask-used')
        if mask_used_ds is None:
            return None
        data = {'mask_used': bool(np.array(mask_used_ds).flatten()[0])}
        for key, dataset_name in self.BACKGROUND_ROI.items():
            ds = self.group.get(dataset_name)
            data[key] = np.array(ds) if ds is not None else None
        return data

    # Spectral proxy ROI(s) used for surface-follow autofocus scoring
    # (Brillouin::estimateFrameMetric()) - up to two rectangles, each in
    # the pixel coordinates of the camera frame (crop/binning) they were
    # drawn against. SPECTRAL_PROXY_ROI_FRAME records that frame's own
    # geometry, so a reader can remap a ROI onto a different frame the
    # same way Brillouin::remapProxyRoi() does (a size-only rescale is
    # only correct if both frames share the same sensor origin).
    SPECTRAL_PROXY_ROI = {
        1: 'positions-surface-proxy-roi-1-used',
        2: 'positions-surface-proxy-roi-2-used',
    }
    SPECTRAL_PROXY_ROI_FRAME = {
        1: 'positions-surface-proxy-roi-1-frame-used',
        2: 'positions-surface-proxy-roi-2-frame-used',
    }

    def get_spectral_proxy_rois(self):
        """
        Returns the spectral proxy ROI(s) used for surface-follow
        autofocus scoring, or None if this file predates the feature
        entirely (no 'surface-proxy-roi-1-used' dataset at all).

        Returns
        -------
        out: dict or None
            {1: roi, 2: roi}, where each roi is a dict with 'left',
            'top', 'width', 'height' (pixel coordinates in the frame
            it was drawn against) and 'frame' (a dict with 'width',
            'height', 'origin_left', 'origin_bottom', 'width_physical',
            'height_physical'), or None if that ROI slot was never
            drawn (width/height <= 0).
        """
        if self.group is None:
            return None
        rois = {}
        for index, dataset_name in self.SPECTRAL_PROXY_ROI.items():
            ds = self.group.get(dataset_name)
            if ds is None:
                return None
            left, top, width, height = np.array(ds).flatten()
            if width <= 0 or height <= 0:
                rois[index] = None
                continue
            frame_ds = self.group.get(self.SPECTRAL_PROXY_ROI_FRAME[index])
            frame = None
            if frame_ds is not None:
                (frame_width, frame_height, origin_left, origin_bottom,
                 width_physical, height_physical) = \
                    np.array(frame_ds).flatten()
                frame = {
                    'width': int(frame_width),
                    'height': int(frame_height),
                    'origin_left': int(origin_left),
                    'origin_bottom': int(origin_bottom),
                    'width_physical': int(width_physical),
                    'height_physical': int(height_physical),
                }
            rois[index] = {
                'left': int(left), 'top': int(top),
                'width': int(width), 'height': int(height),
                'frame': frame,
            }
        return rois

    # General per-repetition acquisition settings, written unconditionally
    # for every Brillouin repetition regardless of whether surface-follow
    # was used (unlike SURFACE_SCAN_SETTINGS, which get_surface_scan_data()
    # only returns when has_surface_scan() is True) - each individually
    # None on a file old enough to predate that particular setting.
    ACQUISITION_SETTINGS = {
        'pre_calibration_used': 'positions-pre-calibration-used',
        'post_calibration_used': 'positions-post-calibration-used',
        'con_calibration_used': 'positions-con-calibration-used',
        'con_calibration_interval_min_used':
            'positions-con-calibration-interval-min-used',
        'nr_calibration_images_used':
            'positions-nr-calibration-images-used',
        'calibration_exposure_time_s_used':
            'positions-calibration-exposure-time-s-used',
        'repetitions_count_used': 'positions-repetitions-count-used',
        'repetitions_interval_min_used':
            'positions-repetitions-interval-min-used',
        'repetitions_file_per_repetition_used':
            'positions-repetitions-file-per-repetition-used',
        'overview_brightfield_save_per_z_used':
            'positions-overview-brightfield-save-per-z-used',
        'overview_brightfield_full_grid_used':
            'positions-overview-brightfield-full-grid-used',
        'overview_brightfield_full_stack_single_used':
            'positions-overview-brightfield-full-stack-single-used',
        'overview_brightfield_full_stack_mosaic_used':
            'positions-overview-brightfield-full-stack-mosaic-used',
        'overview_brightfield_exposure_ms_used':
            'positions-overview-brightfield-exposure-ms-used',
        'overview_brightfield_gain_used':
            'positions-overview-brightfield-gain-used',
        'use_grid_hysteresis_compensation_used':
            'positions-use-grid-hysteresis-compensation-used',
        'use_dose_protection_used': 'positions-use-dose-protection-used',
        'capture_per_point_brightfield_used':
            'positions-capture-per-point-brightfield-used',
        'per_point_brightfield_every_n_used':
            'positions-per-point-brightfield-every-n-used',
        'per_point_brightfield_during_acquisition_used':
            'positions-per-point-brightfield-during-acquisition-used',
        'camera_frame_count_used': 'positions-camera-frame-count-used',
        'camera_spurious_noise_filter_used':
            'positions-camera-spurious-noise-filter-used',
        'background_roi_mask_used': 'positions-background-roi-mask-used',
        # The overview engine and drift correction of BrillouinAcquisition,
        # see get_overview_events()/get_runtime_drift().
        'overview_engine_used': 'positions-overview-engine-used',
        'overview_before_measurement_used':
            'positions-overview-before-measurement-used',
        'overview_after_measurement_used':
            'positions-overview-after-measurement-used',
        'overview_trigger_used': 'positions-overview-trigger-used',
        'overview_brightfield_spread_across_scan_used':
            'positions-overview-brightfield-spread-across-scan-used',
        'overview_objective_retract_um_used':
            'positions-overview-objective-retract-um-used',
        'overview_measurement_objective_slot':
            'positions-overview-measurement-objective-slot',
        'drift_correction_used': 'positions-drift-correction-used',
        'drift_objective_slot_used': 'positions-drift-objective-slot-used',
        'drift_min_ncc_used': 'positions-drift-min-ncc-used',
        'drift_max_step_um_used': 'positions-drift-max-step-um-used',
        'drift_extrapolate_used': 'positions-drift-extrapolate-used',
        'drift_carried_over_x_um': 'positions-drift-carried-over-x-um',
        'drift_carried_over_y_um': 'positions-drift-carried-over-y-um',
        # 0 never, 1 with every overview, 2 after every z plane
        'surface_rescan_used': 'positions-surface-rescan-used',
    }

    def get_acquisition_settings(self):
        """
        Returns the general per-repetition acquisition settings that
        were actually used, or None if self.group is None. Each entry
        is individually None if this file predates that particular
        setting - unlike get_surface_scan_data(), this is not gated
        behind has_surface_scan(), since none of these settings are
        specific to surface-following.

        Returns
        -------
        out: dict or None
            See ACQUISITION_SETTINGS for the available keys.
        """
        if self.group is None:
            return None
        data = {}
        for key, dataset_name in self.ACQUISITION_SETTINGS.items():
            ds = self.group.get(dataset_name)
            data[key] = np.array(ds).flatten()[0] if ds is not None else None
        return data

    # How the drift decision of an overview event is stored
    # (overview-event-N-drift-decision).
    DRIFT_DECISIONS = {
        0: 'reference', 1: 'applied', 2: 'no match',
        3: 'low confidence', 4: 'implausible'}

    def _positions_array(self, name):
        """The dataset 'positions-<name>' as a flat float array, None if
        the payload has none."""
        if self.group is None:
            return None
        dataset = self.group.get('positions-' + name)
        if dataset is None:
            return None
        return np.array(dataset, dtype=float).ravel()

    def _positions_array_with_length(self, name):
        """Like _positions_array() for the numbers BrillouinAcquisition
        writes with their length in front (everything of the resume
        plan); without the length."""
        values = self._positions_array(name)
        if values is None or values.size == 0:
            return None
        return values[1:1 + int(values[0])]

    def get_overview_events(self):
        """
        Returns the overview events the overview engine of
        BrillouinAcquisition recorded during this repetition, in the
        order they happened (empty list if there are none: older files,
        or overviews switched off).

        Returns
        -------
        out: list of dict
            'index': number of the event (the 'overview_event' of its
                images),
            'kind', 'kind_name' (before measurement, after z plane, after
                repetition, after measurement), 'z_index', 'point_index'
                (-1: before the first point), 'seconds_since_start',
                'epoch_seconds', 'repetition',
            'objectives': one dict per objective that took part: 'slot',
                'is_measurement_objective', 'captured', 'tiles',
                'stack_depth', 'images_saved', 'first_image_number',
                'switch_to_seconds', 'capture_seconds',
                'switch_back_seconds', 'fov_compensation' (x, y),
                'focus_delta_um', 'retract_um',
            'drift': None without drift correction, else a dict:
                'evaluated', 'decision' (see DRIFT_DECISIONS),
                'decision_name', 'step_um' (x, y), 'total_um' (x, y),
                'grid_offset_after_um' (x, y), 'mean_ncc', 'tiles_compared',
                'tiles_used', 'seconds_since_reference',
                'registration_seconds', 'reference_replaced'.
        """
        if self.group is None:
            return []
        pattern = re.compile(r'^positions-overview-event-(\d+)-kind$')
        indices = sorted(
            int(match.group(1)) for match in
            (pattern.match(name) for name in self.group.keys()) if match)
        events = []
        for index in indices:
            prefix = 'overview-event-%04d-' % index

            def values(name, prefix=prefix):
                return self._positions_array(prefix + name)

            def scalar(name, default=None, values=values):
                array = values(name)
                return default if array is None or array.size == 0 \
                    else float(array[0])

            def pair(name, values=values):
                array = values(name)
                if array is None or array.size < 2:
                    return None
                return (float(array[0]), float(array[1]))

            kind = scalar('kind')
            kind = None if kind is None else int(kind)

            objectives = []
            slots = values('objective-slot')
            for number in range(0 if slots is None else slots.size):
                def at(name, default=None, number=number, values=values):
                    array = values('objective-' + name)
                    return default if array is None \
                        or array.size <= number else float(array[number])
                fov_x = at('fov-compensation-x-um')
                fov_y = at('fov-compensation-y-um')
                objectives.append({
                    'slot': int(slots[number]),
                    'is_measurement_objective': bool(
                        at('is-measurement-objective', 0)),
                    'captured': bool(at('captured', 0)),
                    'tiles': int(at('tiles', 0)),
                    'stack_depth': int(at('stack-depth', 0)),
                    'images_saved': int(at('images-saved', 0)),
                    'first_image_number': int(at('first-image-number', 0)),
                    'switch_to_seconds': at('switch-to-seconds'),
                    'capture_seconds': at('capture-seconds'),
                    'switch_back_seconds': at('switch-back-seconds'),
                    'fov_compensation': None if fov_x is None
                    else (fov_x, fov_y),
                    'focus_delta_um': at('focus-delta-um'),
                    'retract_um': at('retract-um'),
                })

            drift = None
            if values('drift-decision') is not None:
                decision = scalar('drift-decision')
                decision = None if decision is None else int(decision)
                compared = values('drift-tiles-compared-used')
                drift = {
                    'evaluated': bool(scalar('drift-evaluated', 0)),
                    'decision': decision,
                    'decision_name': self.DRIFT_DECISIONS.get(decision),
                    'step_um': pair('drift-step-um'),
                    'total_um': pair('drift-total-um'),
                    'grid_offset_after_um': pair('grid-offset-after-um'),
                    'mean_ncc': scalar('drift-mean-ncc'),
                    'tiles_compared': None if compared is None
                    or compared.size < 2 else int(compared[0]),
                    'tiles_used': None if compared is None
                    or compared.size < 2 else int(compared[1]),
                    'seconds_since_reference':
                        scalar('drift-seconds-since-reference'),
                    'registration_seconds':
                        scalar('drift-registration-seconds'),
                    # the reference of the previous run could not be used
                    # and this overview became the new one
                    'reference_replaced': bool(
                        scalar('drift-reference-replaced', 0)),
                }
            events.append({
                'index': index,
                'kind': kind,
                'kind_name': MeasurementData.OVERVIEW_KIND_NAMES.get(kind),
                'z_index': int(scalar('z-index', 0)),
                'point_index': int(scalar('point-index', -1)),
                'seconds_since_start': scalar('seconds-since-start'),
                'epoch_seconds': scalar('epoch-seconds'),
                'repetition': scalar('repetition'),
                'objectives': objectives,
                'drift': drift,
            })
        return events

    def get_scan_plan(self):
        """
        Returns the order the grid points were measured in, as stored
        with the plan that Continue needs (BrillouinAcquisition),
        or None for files without it.

        Returns
        -------
        out: dict
            'index_x', 'index_y', 'index_z': int arrays, the grid index
                of the n-th measured point; 'address': the key of that
                point in the payload (x runs fastest), so that
                payload.data[str(address[n])] is the n-th point
                measured.
        """
        if self.positions is None:
            return None
        index = [self._positions_array_with_length(
            'resume-plan-index-' + axis) for axis in ('x', 'y', 'z')]
        if any(array is None for array in index):
            return None
        index_x, index_y, index_z = (array.astype(int) for array in index)
        nz, nx, ny = self.positions['x'].shape
        if index_x.size == 0 or index_x.max() >= nx \
                or index_y.max() >= ny or index_z.max() >= nz:
            return None
        return {
            'index_x': index_x, 'index_y': index_y, 'index_z': index_z,
            'address': index_x + nx * (index_y + ny * index_z),
        }

    def get_point_drift_offsets(self):
        """
        Returns the drift offset (um) that BrillouinAcquisition had added
        to the stage target of each grid point (drift correction; 0
        without), laid out like positions-x/y/z, i.e. arrays of shape
        (nz, nx, ny) with NaN where nothing was measured. The point was
        measured at positions-x/y + this offset, so the position of the
        measurement on the sample is positions-x/y itself.

        None for files without the record (before drift correction, or
        no scan plan stored).
        """
        offset_x = self._positions_array('grid-offset-per-point-x-um')
        offset_y = self._positions_array('grid-offset-per-point-y-um')
        plan = self.get_scan_plan()
        if offset_x is None or offset_y is None or plan is None \
                or offset_x.size != plan['address'].size \
                or offset_y.size != offset_x.size:
            return None
        shape = self.positions['x'].shape
        result = {}
        for axis, offsets in (('x', offset_x), ('y', offset_y)):
            grid = np.full(shape, np.nan)
            grid[plan['index_z'], plan['index_x'], plan['index_y']] = offsets
            result[axis] = grid
        return result

    def get_runtime_drift(self):
        """
        Returns what the drift correction of BrillouinAcquisition did
        during this repetition, or None if it was not switched on (or the
        file predates it).

        Returns
        -------
        out: dict
            'objective_slot', 'min_ncc', 'max_step_um', 'extrapolate',
            'carried_over': (x, y) um, the drift taken over from the
                repetition before,
            'events': the 'drift' parts of get_overview_events() that were
                evaluated (with 'event', 'kind_name', 'seconds_since_start'),
            'applied_events': how many of them moved the grid,
            'max_offset_um': the largest offset any point was measured with,
            'applied': True if the grid was moved at all (False: switched
                on, but nothing could be registered - the CSV positions
                are then not drift corrected).
        """
        if self.group is None:
            return None
        used = self._positions_array('drift-correction-used')
        if used is None or used.size == 0 or not used[0]:
            return None
        events = []
        for event in self.get_overview_events():
            drift = event['drift']
            if drift is not None and drift['evaluated']:
                events.append(dict(
                    drift, event=event['index'], kind_name=event['kind_name'],
                    seconds_since_start=event['seconds_since_start']))
        offsets = self.get_point_drift_offsets()
        max_offset = 0.0
        if offsets is not None:
            with np.errstate(invalid='ignore'):
                size = np.hypot(offsets['x'], offsets['y'])
            if np.isfinite(size).any():
                max_offset = float(np.nanmax(size))
        applied_events = sum(1 for e in events if e['decision'] == 1)

        def setting(name):
            array = self._positions_array(name)
            return None if array is None or array.size == 0 \
                else float(array[0])

        carried = (setting('drift-carried-over-x-um') or 0.0,
                   setting('drift-carried-over-y-um') or 0.0)
        return {
            'objective_slot': setting('drift-objective-slot-used'),
            'min_ncc': setting('drift-min-ncc-used'),
            'max_step_um': setting('drift-max-step-um-used'),
            'extrapolate': bool(setting('drift-extrapolate-used') or 0),
            'carried_over': carried,
            'events': events,
            'applied_events': applied_events,
            'max_offset_um': max_offset,
            'applied': applied_events > 0 or max_offset > 0
            or carried != (0.0, 0.0),
        }

    def get_resume_events(self):
        """
        Returns the notes BrillouinAcquisition left when the measurement
        of this repetition was continued (Continue button), oldest first;
        empty if it never was.

        Returns
        -------
        out: list of dict
            'index', 'point_index' (points done when it had stopped),
            'points_total', 'resumed_epoch_seconds', 'was_cancelled'
            (False: crash or power loss), 'incomplete_points_removed',
            'previous_state', 'last_point_date', 'resumed_date'.
        """
        if self.group is None:
            return []
        pattern = re.compile(r'^positions-resume-event-(\d+)-point-index$')
        indices = sorted(
            int(match.group(1)) for match in
            (pattern.match(name) for name in self.group.keys()) if match)
        events = []
        for index in indices:
            prefix = 'resume-event-%04d-' % index

            def number(name, prefix=prefix):
                array = self._positions_array_with_length(prefix + name)
                return None if array is None or array.size == 0 \
                    else float(array[0])

            def text(name, prefix=prefix):
                value = _get_text_attribute(
                    self.group, 'resume_' + prefix[len('resume-'):] + name)
                # BrillouinAcquisition writes "s:" in front of every text
                return value[2:] if value and value.startswith('s:') \
                    else value

            events.append({
                'index': index,
                'point_index': number('point-index'),
                'points_total': number('points-total'),
                'resumed_epoch_seconds': number('resumed-epoch-seconds'),
                'was_cancelled': bool(number('was-cancelled') or 0),
                'incomplete_points_removed': number(
                    'incomplete-points-removed'),
                'previous_state': text('previous-state'),
                'last_point_date': text('last-point-date'),
                'resumed_date': text('resumed-date'),
            })
        return events

    def get_grid_definition(self):
        """
        Returns how the measurement grid was defined, or None if the
        payload has no positions (nothing was measured).

        Returns
        -------
        out: dict
            'steps': {'x', 'y', 'z'} number of grid points,
            'range_um': {'x', 'y', 'z'} -> (min, max) as configured (None
                for files that do not carry them),
            'step_um': {'x', 'y', 'z'} the spacing of the points derived
                from steps and range (None for a single point),
            'absolute_origin_um': {'x', 'y', 'z'}: the origin the
                operator set for absolute grid coordinates,
            'coordinates_absolute': bool or None,
            'scan_loop_level': {'x', 'y', 'z'}: 0 = innermost loop ...
                2 = outermost, as planned (None for older files),
            'roi_polygon_um' / 'background_roi_polygon_um': (x, y) arrays
                of the polygons drawn at acquisition time, or None.
        """
        if self.positions is None or self.resolution is None:
            return None
        axes = ('x', 'y', 'z')
        steps = dict(zip(axes, self.resolution))
        ranges, spacing, origin = {}, {}, {}
        for axis in axes:
            values = self._positions_array('grid-%s-range-um-used' % axis)
            ranges[axis] = None if values is None or values.size < 2 \
                else (float(values[0]), float(values[1]))
            if ranges[axis] is not None and steps[axis] > 1:
                spacing[axis] = (ranges[axis][1] - ranges[axis][0]) \
                    / (steps[axis] - 1)
            else:
                spacing[axis] = None
            value = self._positions_array('absolute-origin-' + axis)
            origin[axis] = None if value is None or value.size == 0 \
                else float(value[0])
        absolute = self._positions_array('grid-coordinates-absolute')
        order = self._positions_array('grid-scan-order-used')

        def polygon(name):
            x = self._positions_array(name + '-x-um')
            y = self._positions_array(name + '-y-um')
            if x is None or y is None or x.size == 0 or x.size != y.size:
                return None
            return (x, y)

        return {
            'steps': steps,
            'range_um': ranges,
            'step_um': spacing,
            'absolute_origin_um': origin,
            'coordinates_absolute': None if absolute is None
            or absolute.size == 0 else bool(absolute[0]),
            'scan_loop_level': None if order is None or order.size < 3
            else dict(zip(axes, (int(v) for v in order[:3]))),
            'roi_polygon_um': polygon('roi-polygon'),
            'background_roi_polygon_um': polygon('background-roi-polygon'),
        }

    def raw_to_grid_frame(self, raw_xy):
        """
        The (x, y) of a raw position (stage + scanner, um - see
        get_image_centre_raw()) in the frame of this repetition's
        positions-x/y (and so of the CSV of a BrillouinExport), or None
        if the file does not say which frame that is.

        With absolute grid coordinates the frame is the raw position
        minus the absolute origin of the grid; with relative ones it is
        the raw position itself (BrillouinAcquisition's
        Brillouin::rawPositionToGridFrame()).
        """
        if raw_xy is None:
            return None
        grid = self.get_grid_definition()
        if grid is None or grid['coordinates_absolute'] is None:
            return None
        if not grid['coordinates_absolute']:
            return (raw_xy[0], raw_xy[1])
        origin = grid['absolute_origin_um']
        if origin['x'] is None or origin['y'] is None:
            return None
        return (raw_xy[0] - origin['x'], raw_xy[1] - origin['y'])

    def get_image_centre_shift(self, image_key):
        """
        (x, y) in um from the stage target of an image to its middle -
        'image_centre_shift' plus 'fov_compensation' - or None if the
        image was written without it. A target puts the laser on a point,
        and the laser is where its marker was set in the camera image, not
        in the middle (see get_overview_info()). BrillouinAcquisition
        writes it with every brightfield image; tools/add_image_centres.py
        added it to the images taken before.
        """
        shift = self._attribute_pair(
            image_key, 'image_centre_shift_x_um', 'image_centre_shift_y_um')
        if shift is None:
            return None
        fov = self._attribute_pair(
            image_key, 'fov_compensation_x_um', 'fov_compensation_y_um',
            (0.0, 0.0))
        return (shift[0] + fov[0], shift[1] + fov[1])

    def get_image_centre_raw(self, image_key):
        """
        The raw position (stage + scanner, um; the frame of
        ScanControl::getPosition()) of the middle of an image taken in
        the Fluorescence tab - 'stage_position_raw' minus the image
        centre shift - or None if the image was written without them.

        Such an image does not carry a position in the frame of the
        Brillouin run: its 'position' is in the grid frame in force
        when it was taken (before Start, so with another origin, or
        none), and the laser target it describes is not the middle of
        the image.
        """
        raw = self._attribute_pair(
            image_key, 'stage_position_raw_x_um', 'stage_position_raw_y_um')
        shift = self.get_image_centre_shift(image_key)
        if raw is None or shift is None:
            return None
        return (raw[0] - shift[0], raw[1] - shift[1])

    def get_overview_objectives(self):
        """
        Returns the calibrations of the objectives that took overviews
        in this repetition (written before the measurement starts), by
        slot; empty for files without them.

        Returns
        -------
        out: dict of dict
            slot -> {'magnification', 'micrometerToPixX',
            'micrometerToPixY', 'pixToMicrometerX', 'pixToMicrometerY',
            'origin' ((x, y) each), 'has_fov_offset', 'fov_offset' (x, y),
            'fov_offset_sigma_um', 'has_focus_offset', 'focus_offset_um',
            'focus_offset_sigma_um', 'is_reference_objective'}
        """
        if self.group is None:
            return {}
        pattern = re.compile(r'^positions-overview-objective-(\d+)-'
                             r'magnification$')
        slots = sorted(int(match.group(1)) for match in
                       (pattern.match(name) for name in self.group.keys())
                       if match)
        result = {}
        for slot in slots:
            prefix = 'overview-objective-%d-' % slot

            def array(name, prefix=prefix):
                return self._positions_array(prefix + name)

            def first(name, array=array):
                values = array(name)
                return None if values is None or values.size == 0 \
                    else float(values[0])

            def pair(name, array=array):
                values = array(name)
                return None if values is None or values.size < 2 \
                    else (float(values[0]), float(values[1]))

            fov = array('fov-offset-um')
            focus = array('focus-offset-um')
            result[slot] = {
                'magnification': first('magnification'),
                'micrometerToPixX': pair('micrometer-to-pix-x'),
                'micrometerToPixY': pair('micrometer-to-pix-y'),
                'pixToMicrometerX': pair('pix-to-micrometer-x'),
                'pixToMicrometerY': pair('pix-to-micrometer-y'),
                'origin': pair('origin-pix'),
                'has_fov_offset': None if fov is None or fov.size < 4
                else bool(fov[0]),
                'fov_offset': None if fov is None or fov.size < 4
                else (float(fov[1]), float(fov[2])),
                'fov_offset_sigma_um': None if fov is None or fov.size < 4
                else float(fov[3]),
                'has_focus_offset': None if focus is None or focus.size < 3
                else bool(focus[0]),
                'focus_offset_um': None if focus is None or focus.size < 3
                else float(focus[1]),
                'focus_offset_sigma_um': None if focus is None
                or focus.size < 3 else float(focus[2]),
                'is_reference_objective': bool(
                    first('is-reference-objective') or 0),
            }
        return result

    def get_surface_rescans(self):
        """
        Returns the repeated surface scans of this repetition (setting
        'surface rescan': with every overview or after every z plane),
        oldest first; empty if the surface was scanned only once.

        Each entry has 'index', 'success', 'point_index' (the point the
        scan was repeated after), 'seconds_since_start',
        'duration_seconds', 'grid_offset_um' (x, y),
        'total_columns', 'failed_columns', 'points_without_new_surface'
        (those keep their old z), 'prior_used' and 'reference_threshold'.
        The surface found is in the 'positions-surface-rescan-NNNN-*'
        datasets (prescan-x-um, prescan-z-um, ...).
        """
        if self.group is None:
            return []
        pattern = re.compile(r'^positions-surface-rescan-(\d+)-success$')
        indices = sorted(int(match.group(1)) for match in
                         (pattern.match(name) for name in self.group.keys())
                         if match)
        rescans = []
        for index in indices:
            prefix = 'surface-rescan-%04d-' % index

            def array(name, prefix=prefix):
                return self._positions_array(prefix + name)

            def first(name, array=array):
                values = array(name)
                return None if values is None or values.size == 0 \
                    else float(values[0])

            columns = array('total-columns-failed-columns')
            offset = array('grid-offset-um')
            rescans.append({
                'index': index,
                'success': bool(first('success') or 0),
                'point_index': None if first('point-index') is None
                else int(first('point-index')),
                'seconds_since_start': first('seconds-since-start'),
                'duration_seconds': first('duration-seconds'),
                'grid_offset_um': None if offset is None or offset.size < 2
                else (float(offset[0]), float(offset[1])),
                'total_columns': None if columns is None or columns.size < 2
                else int(columns[0]),
                'failed_columns': None if columns is None or columns.size < 2
                else int(columns[1]),
                'points_without_new_surface': None
                if first('points-without-new-surface') is None
                else int(first('points-without-new-surface')),
                'prior_used': bool(first('prior-used') or 0),
                'reference_threshold': first('reference-threshold'),
            })
        return rescans

    # Keys of get_scale_calibration()'s return value that are
    # point-style (an (x, y) tuple, or None if missing) rather than a
    # plain scalar - so a caller that needs to serialize the dict
    # (e.g. BrillouinExport's CSV header) can tell the two apart
    # without relying on isinstance() (which can't distinguish a
    # missing point-style value, itself None, from a missing scalar).
    SCALE_CALIBRATION_POINT_KEYS = {
        'micrometerToPixX', 'micrometerToPixY',
        'pixToMicrometerX', 'pixToMicrometerY',
        'positionScanner', 'positionStage', 'origin', 'fovOffset',
    }

    def get_scale_calibration(self):
        """
        Returns the scale-calibration parameters recorded for this
        payload, or None if this payload has no scaleCalibration group
        at all.

        Besides the original point-style parameters (each an (x, y)
        tuple), newer files also carry per-objective context:
        'objectiveSlot', 'objectiveName', 'magnification',
        'referenceObjectiveName', 'hasFovOffset', 'fovOffset' ((x, y),
        um), 'fovOffsetSigmaUm' and 'missingOffsetAccepted'. Each of
        these degrades independently to None if missing (e.g. an
        older file predating the per-objective FOV-offset feature)
        rather than discarding an otherwise-usable calibration - only
        the group's total absence returns None outright.
        """
        cal = self.group.get('scaleCalibration') \
            if self.group is not None else None
        if cal is None:
            return None

        def point(name):
            try:
                val = cal.get(name)
                return tuple(val.attrs.get(dim)[0] for dim in ['x', 'y'])
            except BaseException:
                return None

        def scalar(name, decode=False):
            try:
                val = cal.attrs.get(name)
                if val is None:
                    return None
                val = val[0]
                return val.decode('utf-8') if decode else val
            except BaseException:
                return None

        scaleCal = dict()
        for attribute in [
                'micrometerToPixX', 'micrometerToPixY',
                'pixToMicrometerX', 'pixToMicrometerY',
                'positionScanner', 'positionStage', 'origin']:
            scaleCal[attribute] = point(attribute)

        scaleCal['objectiveSlot'] = scalar('objectiveSlot')
        scaleCal['objectiveName'] = scalar('objectiveName', decode=True)
        scaleCal['magnification'] = scalar('magnification')
        scaleCal['referenceObjectiveName'] = \
            scalar('referenceObjectiveName', decode=True)
        has_fov_offset = scalar('hasFovOffset')
        scaleCal['hasFovOffset'] = \
            bool(has_fov_offset) if has_fov_offset is not None else None
        scaleCal['fovOffset'] = point('fovOffset')
        # Note: written to the file as "fovOffsetSigma", not
        # "fovOffsetSigmaUm" - see H5BM::setScaleCalibration().
        scaleCal['fovOffsetSigmaUm'] = scalar('fovOffsetSigma')
        missing_offset_accepted = scalar('missingOffsetAccepted')
        scaleCal['missingOffsetAccepted'] = \
            bool(missing_offset_accepted) \
            if missing_offset_accepted is not None else None

        return scaleCal


class Calibration(MeasurementData):

    def __init__(self, payload_group, repetition):
        """
        Creates a calibration representation from the corresponding group of
        a HDF file.

        Parameters
        ----------
        payload_group : HDF group
            Calibration data of a repetition from an HDF file.
        """
        super(Calibration, self).__init__(payload_group, repetition)


class Background(MeasurementData):

    def __init__(self, payload_group, repetition):
        """
        Creates a background-reference-points representation from the
        corresponding group of an HDF file (BrillouinAcquisition's second,
        independent ROI - see its own settings.backgroundRoiPolygonUm).

        Each image is a real Brillouin spectrum (same frame stack shape as
        a regular payload point), captured after the main grid finishes and
        stored under its own top-level "background" group - entirely
        separate from "payload" (the main measurement grid, addressed by
        BrillouinAcquisition's calculateIndex()) and from "calibration"
        (the frequency-reference spectra). Background images are addressed
        by a plain sequential key (like Calibration), not an (x, y, z) grid
        index, so there is no possibility of a background point being
        picked up by code that iterates the main grid's resolution-bounded
        key range, or of it affecting frequency calibration.

        image_keys()/get_image()/get_position()/get_stage_position()/
        get_channel() are all inherited from MeasurementData - each image
        carries its own target/stage position (µm) exactly like a
        Fluorescence-mode or per-point-brightfield image does, since the
        acquisition side writes it through the same FLUOIMAGE mechanism.

        Parameters
        ----------
        payload_group : HDF group
            The "background" group of a repetition from an HDF file.
        """
        super(Background, self).__init__(payload_group, repetition)


class BadFileException(Exception):
    pass
