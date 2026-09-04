"""
Module for interacting with files containing Brillouin microscopy
data.

NOTE: Ideally, the users of bmlab should not have to know about
the file format in which the data are stored. So, if possible,
do not expose HDF objects to the outside (like BMicro).
"""

import datetime

import numpy as np
import h5py
from pathlib import Path

from packaging import version

BRILLOUIN_GROUP = 'Brillouin'
FLUORESCENCE_GROUP = 'Fluorescence'

# Channel name BrillouinAcquisition uses for brightfield overview
# images recorded during a surface-following measurement. These are
# stored as regular Fluorescence-mode images, distinguished only by
# this channel attribute (no dedicated HDF5 group exists for them).
OVERVIEW_BRIGHTFIELD_CHANNEL = 'Brightfield z overview'


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
        self.file = h5py.File(self.path, 'r')
        self.file_version_string = self.file.attrs.get('version')[
            0].decode('utf-8')
        if not self.file_version_string.startswith('H5BM'):
            raise BadFileException('File does not contain any Brillouin data')
        self.file_version = self.file_version_string[-5:]
        # Fluorescence group is optional
        self.Fluorescence_group = None

        if version.parse(self.file_version) >= version.parse("0.0.4"):
            """"
            New Brillouin file format,
            supporting different modes and repetitions
            """
            if BRILLOUIN_GROUP not in self.file:
                raise BadFileException(
                    'File does not contain any Brillouin data')
            self.Brillouin_group = self.file[BRILLOUIN_GROUP]

            if FLUORESCENCE_GROUP in self.file:
                self.Fluorescence_group = self.file[FLUORESCENCE_GROUP]
        else:
            """ Old Brillouin file format """
            self.Brillouin_group = self.file
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
        if version.parse(self.file_version) >= version.parse("0.0.4"):
            return list(mode_group.keys())
        elif mode == BRILLOUIN_GROUP:
            return list(['0'])
        else:
            return []

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
        if version.parse(self.file_version) >= version.parse("0.0.4"):
            return Repetition(mode_group.get(repetition_key), self)
        else:
            return Repetition(self.file, self)


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
        self.payload = Payload(repetition_group.get('payload'), self)
        calibration_group = repetition_group.get('calibration')
        # Having a calibration is optional for a repetition
        if calibration_group is not None:
            self.calibration = Calibration(calibration_group, self)
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
        return np.array(imgs)

    def get_image_count(self, image_key):
        if self.data is None:
            return 0
        imgs = self.data.get(image_key)
        if imgs is None:
            return 0
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
            # For older files we return a default value
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
            # For older files we return a default value
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
        except BaseException:
            self.resolution = None
            self.positions = None

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
        for key, dataset_name in self.SURFACE_SCAN_SETTINGS.items():
            ds = self.group.get(dataset_name)
            data[key] = np.array(ds).flatten()[0] if ds is not None else None
        return data

    def has_overview_brightfield(self):
        """
        Returns whether stage positions for brightfield overview
        images (recorded alongside a surface-following scan) are
        present for this payload.
        """
        return self.group is not None and \
            self.group.get('positions-overview-brightfield-x') is not None

    def get_overview_brightfield_positions(self):
        """
        Returns the stage positions of the brightfield overview
        images recorded during the measurement.

        Returns
        -------
        out: dict or None
            'x'/'y'/'z': ndarray, reshaped to (z_steps, tile_count)
                if more than one tile was recorded per z-slice,
                otherwise a flat 1D array of length z_steps
            'tile_count': int
        """
        if not self.has_overview_brightfield():
            return None
        tile_count_ds = self.group.get(
            'positions-overview-brightfield-tile-count')
        tile_count = int(np.array(tile_count_ds).flatten()[0]) \
            if tile_count_ds is not None else 1
        positions = {}
        for axis in ('x', 'y', 'z'):
            arr = np.array(
                self.group.get('positions-overview-brightfield-' + axis))
            if tile_count > 1 and arr.size % tile_count == 0:
                arr = arr.reshape((-1, tile_count))
            positions[axis] = arr
        positions['tile_count'] = tile_count
        return positions

    def get_scale_calibration(self):
        parameters = [
            'micrometerToPixX', 'micrometerToPixY',
            'pixToMicrometerX', 'pixToMicrometerY',
            'positionScanner', 'positionStage', 'origin'
        ]
        try:
            cal = self.group.get('scaleCalibration')
            scaleCal = dict()
            for attribute in parameters:
                val = cal.get(attribute)
                scaleCal[attribute] = \
                    tuple(val.attrs.get(dim)[0] for dim in ['x', 'y'])

            return scaleCal
        except BaseException:
            return None


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
        """
        For H5BM files < 0.0.4
        there was an inconsistency with the group naming
        """
        if self.data is None and payload_group is not None:
            self.data = payload_group.get('calibrationData')


class BadFileException(Exception):
    pass
