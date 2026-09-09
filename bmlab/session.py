import os
import errno

from pathlib import Path
from functools import reduce

import h5py
from numpy import transpose
import math

from bmlab import __version__ as version
from bmlab.file import BrillouinFile, is_source_file, is_session_file, \
    FLUORESCENCE_GROUP, OVERVIEW_BRIGHTFIELD_CHANNEL
from bmlab.models.extraction_model import ExtractionModel
from bmlab.models.orientation import Orientation
from bmlab.models.crop import Crop
from bmlab.models.setup import AVAILABLE_SETUPS
from bmlab.models.calibration_model import CalibrationModel
from bmlab.models.peak_selection_model import PeakSelectionModel
from bmlab.models.evaluation_model import EvaluationModel
from bmlab.serializer import Serializer


def get_session_file_path(source_file, create_folder=False):
    # If the raw data file is located in a 'RawData' folder,
    # we put the eval data file in an 'EvalData' folder and
    # don't append the 'session' string.
    file = Path(source_file).resolve()
    if file.parent.name == 'RawData':
        eval_folder = file.parents[1] / 'EvalData'
        # Create the evaluation folder if necessary
        if create_folder and not os.path.exists(eval_folder):
            os.mkdir(eval_folder)
        return Path(str(eval_folder / (str(file.name)[:-3] + '.h5')))
    else:
        return Path(str(source_file)[:-3] + '.session.h5')


def get_source_file_path(session_file):
    # If the session file is located in a 'EvalData' folder,
    # we find the source file in an 'RawData' folder
    file = Path(session_file).resolve()
    if file.parent.name == 'EvalData':
        raw_folder = file.parents[1] / 'RawData'
        return Path(str(raw_folder / (str(file.name)[:-3] + '.h5')))
    else:
        return Path(str(session_file)[:-11] + '.h5')


def get_valid_source(path):
    path = Path(path)
    # Check whether this file exists at all
    if not os.path.exists(path):
        raise FileNotFoundError(
            errno.ENOENT,
            "The file '{}' does not exist."
            .format(path),
            path
        )

    # If this is a session file, we need to check whether
    # the source file exists and is valid.
    if is_session_file(path):
        source_file_path = get_source_file_path(path)
        # If the session file is valid, but the source file
        # does not exist, we raise a file not found error.
        if not os.path.exists(source_file_path):
            raise FileNotFoundError(
                errno.ENOENT,
                "Could not find the corresponding source"
                " data file '{}' for session file '{}'."
                .format(source_file_path, path)
                + " Please ensure the source data file exists.",
                source_file_path
            )
        if is_source_file(source_file_path):
            return source_file_path
        # If the session file is valid, but the source file
        # is not, we raise a BmlabInvalidFileError
        else:
            raise BmlabInvalidFileError(
                errno.ENOENT,
                "Could not open file '{}.".format(source_file_path)
                + " The file is not a valid BrillouinAcquisition file.",
                source_file_path
            )

    # If this is a source file, just return its path
    if is_source_file(path):
        return path

    # If we end up here,
    # the file provided was neither a bmlab session file
    # nor a raw data file from BrillouinAcquisition.
    raise BmlabInvalidFileError(
        errno.ENOENT,
        "Could not open file '{}'.".format(path)
        + " The provided file is neither a valid"
          " BrillouinAcquisition nor bmlab file.",
        path
    )


class BmlabInvalidFileError(FileNotFoundError):
    def __init__(self, *args, **kwargs):
        super(BmlabInvalidFileError, self).__init__(*args, **kwargs)


class Session(Serializer):
    """
    Session stores information about the current file
    to be processed.

    Session is a singleton. It can be accessed by
    calling the get_instance() method.
    """

    __instance = None

    def __init__(self):
        """
        Constructor of Session class. Since Session is a singleton,
        users should not call it but instead use the get_instance
        method.
        """
        if Session.__instance is not None:
            raise Exception('Session is a singleton!')
        else:
            Session.__instance = self
            self.clear()

    def current_repetition(self):
        """
        Returns the repetition currently selected in data tab
        """
        if self.file and self._current_repetition_key:
            return self.file.get_repetition(self._current_repetition_key)
        return None

    def set_current_repetition(self, rep_key):
        self._current_repetition_key = rep_key

    def extraction_model(self):
        """
        Returns ExtractionModel instance for currently selected repetition
        """
        return self.extraction_models.get(self._current_repetition_key)

    def calibration_model(self):
        return self.calibration_models.get(self._current_repetition_key)

    def peak_selection_model(self):
        return self.peak_selection_models.get(self._current_repetition_key)

    def evaluation_model(self):
        return self.evaluation_models.get(self._current_repetition_key)

    def set_image_shape(self):
        """
        We set the extraction model image shape for every repetition when
        - a file gets loaded
        - the image orientation changes
        """
        if self.file is None:
            return

        repetitions = self.file.repetition_keys()
        for repetition in repetitions:
            payload = self.file.get_repetition(repetition).payload
            image_keys = payload.image_keys()
            # If no images are available, skip this repetition
            if not image_keys:
                continue
            imgs = payload.get_image(image_keys[0])
            if imgs is None:
                continue
            img = self.crop.apply(self.orientation.apply(imgs[0, ...]))
            self.extraction_models.get(repetition).set_image_shape(img.shape)

    def set_arc_width(self):
        """
        We set the arc_width for arc extraction on file load.
        """
        if self.file is None:
            return

        repetitions = self.file.repetition_keys()
        for repetition in repetitions:
            payload = self.file.get_repetition(repetition).payload
            image_keys = payload.image_keys()
            if not image_keys:
                continue
            binning_factor = payload.get_binning_factor(image_keys[0])

            em = self.extraction_models.get(repetition)
            arc_width = math.ceil(em.arc_width / binning_factor)
            em.set_arc_width(arc_width)

    @staticmethod
    def get_instance():
        """
        Returns the singleton instance of Session

        Returns
        -------
        out: Session
        """
        if Session.__instance is None:
            Session()
        return Session.__instance

    def set_file(self, file_name):
        """
        Set the file to be processed.

        Loads the corresponding data from HDF file.

        Parameters
        ----------
        file_name : Path
            The file name.

        """
        # Get the source file in case it's a session file
        file_name = get_valid_source(file_name)
        # There is no valid source file
        if file_name is None:
            raise Exception('No source data file found')

        try:
            file = BrillouinFile(file_name)
        except Exception as e:
            raise e
        else:
            """ Only load data if the file could be opened """
            session = Session.get_instance()
            session.file = file
            session.extraction_models = {
                key: ExtractionModel()
                for key in self.file.repetition_keys()
            }
            session.calibration_models = {
                key: CalibrationModel()
                for key in self.file.repetition_keys()
            }
            session.peak_selection_models = {
                key: PeakSelectionModel()
                for key in self.file.repetition_keys()
            }
            session.evaluation_models = {
                key: EvaluationModel()
                for key in self.file.repetition_keys()
            }
            self.set_image_shape()
            self.set_arc_width()
            # Initialize current setup
            session.setup = AVAILABLE_SETUPS[0]

        try:
            self.load(file_name)
            self.set_arc_width()
        except Exception as e:
            raise e
        else:
            Session.get_instance().file = file

    def get_calib_keys(self, sort_by_time=False):
        if self.current_repetition() is None:
            return None
        return self.current_repetition()\
            .calibration.image_keys(sort_by_time=sort_by_time)

    def get_calibration_image(self, calib_key, frame_num=None):
        if self.current_repetition() is None:
            return None
        imgs = self.current_repetition().calibration.get_image(calib_key)
        if frame_num is not None:
            imgs = imgs[frame_num, ...]
        return self.crop.apply(self.orientation.apply(imgs))

    def get_calibration_image_count(self, calib_key):
        return self.current_repetition()\
            .calibration.get_image_count(calib_key)

    def get_calibration_time(self, calib_key):
        if self.current_repetition() is None:
            return None
        return self.current_repetition().calibration.get_time(calib_key)

    def get_calibration_exposure(self, calib_key):
        if self.current_repetition() is None:
            return None
        return self.current_repetition().calibration.get_exposure(calib_key)

    def get_calibration_binning(self, calib_key):
        if self.current_repetition() is None:
            return None
        return self.current_repetition().calibration.get_binning(calib_key)

    def get_calibration_binning_factor(self, calib_key):
        if self.current_repetition() is None:
            return None
        return self.current_repetition()\
            .calibration.get_binning_factor(calib_key)

    def get_image_keys(self, sort_by_time=False):
        if self.current_repetition() is None:
            return None
        return self.current_repetition()\
            .payload.image_keys(sort_by_time=sort_by_time)

    def get_payload_image(self, image_key, frame_num=None):
        if self.current_repetition() is None:
            return None
        imgs = self.current_repetition().payload.get_image(image_key)
        if imgs is None:
            return None
        if frame_num is not None:
            imgs = imgs[frame_num, ...]
        return self.crop.apply(self.orientation.apply(imgs))

    def get_payload_image_count(self, calib_key):
        return self.current_repetition().payload.get_image_count(calib_key)

    def get_payload_time(self, image_key):
        if self.current_repetition() is None:
            return None
        return self.current_repetition().payload.get_time(image_key)

    def get_payload_exposure(self, image_key):
        if self.current_repetition() is None:
            return None
        return self.current_repetition().payload.get_exposure(image_key)

    def get_payload_binning(self, image_key):
        if self.current_repetition() is None:
            return None
        return self.current_repetition().payload.get_binning(image_key)

    def get_payload_binning_factor(self, image_key):
        if self.current_repetition() is None:
            return None
        return self.current_repetition()\
            .payload.get_binning_factor(image_key)

    def get_payload_resolution(self):
        if self.current_repetition() is None:
            return None
        return self.current_repetition().payload.resolution

    def get_payload_positions(self):
        if self.current_repetition() is None:
            return None
        positions = self.current_repetition().payload.positions
        if positions is None:
            return None
        # We need to correctly transpose the array to have the
        # axes in order x-y-z
        for axis in positions:
            positions[axis] = transpose(positions[axis], axes=(1, 2, 0))
        return positions

    def current_fluorescence_repetition(self):
        """
        Returns the Fluorescence-mode repetition captured *during*
        the currently selected Brillouin repetition's own measurement
        (by acquisition timing - see
        bmlab.export.timing.classify_timing()), or None if there is
        none.

        This is NOT necessarily the Fluorescence repetition sharing
        the same repetition key: BrillouinAcquisition starts a new
        Fluorescence repetition alongside a Brillouin one specifically
        for its brightfield-overview images
        (saveOverviewBrightfieldPerZ), but Fluorescence's own
        repetition counter also advances for every standalone
        Fluorescence capture triggered independently (e.g. a single
        "before"/"after" snapshot) - so on a file with both kinds of
        capture, the two counters drift apart and a same-key lookup
        can silently return an unrelated Fluorescence repetition
        instead. Falls back to the same-key repetition if timing
        classification finds nothing (e.g. no Brillouin measurement
        has any images yet to classify against), for graceful
        degradation rather than always returning None.
        """
        if self.file is None or self._current_repetition_key is None:
            return None
        # Local import: bmlab.export imports bmlab.session's own
        # Session class at module load time (see e.g.
        # fluorescence_export.py), so a module-level import here
        # would be circular.
        from bmlab.export.timing import get_brillouin_windows, \
            classify_timing
        brillouin_windows = get_brillouin_windows(self.file)
        for fl_key in self.file.repetition_keys(FLUORESCENCE_GROUP):
            fl_rep = self.file.get_repetition(fl_key, FLUORESCENCE_GROUP)
            image_keys = fl_rep.payload.image_keys(sort_by_time=True)
            if not image_keys:
                continue
            start = fl_rep.payload.get_date(image_keys[0])
            end = fl_rep.payload.get_date(image_keys[-1])
            timing = classify_timing(start, end, brillouin_windows)
            if timing is not None \
                    and timing[0] == self._current_repetition_key \
                    and timing[1] == 'during':
                return fl_rep
        if self._current_repetition_key in \
                self.file.repetition_keys(FLUORESCENCE_GROUP):
            return self.file.get_repetition(
                self._current_repetition_key, FLUORESCENCE_GROUP)
        return None

    def has_surface_scan(self):
        rep = self.current_repetition()
        return rep is not None and rep.payload.has_surface_scan()

    def get_surface_scan_data(self):
        rep = self.current_repetition()
        if rep is None:
            return None
        return rep.payload.get_surface_scan_data()

    def has_overview_brightfield(self):
        rep = self.current_repetition()
        return rep is not None and rep.payload.has_overview_brightfield()

    def get_overview_brightfield_positions(self):
        rep = self.current_repetition()
        if rep is None:
            return None
        return rep.payload.get_overview_brightfield_positions()

    def get_overview_brightfield_keys(self, sort_by_time=False):
        rep = self.current_fluorescence_repetition()
        if rep is None:
            return []
        return rep.payload.image_keys_by_channel(
            OVERVIEW_BRIGHTFIELD_CHANNEL, sort_by_time=sort_by_time)

    def get_overview_brightfield_image(self, image_key, frame_num=None):
        rep = self.current_fluorescence_repetition()
        if rep is None:
            return None
        imgs = rep.payload.get_image(image_key)
        if imgs is None:
            return None
        if frame_num is not None:
            imgs = imgs[frame_num, ...]
        return self.orientation.apply(imgs)

    def get_fluorescence_position(self, image_key):
        rep = self.current_fluorescence_repetition()
        if rep is None:
            return None
        return rep.payload.get_position(image_key)

    def get_fluorescence_stage_position(self, image_key):
        rep = self.current_fluorescence_repetition()
        if rep is None:
            return None
        return rep.payload.get_stage_position(image_key)

    def clear(self):
        """
        Close connection to loaded file.
        """

        # Global session data:
        if hasattr(self, 'file') and self.file is not None:
            self.file.close()
            self.file = None
        else:
            self.file = None

        self.orientation = Orientation()
        self.crop = Crop()
        self.setup = None

        # Session data by repetition:
        self.extraction_models = {}
        self.calibration_models = {}
        self.evaluation_models = {}
        self.peak_selection_models = {}

        self._current_repetition_key = None

    def set_setup(self, setup):
        self.setup = setup

    def set_rotation(self, num_rots):
        self.orientation.set_rotation(num_rots)
        self.set_image_shape()

    def set_reflection(self, **kwargs):
        self.orientation.set_reflection(**kwargs)
        self.set_image_shape()

    def set_crop_bounds(self, bounds):
        """
        Restricts every calibration/payload image read from the file to
        the given (x_min, x_max, y_min, y_max) pixel bounds (in the same
        array-index convention as extraction points, i.e. img[x, y]).

        Like set_rotation()/set_reflection(), this does not shift or
        clear any already-placed extraction points - draw the crop before
        placing points for a calibration, or re-place them afterwards.
        """
        self.crop.set_bounds(bounds)
        self.set_image_shape()

    def clear_crop(self):
        self.crop.clear()
        self.set_image_shape()

    def get_crop_bounds(self):
        return self.crop.bounds

    def save(self):
        if self.file is None:
            return

        session_file_name = get_session_file_path(self.file.path,
                                                  create_folder=True)

        with h5py.File(session_file_name, 'w') as f:
            self.serialize(f, 'session', skip=['file'])
            # Store the current bmlab version
            f.attrs['version'] = 'bmlab_' + version

    def load(self, h5_file_name):

        session_file_name = get_session_file_path(h5_file_name)

        if not os.path.exists(session_file_name):
            return

        if not is_session_file(session_file_name):
            raise BmlabInvalidFileError(
                errno.ENOENT,
                "Could not load the session file '{}'."
                .format(session_file_name)
                + " Please ensure the session file is valid.",
                session_file_name
            )

        with h5py.File(session_file_name, 'r') as f:
            new_session = Serializer.deserialize(f['session'])
            session = Session.get_instance()
            for var_name, var_value in new_session.__dict__.items():
                session.__dict__[var_name] = var_value

        self.run_migrations()

    def run_migrations(self):
        """
        This functions runs migrations
        that cannot be done in the respective models only,
        as they might require access to other models.

        """
        session = Session.get_instance()
        repetitions = self.file.repetition_keys()
        for repetition in repetitions:
            session.set_current_repetition(repetition)
            # Migrations from 0.5.1 to 0.6.0
            # Convert evaluation regions from pix to GHz
            # @since 0.6.0
            psm = session.peak_selection_model()
            cm = session.calibration_model()
            evm = session.evaluation_model()
            if not hasattr(psm, 'brillouin_regions_f'):
                evm.invalidate_results()
            # We use the first available measurement image here
            image_keys = session.get_image_keys()
            time = session.get_payload_time(image_keys[0]) \
                if image_keys else None

            def region_to_region_f(regions, region):
                region_f = cm.get_frequency_by_time(time, region)
                if region_f is not None:
                    regions.append(region_f)
                return regions

            if not hasattr(psm, 'brillouin_regions_f'):
                psm.brillouin_regions_f = reduce(
                    region_to_region_f,
                    psm.brillouin_regions,
                    []
                )
                delattr(psm, 'brillouin_regions')

            if not hasattr(psm, 'rayleigh_regions_f'):
                psm.rayleigh_regions_f = reduce(
                    region_to_region_f,
                    psm.rayleigh_regions,
                    []
                )
                delattr(psm, 'rayleigh_regions')
