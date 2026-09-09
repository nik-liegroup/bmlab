import re


def get_brillouin_windows(file):
    """
    Returns the [start, end] capture-time window of every Brillouin
    repetition in `file` that actually has images (an aborted/restarted
    repetition has none and is skipped).

    Returns
    -------
    list of (repetition_key, start, end) tuples, sorted by start.
    """
    windows = []
    for repetition_key in file.repetition_keys('Brillouin'):
        repetition = file.get_repetition(repetition_key, 'Brillouin')
        image_keys = repetition.payload.image_keys(sort_by_time=True)
        if not image_keys:
            continue
        start = repetition.payload.get_date(image_keys[0])
        end = repetition.payload.get_date(image_keys[-1])
        windows.append((repetition_key, start, end))
    windows.sort(key=lambda w: w[1])
    return windows


def classify_timing(fl_start, fl_end, brillouin_windows):
    """
    Relates a Fluorescence repetition's capture-time span to the
    Brillouin repetition it is closest to in time.

    Parameters
    ----------
    fl_start, fl_end: datetime
        The first and last image capture time of the Fluorescence
        repetition (may be equal, for a single-image repetition).
    brillouin_windows: list of (repetition_key, start, end)
        As returned by get_brillouin_windows().

    Returns
    -------
    (brillouin_repetition_key, label) or None if there is no Brillouin
    repetition to relate to. label is one of 'before', 'during', 'after'
    - 'during' if the Fluorescence repetition's time span overlaps the
    Brillouin repetition's, otherwise 'before'/'after' relative to
    whichever Brillouin repetition is closest in time.
    """
    if not brillouin_windows:
        return None

    for repetition_key, start, end in brillouin_windows:
        if fl_start <= end and fl_end >= start:
            return repetition_key, 'during'

    def distance(window):
        _, start, end = window
        if fl_end < start:
            return (start - fl_end).total_seconds()
        return (fl_start - end).total_seconds()

    repetition_key, start, end = min(brillouin_windows, key=distance)
    label = 'before' if fl_end < start else 'after'
    return repetition_key, label


def sanitize_for_filename(name):
    """
    Turns an arbitrary channel/label string (which may contain spaces
    or other characters that are awkward in filenames) into a
    filename-safe CamelCase token, e.g. "Brightfield z overview"
    -> "BrightfieldZOverview".
    """
    words = re.split(r'[^0-9A-Za-z]+', name)
    return ''.join(word[:1].upper() + word[1:] for word in words if word)
