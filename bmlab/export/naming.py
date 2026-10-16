import re


def sanitize_for_filename(name):
    """
    Turns an arbitrary channel/label string (which may contain spaces
    or other characters that are awkward in filenames) into a
    filename-safe CamelCase token, e.g. "Brightfield z overview"
    -> "BrightfieldZOverview".
    """
    words = re.split(r'[^0-9A-Za-z]+', name)
    return ''.join(word[:1].upper() + word[1:] for word in words if word)


def repetition_or_timestamp_tag(payload, image_key):
    """
    Returns a filename tag identifying the image at `image_key`: either
    the Brillouin repetition it was captured as part of ('_BMrepN',
    matching the combined CSV's own '_BMrepN_data.csv' naming - see
    BrillouinExport), from MeasurementData.get_brillouin_repetition_index(),
    or, for a standalone capture with no such repetition (e.g. a manual
    Fluorescence-tab "Acquire" snapshot - not tied to any Brillouin run,
    so there is nothing to guess it belongs to), its own capture
    timestamp instead. This is a real, recorded relationship - not a
    guess from comparing capture-time windows - and only exists on
    files from a BrillouinAcquisition version that records it; older
    files always fall back to the timestamp.

    Returns
    -------
    out: str
        '_BMrep<N>' or '_<YYYYMMDD-HHMMSS>' - '' if this image has
        neither a repetition index nor a usable date.
    """
    brillouin_index = payload.get_brillouin_repetition_index(image_key)
    if brillouin_index is not None:
        return f'_BMrep{brillouin_index}'
    date = payload.get_date(image_key)
    if date:
        return f'_{date:%Y%m%d-%H%M%S}'
    return ''


def timestamp_tag(date):
    """'YYYYMMDD-HHMMSS' of a capture time ('' if there is none)."""
    return f'{date:%Y%m%d-%H%M%S}' if date else ''


def following_brillouin_repetition(file, when):
    """
    The key of the Brillouin repetition that an image taken at `when`
    belongs to when it was not captured as part of one (a manual
    Fluorescence-tab snapshot): the one acquired next - an image is
    taken to prepare the acquisition that follows it. Repetitions that
    no longer exist in the file (aborted starts that were removed) are
    skipped automatically, so such images move on to the next one that
    does.

    Returns None if no Brillouin repetition started after `when` (the
    image was taken after the last acquisition) or `when` is unknown.
    """
    if not when:
        return None
    later = []
    for key in file.repetition_keys():
        started = file.get_repetition(key).date
        if started > when:
            later.append((started, key))
    return min(later)[1] if later else None


def running_brillouin_repetition(file, when, recorded=None):
    """
    The key of the Brillouin repetition an image captured DURING a
    Brillouin run belongs to: the one that was running at `when`, i.e.
    the latest one that had started. This does not rely on the
    repetition number stored with the image - older acquisition
    software numbered a run's images per Start instead of by the
    repetition's key in the file, so after an aborted start (or with
    several runs in one file) the stored number can be wrong.

    Images taken before the first remaining repetition started (an
    aborted run that was removed from the file) belong to the one
    acquired next, like images taken by hand. Without a usable capture
    time, the stored number `recorded` is used.
    """
    if when:
        running = []
        for key in file.repetition_keys():
            started = file.get_repetition(key).date
            if started <= when:
                running.append((started, key))
        if running:
            return max(running)[1]
        following = following_brillouin_repetition(file, when)
        if following is not None:
            return following
    return None if recorded is None else str(recorded)


def folder_name(kind, brillouin_key):
    """'<kind>_BMrep<N>', or '<kind>_afterBM' (taken after the last one)."""
    if brillouin_key is None:
        return f'{kind}_afterBM'
    return f'{kind}_BMrep{brillouin_key}'
