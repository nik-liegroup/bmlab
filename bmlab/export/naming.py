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
