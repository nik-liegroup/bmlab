def repetition_selected(configuration, repetition_key):
    """
    Whether Brillouin repetition `repetition_key` is included by the
    export configuration's 'brillouin'/'repetitions' selection (see
    ExportController.get_configuration()) - None there means "every
    repetition", the default for every caller except BMicro's own
    export dialog, where the user can restrict a single-file export to
    a chosen subset of repetitions.

    Parameters
    ----------
    configuration: dict
        The export configuration passed to *Export.export().
    repetition_key: str
        A Brillouin repetition key, e.g. '0'.
    """
    selected = configuration.get('brillouin', {}).get('repetitions')
    return selected is None or repetition_key in selected
