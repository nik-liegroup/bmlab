from bmlab.serializer import Serializer


class Crop(Serializer):
    """
    A rectangular pixel crop applied to every calibration and payload
    image (after orientation), restricting all downstream extraction and
    fitting to the given field of view.

    Bounds use the same (x, y) array-index convention as extraction
    points and arcs, i.e. img[x, y] with axis 0 = x, axis 1 = y - see
    bmlab.image.extract_lines_along_arc(). This lets a too-wide
    acquisition-camera ROI (e.g. one that captured multiple repeated VIPA
    orders instead of a single one) be restricted after the fact, without
    needing to re-acquire the data.
    """

    def __init__(self, bounds=None):
        # bounds: (x_min, x_max, y_min, y_max), integer indices into the
        # oriented image, or None for "no crop".
        self.bounds = bounds

    def post_deserialize(self):
        # Serializer.do_serialize() silently omits attributes that are
        # None at save time (see Serializer.do_serialize), so a saved
        # session where no crop was ever set has no 'bounds' entry at
        # all in the file - deserialize() then leaves this object
        # without a 'bounds' attribute entirely, instead of restoring
        # the __init__ default of None.
        if not hasattr(self, 'bounds'):
            self.bounds = None

    def set_bounds(self, bounds):
        if bounds is None:
            self.bounds = None
            return
        x_min, x_max, y_min, y_max = bounds
        self.bounds = (
            int(round(min(x_min, x_max))), int(round(max(x_min, x_max))),
            int(round(min(y_min, y_max))), int(round(max(y_min, y_max))),
        )

    def clear(self):
        self.bounds = None

    def is_active(self):
        return self.bounds is not None

    def apply(self, img):
        if self.bounds is None or img is None:
            return img
        x_min, x_max, y_min, y_max = self.bounds
        if img.ndim == 2:
            x_max = min(x_max, img.shape[0])
            y_max = min(y_max, img.shape[1])
            return img[x_min:x_max, y_min:y_max]
        if img.ndim == 3:
            x_max = min(x_max, img.shape[1])
            y_max = min(y_max, img.shape[2])
            return img[:, x_min:x_max, y_min:y_max]
        return img
