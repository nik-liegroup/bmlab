import numpy as np
from skimage import transform


def get_tmatrix(scale_calibration):
    """
    Builds the 3x3 rotation+reflection matrix that maps an image's own
    (local) pixel coordinates onto the stage's x-y axes, from a
    payload's scale calibration. This is normalized to unit magnitude
    (divided by the pixel/um scale, see get_pixels_per_um()) - for an
    isotropic calibration (the same um-per-pixel size along x and y,
    the normal case for a camera sensor) it is a pure rotation+
    reflection, with no separate scale or translation: it corrects the
    camera-to-stage axis mismatch, in the image's own native pixel
    units, and does not place the image at its absolute stage
    position. Converting a real-world (um) distance through this
    matrix alone does NOT give pixels - see um_offset_to_pixels().

    Returns
    -------
    np.matrix or None
        None if `scale_calibration` is None.
    """
    if scale_calibration is None:
        return None
    n = np.linalg.norm(np.array(scale_calibration['micrometerToPixX']))
    return np.matrix([
        [-1 * scale_calibration['micrometerToPixY'][1],
         -1 * scale_calibration['micrometerToPixY'][0], 0],
        [-1 * scale_calibration['micrometerToPixX'][1],
         -1 * scale_calibration['micrometerToPixX'][0], 0],
        [0, 0, n]
    ]) / n


def get_pixels_per_um(scale_calibration):
    """
    The (isotropic) pixel/um scale get_tmatrix() normalizes out -
    needed to convert a real-world (um) distance into pixels, which
    get_tmatrix()'s own unit-magnitude matrix alone cannot do.

    Returns
    -------
    float or None
        None if `scale_calibration` is None.
    """
    if scale_calibration is None:
        return None
    return float(
        np.linalg.norm(np.array(scale_calibration['micrometerToPixX'])))


def warp_local(img_data, tmatrix):
    """
    Warps a single image with `tmatrix` (see get_tmatrix()), then
    translates the result so its own bounding box starts at (0, 0) -
    i.e. the image is rotation/scale-corrected but still anchored to
    its own local origin, not to any absolute stage position.

    Returns
    -------
    warped: np.ndarray
        The warped image, NaN outside the original image's rotated
        footprint.
    shape: tuple
        (height, width) of `warped`.
    translate: np.matrix
        The 3x3 translation applied after `tmatrix` - combine with
        tmatrix (translate * tmatrix) to map a point in the original
        image's local pixel coordinates into `warped`'s own pixel
        coordinates, e.g. to place another image (or a position
        offset) consistently in the same frame.
    """
    height, width = img_data.shape[0], img_data.shape[1]
    corners = [[0, width - 1, 0, width - 1],
               [0, 0, height - 1, height - 1],
               [1, 1, 1, 1]]
    corners_warped = tmatrix * corners

    dx = corners_warped[0, :].min()
    dy = corners_warped[1, :].min()
    sx = corners_warped[0, :].max() - corners_warped[0, :].min()
    sy = corners_warped[1, :].max() - corners_warped[1, :].min()

    translate = np.matrix([
        [1, 0, -dx],
        [0, 1, -dy],
        [0, 0, 1]
    ])

    tform = transform.AffineTransform(
        matrix=np.linalg.inv(translate * tmatrix))
    shape = (int(np.ceil(sy)), int(np.ceil(sx)))

    warped = transform.warp(
        img_data, tform, output_shape=shape, cval=np.nan)

    return warped, shape, translate


def um_offset_to_pixels(tmatrix, pixels_per_um, dx_um, dy_um):
    """
    Converts a stage x-y offset (um) into the corresponding pixel
    offset in the local warped-image frame warp_local() produces:
    tmatrix (unit magnitude, see get_tmatrix()) rotates the offset
    into the image's own pixel-axis orientation, then pixels_per_um
    (see get_pixels_per_um()) scales that rotated offset from um into
    actual pixels. tmatrix has no translation component, so both
    steps apply to a pure offset vector directly.
    """
    offset = tmatrix * np.matrix([[dx_um], [dy_um], [0]])
    return (float(offset[0, 0]) * pixels_per_um,
            float(offset[1, 0]) * pixels_per_um)
