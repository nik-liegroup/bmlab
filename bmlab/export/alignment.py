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


def get_point_to_pixel_matrix(tmatrix, pixels_per_um, anchor_um, image_shape,
                              local_translate, placement_offset_px=(0.0, 0.0)):
    """
    Builds the 3x3 affine matrix M mapping an arbitrary absolute stage
    position (x, y, in um - the same coordinates written into a
    BrillouinExport data CSV) directly onto its pixel location (col,
    row) in one exported overview image, i.e.
    [col, row, 1] = M @ [x_um, y_um, 1].

    `anchor_um` is the (x, y) stage position that image's own raw
    camera frame was captured/targeted at (see
    Payload.get_position()/get_overview_brightfield_positions()) -
    absent any finer (e.g. principal-point) calibration, that position
    is assumed to sit at the raw frame's own geometric center.
    `image_shape` is that raw frame's (height, width), and
    `local_translate` is the `translate` warp_local() returned for it
    (both already computed by the caller while building the actual
    exported image, so the matrix is guaranteed to agree with it).
    `placement_offset_px`, for a tiled mosaic, additionally places the
    tile within the shared canvas (see
    OverviewBrightfieldExport._export_tiled) - (0, 0) for a plain,
    single-position stack.

    Returns
    -------
    np.ndarray or None
        None if `tmatrix`, `pixels_per_um` or `anchor_um` is None (no
        scale calibration, or no recorded position for this image).
    """
    if tmatrix is None or pixels_per_um is None or anchor_um is None:
        return None

    height, width = image_shape[0], image_shape[1]
    center = np.matrix([[(width - 1) / 2], [(height - 1) / 2], [1]])
    local_center = local_translate * tmatrix * center

    # Same direction+scale composition as um_offset_to_pixels(): a
    # stage (dx, dy) offset (um) rotated into the image's own pixel
    # axes, then scaled from um into pixels.
    linear = pixels_per_um * np.asarray(tmatrix)[:2, :2]
    anchor = np.array([anchor_um[0], anchor_um[1]])
    translation = (
        np.array([local_center[0, 0], local_center[1, 0]])
        + np.array(placement_offset_px)
        - linear @ anchor
    )

    return np.array([
        [linear[0, 0], linear[0, 1], translation[0]],
        [linear[1, 0], linear[1, 1], translation[1]],
        [0, 0, 1],
    ])
