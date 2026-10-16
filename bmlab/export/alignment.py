import csv
import os

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
    matrix alone does NOT give pixels - see get_point_to_pixel_matrix().

    Returns
    -------
    np.matrix or None
        None if `scale_calibration` is None.
    """
    if scale_calibration is None:
        return None
    n = np.linalg.norm(np.array(scale_calibration['micrometerToPixX']))
    matrix = np.matrix([
        [-1 * scale_calibration['micrometerToPixY'][1],
         -1 * scale_calibration['micrometerToPixY'][0], 0],
        [-1 * scale_calibration['micrometerToPixX'][1],
         -1 * scale_calibration['micrometerToPixX'][0], 0],
        [0, 0, n]
    ]) / n
    return _nearest_axis_turn(matrix)


def _nearest_axis_turn(matrix):
    """
    The camera is mounted turned and/or mirrored by a multiple of 90
    degrees against the stage; what the calibration measured beyond that
    (a pixel size that differs by a fraction of a percent between the
    axes) is measurement noise. The matrix is therefore taken as exactly
    that turn/mirror: each row keeps only its dominant entry, as +1 or
    -1, in different columns.
    """
    linear = np.asarray(matrix)[:2, :2]
    first = int(np.argmax(np.abs(linear[0])))
    exact = np.zeros((2, 2))
    exact[0, first] = np.sign(linear[0, first]) or 1.0
    exact[1, 1 - first] = np.sign(linear[1, 1 - first]) or 1.0
    result = np.array(matrix, dtype=float)
    result[:2, :2] = exact
    return np.asmatrix(result)


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


def orient_image(img_data, tmatrix):
    """
    Turns/mirrors an image into the stage's axes by plain index
    remapping: no interpolation, the pixel type is kept, and no row or
    column is lost. `tmatrix` (see get_tmatrix()) is always an exact turn
    by a multiple of 90 degrees, optionally mirrored. Works for
    (rows, cols) and (rows, cols, channels) arrays.

    Returns
    -------
    oriented: np.ndarray
    translate: np.matrix
        translate * tmatrix maps a pixel of the original image onto its
        pixel in `oriented` (the same convention warp_local() uses).
    """
    linear = np.asarray(tmatrix)[:2, :2]
    rounded = np.round(linear)
    if not np.allclose(linear, rounded) \
            or abs(abs(np.linalg.det(rounded)) - 1.0) > 1e-6:
        raise ValueError(
            f'not a turn by a multiple of 90 degrees: {linear.tolist()}')

    height, width = img_data.shape[0], img_data.shape[1]
    corners = [[0, width - 1, 0, width - 1],
               [0, 0, height - 1, height - 1],
               [1, 1, 1, 1]]
    corners_warped = tmatrix * corners
    dx = corners_warped[0, :].min()
    dy = corners_warped[1, :].min()
    out_width = int(round(corners_warped[0, :].max() - dx)) + 1
    out_height = int(round(corners_warped[1, :].max() - dy)) + 1
    translate = np.matrix([
        [1, 0, -dx],
        [0, 1, -dy],
        [0, 0, 1]
    ])

    inverse = np.linalg.inv(rounded)
    ys, xs = np.mgrid[0:out_height, 0:out_width]
    xs = xs + dx
    ys = ys + dy
    x_in = np.round(inverse[0, 0] * xs + inverse[0, 1] * ys).astype(int)
    y_in = np.round(inverse[1, 0] * xs + inverse[1, 1] * ys).astype(int)
    return img_data[y_in, x_in], translate


def calibration_stage_to_pixel(scale_calibration):
    """
    The 2x2 matrix mapping a stage offset (dx_um, dy_um) onto the
    offset (d_col, d_row) of the same scene in the RAW camera image as
    it is stored in the file.

    The scale calibration describes how the acquisition software
    converts a stage position into a pixel position of its live view:
    pixel = x * micrometerToPixX + y * micrometerToPixY (see
    ScanControl::microMeterToPix(); each is the pixel vector of one um
    along that stage axis, entries ordered (col, row)). The images
    are stored with their rows (the image's y axis) running the other
    way than in that live view, so the second component - the row - has
    to be inverted. Checked against the tile overlaps of the 20x overview
    mosaics: with this matrix neighbouring tiles line up to a few pixels,
    which is what makes placing the tiles from their stage positions
    alone possible.
    """
    columns = np.array([scale_calibration['micrometerToPixX'],
                        scale_calibration['micrometerToPixY']],
                       dtype=float).T
    return np.diag([1.0, -1.0]) @ columns


def get_point_to_pixel_matrix(tmatrix, stage_map, anchor_um, image_shape,
                              local_translate):
    """
    Builds the 3x3 affine matrix M mapping an arbitrary absolute stage
    position (x, y, in um - the same coordinates written into a
    BrillouinExport data CSV) directly onto its pixel location (col,
    row) in one exported image, i.e.
    [col, row, 1] = M @ [x_um, y_um, 1].

    It is built the same way as the matrix of a stitched mosaic: in the
    RAW camera image the anchor position sits at the image centre and a
    stage offset moves by `stage_map` (see calibration_stage_to_pixel());
    the exported image is that raw image turned by `tmatrix` and shifted
    by `local_translate` (both from orient_image()), so
    M = local_translate * tmatrix * (raw matrix).

    `anchor_um` is the (x, y) stage position that image's own raw
    camera frame was captured/targeted at (see
    Payload.get_position()/get_overview_brightfield_positions()) -
    absent any finer (e.g. principal-point) calibration, that position
    is assumed to sit at the raw frame's own geometric centre.
    `image_shape` is that raw frame's (height, width).

    Returns
    -------
    np.ndarray or None
        None if `tmatrix`, `stage_map` or `anchor_um` is None (no
        scale calibration, or no recorded position for this image).
    """
    if tmatrix is None or stage_map is None or anchor_um is None:
        return None

    height, width = image_shape[0], image_shape[1]
    centre = np.array([(width - 1) / 2, (height - 1) / 2])
    raw_matrix = np.eye(3)
    raw_matrix[:2, :2] = stage_map
    raw_matrix[:2, 2] = centre - stage_map @ np.array(
        [anchor_um[0], anchor_um[1]])
    return np.asarray(local_translate * tmatrix) @ raw_matrix


def write_transform_csv(matrix, image_filename):
    """
    Writes the 3x3 matrix (see get_point_to_pixel_matrix()) mapping an
    absolute stage position (x, y, um) to its pixel location in
    `image_filename`'s image, as a plain 3-row/3-column CSV (no
    header) - same layout as BrainFusion's AFM loader already expects
    for its own GridInversionMatrix.csv - into a 'TransformMatrices'
    subfolder next to the image, one per image rather than mixed in
    among the images themselves. Writes nothing if `matrix` is None
    (no scale calibration, or no recorded position for this image) -
    shared by every exporter that places an image in the stage frame
    (OverviewBrightfieldExport, FluorescenceExport,
    FluorescenceCombinedExport), so they all agree on this layout.
    """
    if matrix is None:
        return
    matrix_dir = image_filename.parent / 'TransformMatrices'
    if not os.path.exists(matrix_dir):
        os.makedirs(matrix_dir, exist_ok=True)
    csv_filename = matrix_dir / (image_filename.stem + '_transform.csv')
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerows(matrix.tolist())
