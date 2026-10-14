import numpy as np
import tifffile


def write_tiff(filename, pages, rgb=False):
    """
    Writes `pages` as a TIFF, keeping the pixel type (8 or 16 bit are
    written as they are, nothing is rescaled or truncated). `pages` is
    one (rows, cols) image, or a list of them for a multi-page stack;
    with `rgb`, (rows, cols, 3) images.
    """
    if isinstance(pages, np.ndarray) and pages.ndim == (3 if rgb else 2):
        data = pages
    else:
        data = np.stack(list(pages))
    if data.dtype == bool:
        data = data.astype(np.uint8) * 255
    tifffile.imwrite(
        filename, data, photometric='rgb' if rgb else 'minisblack')
