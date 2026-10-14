"""
Writes into Brillouin raw files what BrillouinAcquisition writes with every brightfield image since the overview engine (overview
images) and the fix of the Fluorescence mode (FL images), for the images taken before, so that bmlab has
one way to place an image:

  - image_centre_shift_x/y_um (and fov_compensation_x/y_um = 0 where absent): the middle of the image is the stage target minus
    this shift. A target puts the laser on a point and the laser sits where its marker was set in the camera image, not in the
    middle: shift = scanner - A * (middle - origin) + FOV offset of the objective (Brillouin::captureOverviewEvent()). Computed
    from the scale calibration written with the image's repetition (the objective that took it; the scanner position is the
    laser marker) and the image's camera ROI (a cropped ROI enters with its offset on the sensor). Written for every image that
    has none: the overview images from before the engine (taken with the laser on the planned point), the per-point images and
    the images taken in the Fluorescence tab.
  - stage_position_raw_x/y_um, only for the images taken in the Fluorescence tab (their 'position' is in the grid frame in force
    when they were taken, before the run's Start): stage + scanner of the same scale calibration - the position the acquisition
    records when the images are taken (AcquisitionMode::writeScaleCalibration()).

Only attributes are written. A file is marked with the attribute 'bmlab_image_centres' and not touched again.

    python add_image_centres.py FILE [FILE ...]                      dry run
    python add_image_centres.py --apply FILE [...]                   write
"""
import argparse
import datetime
import sys

import h5py
import numpy as np

OVERVIEW_CHANNELS = ('Brightfield z overview', 'Brightfield overview')
PER_POINT_CHANNEL = 'Brightfield per-point'
MARK = 'bmlab_image_centres'


def _text(attrs, name):
    value = attrs.get(name)
    if value is None:
        return None
    value = np.ravel(value)[0]
    return value.decode('utf-8') if hasattr(value, 'decode') else str(value)


def _number(attrs, name):
    value = attrs.get(name)
    return None if value is None else float(np.ravel(value)[0])


def _point(cal, name):
    if name not in cal:
        return None
    return np.array([float(cal[name].attrs['x'][0]), float(cal[name].attrs['y'][0])])


def calibration(payload):
    """What the shift needs from the repetition's scale calibration, or None."""
    cal = payload.get('scaleCalibration')
    if cal is None:
        return None
    out = {name: _point(cal, name) for name in ('positionStage', 'positionScanner', 'pixToMicrometerX', 'pixToMicrometerY',
                                                 'origin', 'fovOffset')}
    if any(out[name] is None for name in ('positionStage', 'positionScanner', 'pixToMicrometerX', 'pixToMicrometerY')):
        return None
    has_fov = _number(cal.attrs, 'hasFovOffset')
    out['fov'] = out['fovOffset'] if has_fov and out['fovOffset'] is not None else np.zeros(2)
    out['origin'] = out['origin'] if out['origin'] is not None else np.zeros(2)
    return out


def roi_offset(attrs):
    """(left, top) of the camera ROI in binned sensor pixels."""
    offset = []
    for start, physical, binned in (('ROI_left', 'ROI_width_physical', 'ROI_width_binned'),
                                    ('ROI_top', 'ROI_height_physical', 'ROI_height_binned')):
        value, size_physical, size_binned = _number(attrs, start), _number(attrs, physical), _number(attrs, binned)
        binning = size_physical / size_binned if size_physical and size_binned else 1.0
        offset.append(0.0 if value is None else value / binning)
    return np.array(offset)


def centre_shift(cal, attrs, shape):
    """scanner - A * (middle - origin) + FOV offset; shape = (height, width) of the stored image."""
    middle = roi_offset(attrs) + 0.5 * np.array([shape[1], shape[0]], float)
    d = middle - cal['origin']
    a = d[0] * cal['pixToMicrometerX'] + d[1] * cal['pixToMicrometerY']
    return cal['positionScanner'] - a + cal['fov']


def plan(f):
    """[(dataset, {attribute: value})] for every image that lacks the attributes."""
    changes = []
    for fk in f['Fluorescence']:
        payload = f['Fluorescence'][fk]['payload']
        data = payload['data']
        cal = None
        for key in data:
            dataset = data[key]
            attrs = dataset.attrs
            if 'image_centre_shift_x_um' in attrs:
                continue
            cal = cal or calibration(payload)
            if cal is None:
                continue
            channel = _text(attrs, 'channel')
            shift = centre_shift(cal, attrs, dataset.shape[-2:])
            values = {'image_centre_shift_x_um': shift[0], 'image_centre_shift_y_um': shift[1]}
            if 'fov_compensation_x_um' not in attrs:
                values.update({'fov_compensation_x_um': 0.0, 'fov_compensation_y_um': 0.0})
            if channel not in OVERVIEW_CHANNELS and channel != PER_POINT_CHANNEL \
                    and 'stage_position_raw_x_um' not in attrs:
                raw = cal['positionStage'] + cal['positionScanner']
                values.update({'stage_position_raw_x_um': raw[0], 'stage_position_raw_y_um': raw[1]})
            changes.append((dataset, values))
    return changes


def apply_file(path, apply):
    with h5py.File(path, 'r+' if apply else 'r') as f:
        if MARK in f.attrs:
            return 'already done', None
        changes = plan(f)
        kinds = {}
        for dataset, values in changes:
            kind = 'FL' if 'stage_position_raw_x_um' in values else _text(dataset.attrs, 'channel')
            kinds[kind] = kinds.get(kind, 0) + 1
        summary = ', '.join(f'{n} {k}' for k, n in sorted(kinds.items())) or 'nothing to add'
        if not apply:
            return summary, None
        for dataset, values in changes:
            for name, value in values.items():
                dataset.attrs[name] = np.array([float(value)])
        f.attrs[MARK] = np.array([datetime.datetime.now().isoformat().encode()])
    return summary, True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='*')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    for path in args.files:
        summary, written = apply_file(path, args.apply)
        print(('written ' if written else '') + f'{path}: {summary}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
