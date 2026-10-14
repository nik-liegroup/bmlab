"""
Writes a new Brillouin raw file that holds only some z planes of a file's repetitions - to separate the planes that were
measured with different objectives when the objective changed from plane to plane.

The file is written from the source; the source is only read. Per repetition the data points of the chosen planes are
copied unchanged (compressed chunks are copied as they are), their keys are renumbered to the new plane numbers (the
key of a point encodes its grid position: z * nx * ny + y * nx + x), and every dataset that depends on the planes
(positions-x/y/z, the sampled mask and path, the offsets per point, the resume plan) is cut to the same planes. The
calibration, background, images (Fluorescence) and everything else are copied as they are.

    python split_by_planes.py detect SOURCE.h5
        which objective each plane of each repetition was measured with (read from the objective switches of the
        overview events - see detect())
    python split_by_planes.py split SOURCE.h5 OUTPUT.h5 --planes 0:0,2,4 --planes 1:0,1,2,3,4 [--objective 10x]
        keeps planes 0, 2, 4 of repetition 0 and all planes of repetition 1 (repetitions not named are left out)
        --objective 10x: the file is labelled as measured with the 10x objective (scale calibration of the payload)
"""
import argparse
import json
import re
import sys

import h5py
import numpy as np

# datasets cut along the plane axis ((nz, nx, ny)), and per point (in the order the points were measured)
GRID_DATASETS = ('positions-x', 'positions-y', 'positions-z', 'positions-sampled-mask')
POINT_DATASETS = ('positions-sampled-x', 'positions-sampled-y', 'positions-sampled-z',
                  'positions-grid-offset-per-point-x-um', 'positions-grid-offset-per-point-y-um')
# a switch back to the measuring objective that is faster than this did not happen (the nosepiece stayed where it was)
QUICK_SWITCH_BACK_SECONDS = 1.5


def _scalar(payload, name):
    return float(np.ravel(payload[name][...])[0]) if name in payload else np.nan


def detect(payload):
    """
    {plane: objective slot name} for the planes of a repetition, or None if the repetition has no overview events with
    objective switches (then every plane has the objective of the repetition's scale calibration).

    The overview engine switches to the overview objective after a plane and back to the measuring one. On the stand of
    these files the switch back sometimes did not happen (it took 0.8 s instead of 2.1 s, and the next switch to the
    overview objective took 0.2 s: nothing to move): the next plane was measured with the overview objective, and the
    following cycle brought the nosepiece back. The state after event e therefore decides the objective of plane e
    (event 0 is the one before the first plane, event k + 1 follows plane k).
    """
    events = sorted({int(re.search(r'event-(\d+)-', n).group(1)) for n in payload
                     if n.startswith('positions-overview-event-')})
    if not events:
        return None
    measuring = 'measuring'
    state = []
    for e in events:
        back = _scalar(payload, f'positions-overview-event-{e:04d}-objective-switch-back-seconds')
        state.append(measuring if not back < QUICK_SWITCH_BACK_SECONDS else 'overview')
    # the plane that follows event e
    return {e: state[e] for e in range(len(events) - 1)}


def _subset_axis0(array, planes):
    return array[list(planes)]


def _plane_of_points(payload):
    index = np.ravel(payload['positions-resume-plan-index-z'][...])
    return index[1:1 + int(index[0])].astype(int)


def _split_repetition(src_rep, dst_rep, planes, objective):
    src_pl = src_rep['payload']
    nx, ny, nz = (int(src_pl.attrs[f'resolution-{a}'][0]) for a in 'xyz')
    per_plane = nx * ny
    planes = list(planes)
    new_of = {z: i for i, z in enumerate(planes)}
    for name in src_rep.attrs:
        dst_rep.attrs[name] = src_rep.attrs[name]
    for group in ('background', 'calibration'):
        if group in src_rep:
            src_rep.copy(group, dst_rep)
    dst_pl = dst_rep.create_group('payload')
    for name, value in src_pl.attrs.items():
        dst_pl.attrs[name] = value
    dst_pl.attrs['resolution-z'] = np.array([len(planes)], dtype=src_pl.attrs['resolution-z'].dtype)

    plane_of_point = _plane_of_points(src_pl)
    keep = np.isin(plane_of_point, planes)
    nominal_z = np.array(src_pl['positions-z'])[:, 0, 0]
    uniform = len(planes) > 1 and np.allclose(np.diff(nominal_z[planes]), np.diff(nominal_z[planes])[0])

    for name in src_pl:
        item = src_pl[name]
        if name == 'data':
            data = dst_pl.create_group('data')
            for attr, value in item.attrs.items():
                data.attrs[attr] = value
            for key in item:
                z = int(key) // per_plane
                if z in new_of:
                    item.copy(item[key], data, name=str(new_of[z] * per_plane + int(key) % per_plane))
        elif isinstance(item, h5py.Group):
            src_pl.copy(name, dst_pl)
        elif name in GRID_DATASETS:
            array = np.array(item)
            if array.shape[0] != nz:
                raise ValueError(f'{name}: {array.shape} does not start with the {nz} planes')
            dst_pl.create_dataset(name, data=_subset_axis0(array, planes)).attrs.update(item.attrs)
        elif name in POINT_DATASETS:
            array = np.ravel(item[...])
            if array.size != keep.size:
                raise ValueError(f'{name}: {array.size} values for {keep.size} points')
            dst_pl.create_dataset(name, data=array[keep]).attrs.update(item.attrs)
        elif name.startswith('positions-resume-plan-'):
            array = np.ravel(item[...])
            if array.size != keep.size + 1:
                raise ValueError(f'{name}: {array.size} values for {keep.size} points')
            body = array[1:][keep]
            if name == 'positions-resume-plan-index-z':
                body = np.array([new_of[int(z)] for z in body], dtype=array.dtype)
            dst_pl.create_dataset(name, data=np.concatenate([[len(body)], body]).astype(array.dtype)).attrs.update(item.attrs)
        elif uniform and name == 'positions-grid-z-range-um-used':
            dst_pl.create_dataset(name, data=np.array([nominal_z[planes[0]], nominal_z[planes[-1]]], dtype=item.dtype))
        elif uniform and name in ('positions-resume-s.zMin', 'positions-resume-s.zMax', 'positions-resume-s.zSteps'):
            old = np.ravel(item[...])
            value = {'zMin': nominal_z[planes[0]], 'zMax': nominal_z[planes[-1]], 'zSteps': len(planes)}[name.rsplit('.', 1)[1]]
            dst_pl.create_dataset(name, data=np.array([old[0], value], dtype=item.dtype))
        else:
            src_pl.copy(name, dst_pl)

    if objective is not None and 'scaleCalibration' in dst_pl:
        _label_objective(dst_pl, objective)
    return int(keep.sum())


def _label_objective(payload, objective):
    """The overview objective's calibration (recorded with the payload) as the payload's scale calibration."""
    prefix = None
    for name in payload:
        match = re.match(r'positions-overview-objective-(\d+)-magnification', name)
        if match and f'{float(np.ravel(payload[name][...])[0]):g}x' == objective:
            prefix = f'positions-overview-objective-{match.group(1)}-'
            slot = int(match.group(1))
            magnification = float(np.ravel(payload[name][...])[0])
    cal = payload['scaleCalibration']
    if prefix is None:
        return
    cal.attrs['objectiveName'] = np.array([objective.encode()], dtype='S%d' % len(objective))
    cal.attrs['objectiveSlot'] = np.array([slot], dtype=np.int32)
    cal.attrs['magnification'] = np.array([magnification])
    cal.attrs['hasFovOffset'] = np.array([0], dtype=np.int32)
    for target, source in (('micrometerToPixX', 'micrometer-to-pix-x'), ('micrometerToPixY', 'micrometer-to-pix-y'),
                           ('pixToMicrometerX', 'pix-to-micrometer-x'), ('pixToMicrometerY', 'pix-to-micrometer-y'),
                           ('origin', 'origin-pix')):
        if target in cal and prefix + source in payload:
            values = np.ravel(payload[prefix + source][...])
            cal[target].attrs['x'] = np.array([values[0]])
            cal[target].attrs['y'] = np.array([values[1]])


def split(source, output, planes_by_repetition, objective=None):
    with h5py.File(source, 'r') as src, h5py.File(output, 'w') as dst:
        for name, value in src.attrs.items():
            dst.attrs[name] = value
        for group in src:
            if group != 'Brillouin':
                src.copy(group, dst)
        brillouin = dst.create_group('Brillouin')
        for attr, value in src['Brillouin'].attrs.items():
            brillouin.attrs[attr] = value
        counts = {}
        for key, planes in planes_by_repetition.items():
            counts[key] = _split_repetition(src['Brillouin'][key], brillouin.create_group(key), planes, objective)
        dst.attrs['bmlab_split'] = json.dumps({
            'source': str(source), 'planes': {k: list(v) for k, v in planes_by_repetition.items()},
            'objective': objective, 'points': counts})
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    d = sub.add_parser('detect')
    d.add_argument('source')
    s = sub.add_parser('split')
    s.add_argument('source')
    s.add_argument('output')
    s.add_argument('--planes', action='append', required=True, help='REPETITION:PLANE,PLANE,...')
    s.add_argument('--objective')
    args = parser.parse_args(argv)
    if args.command == 'detect':
        with h5py.File(args.source, 'r') as f:
            for key in sorted(f['Brillouin'], key=int):
                payload = f['Brillouin'][key]['payload']
                found = detect(payload)
                cal = payload.get('scaleCalibration')
                name = cal.attrs['objectiveName'][0].decode() if cal is not None and 'objectiveName' in cal.attrs else '?'
                print(f'repetition {key}: scale calibration says {name};',
                      'planes (plane: objective after the switch back)', found)
        return 0
    planes = {}
    for spec in args.planes:
        key, values = spec.split(':')
        planes[key] = [int(v) for v in values.split(',')]
    print(split(args.source, args.output, planes, args.objective))
    return 0


if __name__ == '__main__':
    sys.exit(main())
