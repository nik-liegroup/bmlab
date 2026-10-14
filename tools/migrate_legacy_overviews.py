"""
One-off migration of Brillouin files written by older BrillouinAcquisition versions to the layout of the overview engine,
so that bmlab needs only one reader for brightfield overviews.

What it changes (attributes only - no spectrum, image or dataset is rewritten, the file does not grow noticeably):
  - every image of the channel 'Brightfield z overview' becomes 'Brightfield overview' and gets the attributes the overview
    engine writes: overview_event, overview_kind, overview_kind_name, overview_z_index, overview_tile_index/_count,
    overview_stack_index/_count, brillouin_repetition_index (the repetition the image was taken during, found as
    bmlab.export.naming.running_brillouin_repetition() did), and position_x/y/z_um where the image has none (taken from the
    planned overview positions the Brillouin side recorded, as the old overview export did);
  - an attribute 'bmlab_migrated' on the file.
The grouping of images into overview events, tiles and z stacks is the one the old OverviewBrightfieldExport used.

    python migrate_legacy_overviews.py FILE [FILE ...]            dry run: what would be done
    python migrate_legacy_overviews.py --apply FILE [FILE ...]    do it (one file after the other)
"""
import argparse
import datetime
import sys

import h5py
import numpy as np

LEGACY = 'Brightfield z overview'
NEW = 'Brightfield overview'
TOLERANCE_UM = 1.0


def _text(attrs, name):
    value = attrs.get(name)
    if value is None:
        return None
    value = np.ravel(value)[0]
    return value.decode('utf-8') if hasattr(value, 'decode') else str(value)


def _date(attrs):
    text = _text(attrs, 'date')
    return datetime.datetime.fromisoformat(text) if text else None


def _number(attrs, name):
    value = attrs.get(name)
    return None if value is None else float(np.ravel(value)[0])


def running_repetition(reps, when, recorded):
    """bmlab.export.naming.running_brillouin_repetition(): the repetition that was running at `when`."""
    if when:
        running = [(started, key) for key, started in reps.items() if started <= when]
        if running:
            return max(running)[1]
        later = [(started, key) for key, started in reps.items() if started > when]
        if later:
            return min(later)[1]
    return None if recorded is None else str(recorded)


def planned_positions(payload):
    """Payload.get_overview_brightfield_positions() of the Brillouin repetition's payload, or None."""
    if payload.get('positions-overview-brightfield-x') is None:
        return None
    count = payload.get('positions-overview-brightfield-point-count')
    if count is not None:
        point_count = int(np.ravel(count[...])[0])
        stack = payload.get('positions-overview-brightfield-point-stack-counts')
        stack_counts = np.ravel(stack[...]).astype(int) if stack is not None else np.ones(point_count, dtype=int)
    else:
        tiles = payload.get('positions-overview-brightfield-tile-count')
        point_count = int(np.ravel(tiles[...])[0]) if tiles is not None else 1
        stack_counts = np.ones(point_count, dtype=int)
    total = int(stack_counts.sum()) if point_count else 1
    out = {'tile_count': total}
    for axis in 'xyz':
        arr = np.array(payload['positions-overview-brightfield-' + axis])
        if total > 1 and arr.size % total == 0:
            arr = arr.reshape((-1, total))
        out[axis] = arr
    return out


def group_by_tile(positions):
    tiles = {}
    for index, pos in enumerate(positions):
        match = next((xy for xy in tiles if abs(xy[0] - pos['x']) < TOLERANCE_UM and abs(xy[1] - pos['y']) < TOLERANCE_UM), None)
        tiles.setdefault(match if match is not None else (pos['x'], pos['y']), []).append(index)
    return tiles


def own_position(attrs, prefix):
    values = [_number(attrs, f'{prefix}_{axis}_um') for axis in 'xyz']
    return None if any(v is None for v in values) else dict(zip('xyz', values))


def plan(f):
    """[(image dataset, {attribute: value})] for every legacy overview image of the file."""
    reps = {}
    for key in f['Brillouin']:
        started = _date(f['Brillouin'][key].attrs)
        if started is not None:
            reps[key] = started
    events = {}                                    # per Brillouin repetition: number of events so far
    changes = []
    for fluo_key in sorted(f['Fluorescence'], key=int):
        data = f['Fluorescence'][fluo_key]['payload']['data']
        members = []
        for key in data:
            if _text(data[key].attrs, 'channel') == LEGACY:
                members.append((_date(data[key].attrs), int(key), key))
        if not members:
            continue
        keys = [key for _, _, key in sorted(members)]            # by time, as image_keys_by_channel(sort_by_time=True)
        first = data[keys[0]].attrs
        recorded = _number(first, 'brillouin_repetition_index')
        bkey = running_repetition(reps, _date(first), None if recorded is None else int(recorded))
        planned = planned_positions(f['Brillouin'][bkey]['payload']) if bkey in f['Brillouin'] else None
        total = planned['tile_count'] if planned else 1
        steps = len(keys) // total if total else 0
        groups = []                                              # (keys, positions, z_index)
        if total <= 1 or steps < 1 or steps * total != len(keys):
            positions = [own_position(data[k].attrs, 'stage_position') or own_position(data[k].attrs, 'position') for k in keys]
            if planned is not None and total <= 1 and any(p is None for p in positions) and np.size(planned['x']) >= len(keys):
                flat = list(zip(*(np.ravel(planned[a]) for a in 'xyz')))
                positions = [p or dict(zip('xyz', xyz)) for p, xyz in zip(positions, flat)]
            groups.append((keys, positions, 0))
        else:
            for z in range(steps):
                at_z = keys[z * total:(z + 1) * total]
                groups.append((at_z, [{'x': planned['x'][z, p], 'y': planned['y'][z, p], 'z': planned['z'][z, p], '_planned': True}
                                      for p in range(total)], z))
        for group_keys, positions, z_index in groups:
            event = events.get(bkey, 0)
            events[bkey] = event + 1
            if all(p is not None for p in positions):
                tiles = list(group_by_tile(positions).values())
            else:
                tiles = [list(range(len(group_keys)))]
            for tile_index, members_of_tile in enumerate(tiles):
                for stack_index, i in enumerate(members_of_tile):
                    changes.append((data[group_keys[i]], group_keys[i], {
                        'channel': NEW, 'overview_event': event, 'overview_kind': 1,
                        'overview_kind_name': 'after z plane', 'overview_z_index': z_index, 'overview_point_index': -1,
                        'overview_objective_slot': -1, 'overview_is_measurement_objective': 1,
                        'overview_tile_index': tile_index, 'overview_tile_count': len(tiles),
                        'overview_stack_index': stack_index, 'overview_stack_count': len(members_of_tile),
                        'brillouin_repetition_index': None if bkey is None or not str(bkey).isdigit() else int(bkey),
                        '_position': positions[i]}))
    return changes


def set_attr(dataset, name, value):
    if isinstance(value, str):
        dataset.attrs[name] = np.array([value.encode('utf-8')])
    else:
        dataset.attrs[name] = np.array([float(value)]) if name != 'brillouin_repetition_index' else np.array([int(value)], dtype=np.int32)


def migrate(path, apply):
    mode = 'r+' if apply else 'r'
    with h5py.File(path, mode) as f:
        if 'bmlab_migrated' in f.attrs:
            return 'already migrated', False
        changes = plan(f)
        if not changes:
            return 'no legacy overview images', False
        if apply:
            for dataset, key, attrs in changes:
                position = attrs.pop('_position')
                for name, value in attrs.items():
                    if value is not None:
                        set_attr(dataset, name, value)
                if position is not None and own_position(dataset.attrs, 'position') is None:
                    for axis in 'xyz':
                        set_attr(dataset, f'position_{axis}_um', position[axis])
                elif position is not None and position.get('_planned'):
                    # The old exporter placed the images of a tiled event at the planned positions of the Brillouin side, not at the
                    # positions stored with the images. Where the two disagree by more than the tolerance the stored ones are stale
                    # (e.g. the overview before the run, taken before the grid was moved: its tiles match the in-run overview
                    # only at the planned positions, 0.90-0.99 vs. no match at the stored ones): the planned positions are written
                    # over both.
                    for prefix in ('position', 'stage_position'):
                        own = own_position(dataset.attrs, prefix)
                        if own is not None and (abs(own['x'] - position['x']) > TOLERANCE_UM or abs(own['y'] - position['y']) > TOLERANCE_UM):
                            for axis in 'xyz':
                                set_attr(dataset, f'{prefix}_{axis}_um', position[axis])
            f.attrs['bmlab_migrated'] = np.array([datetime.datetime.now().isoformat().encode()])
        events = {(a['overview_event'], a['brillouin_repetition_index']) for _, _, a in changes}
        return f'{len(changes)} images in {len(events)} events', apply


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='*')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    for path in args.files:
        message, written = migrate(path, args.apply)
        print(f"{'migrated' if written else 'dry run'} {path}: {message}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
