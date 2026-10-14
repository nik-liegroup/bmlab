"""
Second step for files migrated with migrate_legacy_overviews.py: where the image count did not match the planned overview
positions, the migration put all overview images of a repetition into one event (one z stack per tile), although they were
taken at different times (one overview after every z plane, or one z stack after every plane). This splits such an event at
every pause of more than GAP_S seconds between two images: each part becomes an event of its own ('overview_event' counted
on from the first, 'overview_z_index' = its number in the repetition), and within it the images at one position are a tile
and, in the order they were taken, its z stack (overview_tile_index/_count, overview_stack_index/_count). A single image
stays a single overview.

Only files carrying 'bmlab_migrated' are looked at, and in them only a repetition whose images all share one event and
contain such a pause; files written by the overview engine and correctly migrated events are left as they are.

Attributes only (nothing else is rewritten); a regrouped file is marked with 'bmlab_regrouped' and not touched again.

    python regroup_overviews_by_time.py FILE [FILE ...]            dry run: what would be done
    python regroup_overviews_by_time.py --apply FILE [FILE ...]    do it
"""
import argparse
import sys

import h5py
import numpy as np

from migrate_legacy_overviews import NEW, _date, _number, _text, group_by_tile, own_position, set_attr

GAP_S = 60.0


def plan(f):
    """[(image dataset, {attribute: value})] for every image of an event to be split, and a description per event."""
    changes, described = [], []
    for fluo_key in sorted(f['Fluorescence'], key=int):
        data = f['Fluorescence'][fluo_key]['payload']['data']
        by_repetition = {}
        for key in data:
            attrs = data[key].attrs
            if _text(attrs, 'channel') == NEW and _number(attrs, 'overview_event') is not None:
                by_repetition.setdefault(_number(attrs, 'brillouin_repetition_index'), []).append(
                    (_date(attrs), int(key), key))
        for repetition, members in by_repetition.items():
            keys = [key for _, _, key in sorted(members)]
            first_event = {int(_number(data[k].attrs, 'overview_event')) for k in keys}
            if len(first_event) > 1:
                continue                                         # already more than one event: correctly migrated
            times = [_date(data[k].attrs) for k in keys]
            pauses = [(b - a).total_seconds() > GAP_S for a, b in zip(times, times[1:])]
            if not any(pauses):
                continue
            parts = np.concatenate([[0], np.cumsum(pauses)])
            for part in range(parts[-1] + 1):
                part_keys = [k for k, p in zip(keys, parts) if p == part]
                positions = [own_position(data[k].attrs, 'position') or own_position(data[k].attrs, 'stage_position')
                             for k in part_keys]
                tiles = list(group_by_tile(positions).values()) if all(p is not None for p in positions) \
                    else [list(range(len(part_keys)))]
                described.append((repetition, len(part_keys), len(tiles)))
                for tile_index, members_of_tile in enumerate(tiles):
                    for stack_index, i in enumerate(members_of_tile):
                        changes.append((data[part_keys[i]], {
                            'overview_event': min(first_event) + part, 'overview_z_index': part,
                            'overview_tile_index': tile_index, 'overview_tile_count': len(tiles),
                            'overview_stack_index': stack_index, 'overview_stack_count': len(members_of_tile)}))
    return changes, described


def describe(described):
    counts = {}
    for _, images, tiles in described:
        label = f'{images // tiles} z x {tiles} tiles' if images % tiles == 0 else f'{images} images on {tiles} tiles'
        counts[label] = counts.get(label, 0) + 1
    return ', '.join(f'{n} x ({label})' for label, n in counts.items())


def regroup(path, apply):
    with h5py.File(path, 'r') as f:                          # opened for writing only if there is something to split
        if 'bmlab_migrated' not in f.attrs:
            return 'not migrated: left as it is', False
        if 'bmlab_regrouped' in f.attrs:
            return 'already regrouped', False
        if not plan(f)[0]:
            return 'nothing to split', False
    with h5py.File(path, 'r+' if apply else 'r') as f:
        changes, described = plan(f)
        if apply:
            for dataset, attrs in changes:
                for name, value in attrs.items():
                    set_attr(dataset, name, value)
            f.attrs['bmlab_regrouped'] = np.array([f'split at pauses > {GAP_S:g} s'.encode()])
        return f'{len(changes)} images -> {len(described)} events: {describe(described)}', apply


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='*')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    for path in args.files:
        message, written = regroup(path, args.apply)
        print(f"{'regrouped' if written else 'dry run'} {path}: {message}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
