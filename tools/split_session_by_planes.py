"""
Cuts the saved evaluation (session) of a Brillouin file to the planes of a file written by split_by_planes.py, so the
split file comes with the evaluation of its points and nothing has to be fitted again.

    python split_session_by_planes.py SESSION.h5 SPLIT_RAW.h5 SPLIT_SESSION.h5

SESSION.h5 is the session of the source file (it is only read), SPLIT_RAW.h5 the file written by split_by_planes.py (which
records which planes of which repetitions it holds), SPLIT_SESSION.h5 the new session. Per repetition kept: the results
(an array over x, y, z, ...) are cut to the planes, and the fitted spectra and frequencies of the points are renamed to the
keys the points have in the split file. Repetitions that are not in the split file are removed. Everything else (the
calibration, peak selection, thresholds, ...) is copied as it is. The batch table's cache attribute is dropped; BMicro
rebuilds it the first time it looks at the file.
"""
import json
import shutil
import sys

import h5py
import numpy as np

MODEL_GROUPS = ('background_models', 'calibration_models', 'evaluation_models', 'extraction_models',
                'peak_selection_models')


def _cut_results(group, planes, axis=2):
    for name in list(group):
        old = group[name]
        if not isinstance(old, h5py.Dataset):
            continue
        data = np.take(old[...], planes, axis=axis)
        attrs = dict(old.attrs)
        options = {'compression': old.compression, 'compression_opts': old.compression_opts, 'shuffle': old.shuffle}
        del group[name]
        new = group.create_dataset(name, data=data, **options)
        for key, value in attrs.items():
            new.attrs[key] = value


def _renumber(group, keep_of_key):
    """Renames the member groups named by point keys: {old key: new key}; the others are deleted."""
    for key in [k for k in group if k not in keep_of_key]:
        del group[key]
    for old in sorted(keep_of_key, key=int):
        new = keep_of_key[old]
        if new != old:
            group.move(old, new)


def split_session(session, split_raw, output):
    with h5py.File(split_raw, 'r') as raw:
        info = json.loads(raw.attrs['bmlab_split'])
        per_plane = None
        for key in info['planes']:
            payload = raw['Brillouin'][key]['payload']
            per_plane = int(payload.attrs['resolution-x'][0]) * int(payload.attrs['resolution-y'][0])
    planes_by_rep = {k: list(v) for k, v in info['planes'].items()}
    shutil.copyfile(session, output)
    with h5py.File(output, 'r+') as f:
        s = f['session']
        for model in MODEL_GROUPS:
            if model in s:
                for rep in [r for r in s[model] if r not in planes_by_rep]:
                    del s[model][rep]
        for rep, planes in planes_by_rep.items():
            model = s['evaluation_models'][rep]
            _cut_results(model['results'], planes)
            new_of_plane = {z: i for i, z in enumerate(planes)}
            keys = {}
            for key in model['frequencies']:
                z, rest = divmod(int(key), per_plane)
                if z in new_of_plane:
                    keys[key] = str(new_of_plane[z] * per_plane + rest)
            _renumber(model['frequencies'], keys)
            _renumber(model['spectra'], keys)
        if '_current_repetition_key' in s:
            current = s['_current_repetition_key'][()]
            current = current.decode() if hasattr(current, 'decode') else str(current)
            if current not in planes_by_rep:
                del s['_current_repetition_key']
                s.create_dataset('_current_repetition_key', data=sorted(planes_by_rep, key=int)[0], dtype=h5py.string_dtype())
        if 'bmicro_batch' in f.attrs:
            del f.attrs['bmicro_batch']
    return output


def main(argv=None):
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    print(split_session(*sys.argv[1:4]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
