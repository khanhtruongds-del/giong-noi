"""Lay out model files for the web app: split big ONNX files into parts that static hosts accept.

    python tools/build_models.py --supertonic <supertonic-3 dir> --kokoro <kokoro-82m dir> --out models [--part-mb 45]

<supertonic-3 dir>: onnx/*.onnx, onnx/tts.json, onnx/unicode_indexer.json, voice_styles/*.json (fp16 weights fine)
<kokoro-82m dir>:   kokoro.onnx, config.json, voices/*.npy (from training/export_kokoro.py)
Writes models/manifest.json: {engines: {name: {bytes, files: {name: {size, sha256, parts: [...]}}}}}.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ST_FILES = ['duration_predictor.onnx', 'text_encoder.onnx', 'vector_estimator.onnx', 'vocoder.onnx', 'tts.json',
            'unicode_indexer.json']


def split(src, dst_dir, name, part_bytes):
    data = Path(src).read_bytes()
    parts = []
    for i in range(0, len(data), part_bytes):
        pname = f'{name}.part{i // part_bytes:02d}'
        (dst_dir / pname).write_bytes(data[i:i + part_bytes])
        parts.append(pname)
    return dict(size=len(data), sha256=hashlib.sha256(data).hexdigest(), parts=parts)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--supertonic', type=Path, required=True)
    p.add_argument('--kokoro', type=Path, required=True)
    p.add_argument('--kokoro-voices', default='zf_,zm_', help='tiền tố giọng Kokoro cần đưa lên (mặc định: tiếng Trung)')
    p.add_argument('--out', type=Path, default=Path('models'))
    p.add_argument('--part-mb', type=int, default=45)
    a = p.parse_args()
    part = a.part_mb * 1000 * 1000
    if a.out.exists():
        shutil.rmtree(a.out)
    manifest = {'engines': {}}
    st = a.out / 'supertonic-3'
    (st / 'voice_styles').mkdir(parents=True)
    files = {f: split(a.supertonic / 'onnx' / f, st, f, part) for f in ST_FILES}
    for v in sorted((a.supertonic / 'voice_styles').glob('*.json')):
        shutil.copy2(v, st / 'voice_styles' / v.name)
    manifest['engines']['supertonic-3'] = dict(bytes=sum(f['size'] for f in files.values()), files=files,
                                               voices=sorted(v.stem for v in (a.supertonic / 'voice_styles').glob('*.json')))
    kk = a.out / 'kokoro-82m'
    (kk / 'voices').mkdir(parents=True)
    files = {f: split(a.kokoro / f, kk, f, part) for f in ('kokoro.onnx', 'config.json')}
    voices = []
    prefixes = tuple(a.kokoro_voices.split(','))
    for v in sorted((a.kokoro / 'voices').glob('*.npy')):
        if v.stem.startswith(prefixes):
            arr = np.load(v).astype('<f4').reshape(-1, 256)
            assert arr.shape == (510, 256), arr.shape
            (kk / 'voices' / f'{v.stem}.bin').write_bytes(arr.tobytes())
            voices.append(v.stem)
    manifest['engines']['kokoro-82m'] = dict(bytes=sum(f['size'] for f in files.values()), files=files, voices=voices)
    (a.out / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n', encoding='utf-8')
    for name, e in manifest['engines'].items():
        print(name, round(e['bytes'] / 1e6, 1), 'MB', sum(len(f['parts']) for f in e['files'].values()), 'parts', len(e['voices']), 'voices')


if __name__ == '__main__':
    main()
