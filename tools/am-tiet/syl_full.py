# Full pinyin chart: every base syllable read as one series of its existing tones ("妈，麻，马，骂。"), cut at the pauses,
# scored by pitch contour, best take kept. Bases with 1-2 tones are read twice in a row so the model is never given a
# lone character. Resumable: finished bases are skipped. Output: ket-qua-full/{voice}_{base}{tone}.wav + scores.json
import os, sys, json, time, numpy as np, torch, soundfile as sf, librosa
from qwen_tts import Qwen3TTSModel
D = os.path.dirname(os.path.abspath(__file__)); O = os.path.join(D, 'ket-qua-full'); os.makedirs(O, exist_ok=True)
TABLE = json.load(open(os.path.join(D, 'syl_table.json')))
INS = '这是汉语拼音声调示范：一字一顿，每个字之间停顿一下，发音饱满清晰，声调准确，字正腔圆，语速适中。'
dev = 'mps' if torch.backends.mps.is_available() else 'cpu'

def split(w, sr):
    win = int(sr * 0.01); n = len(w) // win
    e = np.array([np.sqrt(np.mean(w[i * win:(i + 1) * win] ** 2)) for i in range(n)])
    loud = e > e.max() * 0.05
    segs, i = [], 0
    while i < n:
        if loud[i]:
            j = i
            while j < n and (loud[j] or (j + 8 < n and loud[j:j + 8].any())): j += 1
            segs.append([i, j]); i = j
        else: i += 1
    segs = [s for s in segs if s[1] - s[0] >= 12]
    return [w[max(0, a * win - int(0.05 * sr)):min(len(w), b * win + int(0.08 * sr))] for a, b in segs]

def contour(seg, sr):
    y = librosa.resample(seg.astype(np.float32), orig_sr=sr, target_sr=16000)
    f0, _, _ = librosa.pyin(y, fmin=65, fmax=500, sr=16000, frame_length=1024, hop_length=160)
    f0 = f0[~np.isnan(f0)]
    return 12 * np.log2(f0) if len(f0) >= 5 else None

def score(segs, tones, sr):
    c = [contour(s, sr) for s in segs]
    if any(x is None for x in c): return -99, 0, []
    ref = np.median(np.concatenate(c)); f = []
    for x in c:
        k = max(1, len(x) // 5)
        f.append(dict(start=float(np.mean(x[:k]) - ref), end=float(np.mean(x[-k:]) - ref), lo=float(np.min(x) - ref), mean=float(np.mean(x) - ref)))
    lows = [g['lo'] for g in f]; ok = []; cont = 0
    for t, g in zip(tones, f):
        rise = g['end'] - g['start']
        if t == '1': ok.append(abs(rise) < 2.5); cont -= abs(rise)
        elif t == '2': ok.append(rise > 1.5); cont += rise
        elif t == '3': ok.append(len(f) == 1 or g['lo'] <= min(lows) + 0.5); cont -= g['lo'] - min(lows)
        else: ok.append(-rise > 2.5); cont -= rise
    return sum(ok) * 10 + cont, sum(ok), f

m = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', device_map=dev, dtype=torch.float32, attn_implementation='sdpa')
SC = os.path.join(O, 'scores.json'); res = json.load(open(SC)) if os.path.exists(SC) else {}
todo = [(spk, b) for spk in ('serena', 'aiden') for b in TABLE if f'{spk}_{b}' not in res]
t0 = time.time()
for n, (spk, base) in enumerate(todo):
    tones = list(TABLE[base]); chars = [TABLE[base][t] for t in tones]
    rep = 2 if len(chars) <= 2 else 1
    best = None
    for take in range(8):
        w, sr = m.generate_custom_voice(text='，'.join(chars * rep) + '。', language='Chinese', speaker=spk, instruct=INS, max_new_tokens=40 * len(chars) * rep + 40)
        segs = split(np.asarray(w[0]), sr)
        if len(segs) != len(chars) * rep or not all(0.25 <= len(s) / sr <= 1.4 for s in segs): continue
        segs = segs[:len(chars)]
        sc, passes, f = score(segs, tones, sr)
        if not best or sc > best[0]: best = (sc, passes, segs, f)
        if passes == len(tones): break
    if best:
        for t, s in zip(tones, best[2]): sf.write(f'{O}/{spk}_{base}{t}.wav', s, sr)
        res[f'{spk}_{base}'] = {'tones': tones, 'passes': best[1], 'feat': best[3]}
    else: res[f'{spk}_{base}'] = {'tones': tones, 'passes': -1}
    json.dump(res, open(SC, 'w'))
    el = time.time() - t0; left = el / (n + 1) * (len(todo) - n - 1)
    print(f'[{n + 1}/{len(todo)}] {spk} {base} {res[spk + "_" + base]["passes"]}/{len(tones)} · còn khoảng {left / 3600:.1f} giờ', flush=True)
print('SYL_FULL DONE', flush=True)
