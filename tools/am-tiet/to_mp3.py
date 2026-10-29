# WAV (ket-qua đã gộp) → giong-noi/am-tiet/{serena,aiden}/{base}{tone}.mp3, cân âm lượng theo RMS, vuốt đầu/cuối 8 ms.
import os, sys, json, subprocess, numpy as np, soundfile as sf
SRC, DST = sys.argv[1], sys.argv[2]
T = json.load(open('syl_table.json'))
TARGET = 10 ** (-20 / 20); PEAK = 10 ** (-1 / 20)
for spk in ('serena', 'aiden'):
    os.makedirs(f'{DST}/{spk}', exist_ok=True)
    for b, tones in T.items():
        for t in tones:
            w, sr = sf.read(f'{SRC}/{spk}_{b}{t}.wav', dtype='float32')
            e = np.sqrt(np.convolve(w ** 2, np.ones(240) / 240, 'same')); voiced = w[e > e.max() * 0.1]
            g = TARGET / max(np.sqrt(np.mean(voiced ** 2)), 1e-6); g = min(g, PEAK / max(np.abs(w).max(), 1e-6))
            w = w * g; f = int(sr * 0.008); r = np.linspace(0, 1, f, dtype=np.float32); w[:f] *= r; w[-f:] *= r[::-1]
            p = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-f', 'f32le', '-ar', str(sr), '-ac', '1', '-i', '-', '-c:a', 'libmp3lame', '-q:a', '5', f'{DST}/{spk}/{b}{t}.mp3'], input=w.tobytes())
            assert p.returncode == 0
json.dump({'voices': {'female': 'serena', 'male': 'aiden'}, 'tones': {b: ''.join(v) for b, v in T.items()}}, open(f'{DST}/index.json', 'w'), separators=(',', ':'))
print('ok')
