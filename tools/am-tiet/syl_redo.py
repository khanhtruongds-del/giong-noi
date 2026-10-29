# Chạy lại các âm chưa đạt trong ket-qua-full/scores.json (cùng cách đọc và chấm điểm như syl_full.py, nhiều lần thử hơn).
# Chỉ giữ bản mới khi điểm cao hơn bản cũ. Chạy lại được: âm đã làm sẽ bỏ qua.
# Đầu ra: ket-qua-redo/{voice}_{base}{tone}.wav + scores_redo.json (chỉ các âm có bản tốt hơn mới có file wav).
import os, json, time, numpy as np, torch, soundfile as sf
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'syl_full.py')).read()
exec(src[:src.index('m = Qwen3TTSModel')])  # dùng lại split / contour / score / INS / TABLE
from qwen_tts import Qwen3TTSModel
R = os.path.join(D, 'ket-qua-redo'); os.makedirs(R, exist_ok=True)
OLD = json.load(open(os.path.join(O, 'scores.json')))
RS = os.path.join(R, 'scores_redo.json'); res = json.load(open(RS)) if os.path.exists(RS) else {}
BATCH, ROUNDS = int(os.environ.get('BATCH', 8)), int(os.environ.get('ROUNDS', 2))  # bf16 + 8 lượt một lần: nhanh ~6 lần so với fp32 từng lượt trên CPU

def old_score(r):
    if r['passes'] < 0: return -1e9
    f = r['feat']; lows = [g['lo'] for g in f]; cont = 0
    for t, g in zip(r['tones'], f):
        rise = g['end'] - g['start']
        cont += -abs(rise) if t == '1' else rise if t == '2' else -(g['lo'] - min(lows)) if t == '3' else -rise
    return r['passes'] * 10 + cont

todo = []
for spk in ('serena', 'aiden'):
    for b in TABLE:
        r = OLD[f'{spk}_{b}']
        if r['passes'] != len(r['tones']) and f'{spk}_{b}' not in res:
            todo.append((r['passes'] >= 0, r['passes'] / len(r['tones']), spk, b))
todo.sort()
m = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', device_map=dev, dtype=torch.bfloat16, attn_implementation='sdpa')
t0 = time.time()
for n, (had, _, spk, base) in enumerate(todo):
    old = OLD[f'{spk}_{base}']; tones = list(TABLE[base]); chars = [TABLE[base][t] for t in tones]
    rep = 2 if len(chars) <= 2 else 1
    best = None; text = '，'.join(chars * rep) + '。'
    for rnd in range(ROUNDS):
        ws, sr = m.generate_custom_voice(text=[text] * BATCH, language=['Chinese'] * BATCH, speaker=[spk] * BATCH, instruct=[INS] * BATCH, max_new_tokens=40 * len(chars) * rep + 40)
        for w in ws:
            segs = split(np.asarray(w, dtype=np.float32), sr)
            if len(segs) != len(chars) * rep or not all(0.25 <= len(s) / sr <= 1.4 for s in segs): continue
            segs = segs[:len(chars)]
            sc, passes, f = score(segs, tones, sr)
            if not best or sc > best[0]: best = (sc, passes, segs, f)
        if best and best[1] == len(tones): break
    better = bool(best) and best[0] > old_score(old)
    if better:
        for t, s in zip(tones, best[2]): sf.write(f'{R}/{spk}_{base}{t}.wav', s, sr)
    res[f'{spk}_{base}'] = {'tones': tones, 'old': old['passes'], 'passes': best[1] if best else -1, 'better': better, 'feat': best[3] if best else []}
    json.dump(res, open(RS, 'w'))
    el = time.time() - t0; left = el / (n + 1) * (len(todo) - n - 1)
    print(f'[{n + 1}/{len(todo)}] {spk} {base} {old["passes"]} -> {res[spk + "_" + base]["passes"]}/{len(tones)} {"GIỮ" if better else "bỏ"} · còn khoảng {left / 3600:.1f} giờ', flush=True)
print('SYL_REDO DONE', flush=True)
