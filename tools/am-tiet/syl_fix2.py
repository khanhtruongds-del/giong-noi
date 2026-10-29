# Thử lại aiden nan / wen: (a) đổi chữ Hán, (b) đọc từng chữ lặp đôi "X，X。" rồi lấy đoạn đầu. Ghi ket-qua-fix/.
import os, json, numpy as np, torch, soundfile as sf
src = open('syl_full.py').read(); exec(src[:src.index('m = Qwen3TTSModel')])
from qwen_tts import Qwen3TTSModel
F = os.path.join(D, 'ket-qua-fix'); os.makedirs(F, exist_ok=True)
m = Qwen3TTSModel.from_pretrained('Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice', device_map='cpu', dtype=torch.bfloat16, attn_implementation='sdpa')
B = 8
def gen(text, n_seg, rounds=2):
    out = []
    for _ in range(rounds):
        ws, sr = m.generate_custom_voice(text=[text] * B, language=['Chinese'] * B, speaker=['aiden'] * B, instruct=[INS] * B, max_new_tokens=40 * n_seg + 40)
        for w in ws:
            segs = split(np.asarray(w, dtype=np.float32), sr)
            if len(segs) == n_seg and all(0.25 <= len(s) / sr <= 1.4 for s in segs): out.append(segs)
    return out, sr
res = {}
# (a) cả chuỗi, đổi chữ
for base, chars in [('nan', {'2': '南', '3': '腩'}), ('nan', {'2': '难', '3': '腩'}), ('wen', {'1': '温', '2': '文', '3': '稳', '4': '问'})]:
    tones = list(chars); cs = [chars[t] for t in tones]; rep = 2 if len(cs) <= 2 else 1
    takes, sr = gen('，'.join(cs * rep) + '。', len(cs) * rep)
    best = max((score(s[:len(cs)], tones, sr) + (s[:len(cs)],) for s in takes), key=lambda x: x[0], default=None)
    print('chuoi', base, ''.join(cs), 'takes ok', len(takes), 'best', best[1] if best else None, flush=True)
    if best and (base not in res or best[0] > res[base][0]): res[base] = (best[0], best[1], dict(zip(tones, best[3])), ''.join(cs), 'chuoi')
# (b) từng chữ, lặp đôi
for base, chars in [('nan', {'2': '男', '3': '腩'}), ('wen', {'1': '温', '2': '文', '3': '稳', '4': '问'})]:
    got = {}
    for t, c in chars.items():
        takes, sr = gen(f'{c}，{c}。', 2)
        cand = [(score([s[0]], [t], sr), s[0]) for s in takes]
        # chấm 1 âm: dùng điểm thanh đơn (thanh 3 luôn qua khi 1 đoạn nên ưu tiên đoạn có lo thấp so với đầu)
        cand = [(sc[0] + (-(sc[2][0]['lo'] - sc[2][0]['start']) * -1 if t == '3' and sc[2] else 0), sc[1], seg) for sc, seg in cand if sc[2]]
        if cand: got[t] = max(cand, key=lambda x: x[0])
        print('tung chu', base, t, c, 'takes ok', len(takes), 'pass', got[t][1] if t in got else None, flush=True)
    if len(got) == len(chars):
        passes = sum(g[1] for g in got.values())
        if base not in res or passes > res[base][1]: res[base] = (None, passes, {t: g[2] for t, g in got.items()}, ''.join(chars.values()), 'tung chu')
for base, (_, passes, segs, cs, how) in res.items():
    for t, s in segs.items(): sf.write(f'{F}/aiden_{base}{t}.wav', s, sr)
    print('KET QUA', base, cs, how, 'passes', passes, '/', len(segs), flush=True)
print('FIX DONE', flush=True)
