# Gộp ket-qua-redo (cạnh file này) vào qwen-am-tiet/ket-qua-full: chép đè wav của các âm có bản tốt hơn, cập nhật scores.json.
# Bản sao scores.json cũ: ket-qua-full/scores.before-redo.json. Bản wav cũ bị thay được cất vào ket-qua-full/_cu/.
import os, sys, json, shutil
H = os.path.dirname(os.path.abspath(__file__)); F = sys.argv[1]; R = os.path.join(H, 'ket-qua-redo')
S = json.load(open(os.path.join(F, 'scores.json'))); RS = json.load(open(os.path.join(R, 'scores_redo.json')))
bk = os.path.join(F, 'scores.before-redo.json')
if not os.path.exists(bk): shutil.copy(os.path.join(F, 'scores.json'), bk)
old = os.path.join(F, '_cu'); os.makedirs(old, exist_ok=True); n = 0
for k, r in RS.items():
    if not r.get('better'): continue
    for t in r['tones']:
        dst = os.path.join(F, f'{k}{t}.wav')
        if os.path.exists(dst) and not os.path.exists(os.path.join(old, f'{k}{t}.wav')): shutil.copy(dst, old)
        shutil.copy(os.path.join(R, f'{k}{t}.wav'), dst)
    S[k] = {'tones': r['tones'], 'passes': r['passes'], 'feat': r['feat'], 'redo': True}; n += 1
json.dump(S, open(os.path.join(F, 'scores.json'), 'w'))
print(f'Đã gộp {n} âm vào {F}')
