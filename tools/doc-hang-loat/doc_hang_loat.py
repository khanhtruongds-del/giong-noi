# -*- coding: utf-8 -*-
"""Đọc hàng loạt (jobs.json → mp3 + ket-qua.json) bằng ĐÚNG các file model của bản web giong-noi.

Chạy lại y hệt qwen.js + qwen-worker.js (Qwen3-TTS-12Hz-0.6B-Base, ONNX ở huggingface.co/Khanhpls/qwen3-tts-web,
giọng = vector người nói trong voices.json), nhưng bằng ONNX Runtime Python trên GPU (CUDA) hoặc CPU.
Định dạng trao đổi: docs/am-thanh-tts.md của PLS Classroom.

  python doc_hang_loat.py --jobs jobs.json --out ket-qua [--model thu-muc-model] [--gioi-han 20] [--het-gio 11.5]

Chạy lại cùng --out: việc đã "xong" (có trong ket-qua.json và còn file mp3) được bỏ qua, chỉ đọc việc lỗi / chưa làm.
Notebook Kaggle (kaggle/doc-hang-loat.ipynb) được tạo từ file này bằng tools/doc-hang-loat/tao-notebook.js.
"""
import argparse, glob, hashlib, json, os, re, sys, time, unicodedata
from datetime import datetime, timezone
import numpy as np
try:
    import chat_luong as CL                         # làm sạch chữ, tách vế, nhịp theo tiếng, hoàn thiện, chấm Whisper (13/11/2026)
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import chat_luong as CL

HF_BASE = 'https://huggingface.co/Khanhpls/qwen3-tts-web/resolve/main/'
MODEL_FILES = ['tokenizer.json', 'tokenizer_config.json', 'voices.json', 'text.f16', 'codec.f32',
               'talker_decode_q8nb.onnx', 'cp_fused2_f32.onnx', 'decoder_f32.onnx']
MODEL_NAME = 'Qwen3-TTS 0.6B (Khanhpls/qwen3-tts-web)'

# ---- hằng số của qwen.js -------------------------------------------------------------------------
D, KVH, HD, LAYERS, GROUPS, CP_VOCAB = 1024, 8, 128, 28, 16, 2048
CFG = dict(tts_bos=151672, tts_eos=151673, tts_pad=151671, codec_bos=2149, codec_eos=2150, codec_pad=2148,
           think=2154, nothink=2155, think_bos=2156, think_eos=2157, vocab=3072)
LANG_ID = {'chinese': 2055, 'english': 2050, 'german': 2053, 'spanish': 2054, 'japanese': 2058, 'korean': 2064,
           'french': 2061, 'russian': 2069, 'italian': 2070, 'portuguese': 2071}
LANG = {'de': 'german', 'en': 'english', 'fr': 'french', 'zh': 'chinese', 'ja': 'japanese', 'ko': 'korean'}
SAMPLE_RATE = 24000

# ---- config.json của giong-noi (giọng, quãng nghỉ) – bản sao để notebook chạy độc lập ----------------
VOICE_LABELS = {'nu1': 'Nữ 1', 'nu2': 'Nữ 2', 'nu3': 'Nữ 3', 'nu4': 'Nữ 4', 'nu5': 'Nữ 5', 'co': 'Cô', 'co_ja': 'Cô',
                'bac_gai': 'Bác gái', 'bac_gai_ja': 'Bác gái', 'ba': 'Bà', 'nam1': 'Nam 1', 'nam2': 'Nam 2', 'nam3': 'Nam 3',
                'nam4': 'Nam 4', 'nam5': 'Nam 5', 'chu': 'Chú', 'bac_trai': 'Bác trai', 'ong': 'Ông'}
PAUSES = {'sentence': 0.3, 'comma': 0.15, 'turn': 0.6}
PAUSES_BY_LANG = {'zh': {'sentence': 0.5, 'comma': 0.25}}


def voice_code(label, lang):
    """'Nữ 4' → 'nu4'. Tiếng Nhật: 'Cô' / 'Bác gái' → co_ja / bac_gai_ja (như config.json). Nhận cả mã sẵn ('nu4')."""
    if label in VOICE_LABELS:
        return label
    hits = [k for k, v in VOICE_LABELS.items() if v == label]
    if not hits:
        raise ValueError(f'Không biết giọng "{label}"')
    if lang == 'ja':
        hits.sort(key=lambda k: not k.endswith('_ja'))
    else:
        hits = [k for k in hits if not k.endswith('_ja')] or hits
    return hits[0]


# ---- text.js: sentenceChunks ---------------------------------------------------------------------
SENTENCE_END = '.!?。！？…\n'
SOFT_BREAK = ',，、;；:：'


def sentence_chunks(text, limit=200, group=True):
    chars = list(unicodedata.normalize('NFC', text))
    sentences, start = [], 0
    for i, c in enumerate(chars):
        if c in SENTENCE_END and (i + 1 == len(chars) or chars[i + 1] not in SENTENCE_END):
            sentences.append(chars[start:i + 1]); start = i + 1
    sentences.append(chars[start:])
    pieces = []
    for s in sentences:
        while len(s) > limit:
            head = s[:limit]
            cuts = [i + 1 for i, c in enumerate(head) if c in SOFT_BREAK]
            if not cuts:
                cuts = [i + 1 for i, c in enumerate(head) if c.isspace()]
            cut = cuts[-1] if cuts and cuts[-1] > limit // 3 else limit
            pieces.append(s[:cut]); s = s[cut:]
        pieces.append(s)
    chunks, cur = [], []
    for p in pieces:
        if cur and (not group or len(cur) + len(p) > limit):
            chunks.append(''.join(cur)); cur = []
        cur = cur + p
    if cur:
        chunks.append(''.join(cur))
    out = [' '.join(c.split()) for c in chunks]
    return [c for c in out if c.strip()]


RUBY = re.compile(r'\{([^{}|]+)\|[^{}]*\}')        # {漢字|かんじ} → 漢字 (ký hiệu furigana trong DATA APP)
def clean(text):
    return RUBY.sub(r'\1', text or '').strip()


# ---- model ---------------------------------------------------------------------------------------
def download_model(dest):
    import urllib.request
    os.makedirs(dest, exist_ok=True)
    for f in MODEL_FILES:
        p = os.path.join(dest, f)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            continue
        print('Tải', f, '…', flush=True)
        urllib.request.urlretrieve(HF_BASE + f, p + '.part')
        os.replace(p + '.part', p)
    return dest


class QwenONNX:
    """Bản Python của QwenTTS (qwen.js), chế độ giọng x-vector (icl = false) như qwen-worker.js."""

    def __init__(self, model_dir, device='auto'):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        self.ort = ort
        ort.set_default_logger_severity(3)                 # bỏ cảnh báo ScatterND của CUDA (vô hại)
        if hasattr(ort, 'preload_dlls'):
            try: ort.preload_dlls()
            except Exception: pass
        avail = ort.get_available_providers()
        gpu = device != 'cpu' and 'CUDAExecutionProvider' in avail
        prov = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if gpu else ['CPUExecutionProvider']
        so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.log_severity_level = 3
        P = lambda f: os.path.join(model_dir, f)
        self.talker = ort.InferenceSession(P('talker_decode_q8nb.onnx'), so, providers=prov)
        self.cpf = ort.InferenceSession(P('cp_fused2_f32.onnx'), so, providers=prov)
        self.dec = ort.InferenceSession(P('decoder_f32.onnx'), so, providers=prov)
        self.gpu = gpu and 'CUDAExecutionProvider' in self.talker.get_providers()
        self.device = 'cuda' if self.gpu else 'cpu'
        if gpu and not self.gpu:
            print('CẢNH BÁO: có CUDA nhưng không nạp được model lên GPU (thiếu bộ nhớ card? chương trình khác đang giữ GPU?) '
                  '→ đang chạy CPU, chậm hơn ~30 lần.', flush=True)
        self.tok = Tokenizer.from_file(P('tokenizer.json'))
        with open(P('voices.json'), encoding='utf-8') as f:
            self.voices = json.load(f)
        self.text_tab = np.fromfile(P('text.f16'), dtype=np.float16).reshape(-1, D)
        self.codec_tab = np.fromfile(P('codec.f32'), dtype=np.float32).reshape(-1, D)
        self.stats = {'talker': 0.0, 'tn': 0, 'cp': 0.0, 'cn': 0, 'dec': 0.0}

    def T(self, i): return self.text_tab[i].astype(np.float32)
    def C(self, i): return self.codec_tab[i].copy()

    def build_prompt(self, text, lang, persona):
        ids = self.tok.encode(f'<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n', add_special_tokens=False).ids
        rows = [self.T(i) for i in ids[:3]]
        pad, bos, eos = self.T(CFG['tts_pad']), self.T(CFG['tts_bos']), self.T(CFG['tts_eos'])
        lid = LANG_ID.get(lang.lower())
        tags = [CFG['nothink'], CFG['think_bos'], CFG['think_eos']] if lid is None else [CFG['think'], CFG['think_bos'], lid, CFG['think_eos']]
        codec = [self.C(t) for t in tags] + [np.asarray(persona['spk_emb'], np.float32), self.C(CFG['codec_pad']), self.C(CFG['codec_bos'])]
        for i in range(len(codec) - 1):
            rows.append((pad if i < len(codec) - 2 else bos) + codec[i])
        text_ids = ids[3:-5]
        rows.append(self.T(text_ids[0]) + codec[-1])
        trailing = [self.T(i) for i in text_ids[1:]] + [eos]
        return rows, trailing, pad

    @staticmethod
    def sample(logits, rng, history, min_done, temperature=0.9, top_k=50, penalty=1.05):
        l = logits.astype(np.float64).copy()
        if history and penalty != 1:
            h = np.fromiter(set(history), dtype=np.int64)
            l[h] = np.where(l[h] < 0, l[h] * penalty, l[h] / penalty)
        V = CFG['vocab']                                   # chỉ giữ mã âm thanh + codec_eos
        eos = l[CFG['codec_eos']]; l[V - 1024:] = -np.inf; l[CFG['codec_eos']] = eos
        if not min_done: l[CFG['codec_eos']] = -np.inf
        l /= temperature
        idx = np.argsort(-l, kind='stable')[:top_k]; idx = idx[np.isfinite(l[idx])]
        p = np.exp(l[idx] - l[idx[0]]); p /= p.sum()
        r = rng.random()
        c = np.cumsum(p)
        k = int(np.searchsorted(c, r, side='left'))
        return int(idx[min(k, len(idx) - 1)])

    def _empty_kv(self):
        return np.zeros((1, KVH, 0, HD), np.float32)

    def talker_run(self, rows, past, past_len):
        x = np.stack(rows)[None].astype(np.float32)
        mask = np.ones((1, past_len + len(rows)), np.int64)
        t0 = time.perf_counter()
        if self.gpu:                                       # KV cache ở lại trên GPU (như preferredOutputLocation của bản web)
            io = self.talker.io_binding()
            io.bind_cpu_input('inputs_embeds', x); io.bind_cpu_input('attention_mask', mask)
            for i in range(LAYERS):
                for kind, j in (('key', 2 * i), ('value', 2 * i + 1)):
                    if past is None: io.bind_cpu_input(f'past_{kind}_{i}', self._empty_kv())
                    else: io.bind_ortvalue_input(f'past_{kind}_{i}', past[j])
            io.bind_output('logits'); io.bind_output('last_hidden')
            for i in range(LAYERS):
                io.bind_output(f'present_key_{i}', 'cuda'); io.bind_output(f'present_value_{i}', 'cuda')
            self.talker.run_with_iobinding(io)
            outs = io.get_outputs()
            logits, hidden, kv = outs[0].numpy(), outs[1].numpy(), outs[2:]
        else:
            feeds = {'inputs_embeds': x, 'attention_mask': mask}
            for i in range(LAYERS):
                feeds[f'past_key_{i}'] = self._empty_kv() if past is None else past[2 * i]
                feeds[f'past_value_{i}'] = self._empty_kv() if past is None else past[2 * i + 1]
            names = ['logits', 'last_hidden'] + [f'present_{k}_{i}' for i in range(LAYERS) for k in ('key', 'value')]
            outs = self.talker.run(names, feeds)
            logits, hidden, kv = outs[0], outs[1], outs[2:]
        self.stats['talker'] += time.perf_counter() - t0; self.stats['tn'] += 1
        return logits.reshape(-1)[-CFG['vocab']:], hidden.reshape(-1)[-D:].astype(np.float32), kv

    def generate(self, text, lang, persona, max_frames=600, rng=None):
        rng = rng or np.random.default_rng()
        rows, trailing, pad = self.build_prompt(text, lang, persona)
        logits, hidden, past = self.talker_run(rows, None, 0)
        past_len = len(rows)
        frames, history = [], []
        tok = self.sample(logits, rng, history, False)
        inv_temp = np.array([1 / 0.9], np.float32)
        for step in range(max_frames):
            if tok == CFG['codec_eos']:
                break
            history.append(tok)
            last = self.C(tok)
            gum = -np.log(-np.log(np.maximum(rng.random(15 * CP_VOCAB), 1e-12))).astype(np.float32).reshape(15, CP_VOCAB)
            t0 = time.perf_counter()
            codes, sum_embed = self.cpf.run(['codes', 'sum_embed'], {'past_hidden': hidden.reshape(1, 1, D), 'first_embed': last.reshape(1, 1, D),
                                                                     'gumbel': gum, 'inv_temp': inv_temp})
            self.stats['cp'] += time.perf_counter() - t0; self.stats['cn'] += 1
            frames.append([tok] + [int(c) for c in codes.reshape(-1)])
            s = sum_embed.reshape(-1).astype(np.float32) + (trailing[step] if step < len(trailing) else pad)
            logits, hidden, past = self.talker_run([s], past, past_len); past_len += 1
            tok = self.sample(logits, rng, history, len(history) >= 2)
        return frames

    def decode_window(self, frames):
        codes = np.asarray(frames, np.int64)[None]       # [1, T, 16]
        out = self.dec.run(None, {'audio_codes': codes})
        names = [o.name for o in self.dec.get_outputs()]
        o = dict(zip(names, out))
        wav = o['audio_values'].reshape(-1)
        if 'lengths' in o:
            wav = wav[:int(np.asarray(o['lengths']).reshape(-1)[0])]
        return wav

    def decode(self, frames, win=100, ctx=25):
        t0 = time.perf_counter()
        T = len(frames)
        if T <= win:
            wav = self.decode_window(frames).astype(np.float32)
        else:
            parts, step = [], win - ctx
            for s in range(0, T, step):
                a, b = max(0, s - ctx), min(T, s + step)
                w = self.decode_window(frames[a:b])
                parts.append(w[round(len(w) * (s - a) / (b - a)):])
            wav = np.concatenate(parts).astype(np.float32)
        self.stats['dec'] += time.perf_counter() - t0
        return np.nan_to_num(wav)


# ---- qwen-worker.js: speak / synthesize ----------------------------------------------------------
CJK = re.compile('[぀-ヿ㐀-鿿가-힯]')
def max_frames(t):
    n = sum(0.6 if CJK.match(c) else 0.05 if c.isspace() else 0.17 for c in t)
    return min(900, 40 + int(np.ceil(n * 12.5 * 1.6)))


def silent(x):
    return (not np.all(np.isfinite(x))) or np.max(np.abs(x), initial=0) < 1e-3


def trim_silence(x, sr=SAMPLE_RATE, margin=0.05):
    peak = np.max(np.abs(x), initial=0)
    if not peak: return x
    th, win = peak * 0.02, round(sr * 0.01)
    nwin = int(np.ceil(len(x) / win))
    loud = np.array([np.any(np.abs(x[i * win:(i + 1) * win]) > th) for i in range(nwin)])
    if not loud.any(): return x[:0]
    a = int(np.argmax(loud)) * win
    b = min(len(x), (nwin - int(np.argmax(loud[::-1]))) * win)
    m = round(sr * margin)
    return x[max(0, a - m):min(len(x), b + m)]


def speak(q, text, lang, voice, rng):
    persona = q.voices.get(voice)
    if persona is None: raise ValueError('Không tìm thấy giọng ' + voice)
    lim = max_frames(text)
    for attempt in range(4):
        frames = q.generate(text, LANG[lang], persona, max_frames=lim, rng=rng)
        if len(frames) < 3 or len(frames) >= lim:
            continue                                       # rỗng hoặc đọc lan man (chạm trần khung) – lần thử nào cũng loại
        wav = q.decode(frames)
        if not silent(wav): return trim_silence(wav)
    raise RuntimeError('Không tạo được bản đọc đạt (rỗng / lan man) cho đoạn: ' + text[:40])


def synthesize(q, lang, segments, rng, loai='cau', hoan_thien=True):
    nh = CL.NHIP.get(lang) or dict(cau=PAUSES['sentence'], phay=PAUSES['comma'], doi_vai=PAUSES['turn'])
    limit = 60 if lang in ('zh', 'ja', 'ko') else 160
    plan = []
    for t, v in segments:
        ve = [(c, k) for x, k in CL.tach_ve(t, lang) for c in (sentence_chunks(x, limit) or [])] if loai != 'tu' else [(t, None)]
        if ve: plan.append((ve, v))
    if not plan: raise ValueError('Không có đoạn văn bản hợp lệ.')
    sil = lambda s: np.zeros(round(s * SAMPLE_RATE), np.float32)
    pieces = []
    for ti, (ve, voice) in enumerate(plan):
        for i, (c, nghi) in enumerate(ve):
            pieces.append(speak(q, c, lang, voice, rng))
            if i < len(ve) - 1: pieces.append(sil(nh['phay'] if nghi == 'phay' else nh['cau']))
        if ti < len(plan) - 1: pieces.append(sil(nh['doi_vai']))
    wav = np.concatenate(pieces)
    return CL.hoan_thien(wav, lang, loai, ' '.join(t for t, _ in segments)) if hoan_thien else wav


def job_segments(job, kana=False):
    lang, loai = job.get('lang'), job.get('loai')
    clean = lambda t: CL.lam_sach(t, lang, 'cau' if loai == 'hoi-thoai' else loai, kana)
    if job.get('loai') == 'hoi-thoai' and job.get('lines'):
        vai = job.get('vai') or {}
        segs = []
        for ln in job['lines']:
            label = vai.get(ln.get('vai')) or job.get('giong') or ('Nữ 4' if lang == 'zh' else 'Nữ 1')
            t = clean(ln.get('text'))
            if t: segs.append((t, voice_code(label, lang)))
        return segs
    t = job.get('text')
    if kana and lang == 'ja' and job.get('reading') and re.search(r'[\u3400-\u9fff]', t or '') and re.fullmatch(r'[\u3040-\u30ff・ー\s]+', job['reading']): t = job['reading']
    return [(clean(t), voice_code(job.get('giong') or ('Nữ 4' if lang == 'zh' else 'Nữ 1'), lang))]


def job_label(job):
    if job.get('loai') == 'hoi-thoai': return job.get('vai') or job.get('giong')
    return job.get('giong')


def to_mp3(wav, path, kbps=128):
    pcm = (np.clip(wav, -1, 1) * 32767).astype('<i2')
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    try:
        import lameenc
        enc = lameenc.Encoder(); enc.set_bit_rate(kbps); enc.set_in_sample_rate(SAMPLE_RATE); enc.set_channels(1); enc.set_quality(2)
        data = enc.encode(pcm.tobytes()) + enc.flush()
        with open(path, 'wb') as f: f.write(data)
    except ImportError:                                    # dự phòng: ffmpeg
        import subprocess
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-f', 's16le', '-ar', str(SAMPLE_RATE), '-ac', '1', '-i', '-',
                        '-b:a', f'{kbps}k', path], input=pcm.tobytes(), check=True)


# ---- chạy cả danh sách -----------------------------------------------------------------------------
def load_previous(out_dir, extra=()):
    prev = {}
    for p in [os.path.join(out_dir, 'ket-qua.json'), *extra]:
        try:
            with open(p, encoding='utf-8') as f:
                for r in json.load(f).get('ket_qua', []):
                    if r.get('trang_thai') == 'xong': prev[r['id']] = r
        except (OSError, ValueError):
            pass
    return prev


def run(jobs_path, out_dir, model_dir=None, limit=None, hours=None, device='auto', prev_results=(), log=print, require_gpu=False, so_ban=3, cham=True, kbps=128):
    t_start = time.time()
    with open(jobs_path, encoding='utf-8') as f:
        doc = json.load(f)
    jobs = doc['jobs'][:limit] if limit else doc['jobs']
    model_dir = model_dir or download_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model'))
    os.makedirs(out_dir, exist_ok=True)
    prev = load_previous(out_dir, prev_results)
    results = {}
    for j in jobs:                                         # giữ lại việc đã xong (file còn ở thư mục kết quả)
        r = prev.get(j['id'])
        if r and r.get('file') == j['file'] and os.path.exists(os.path.join(out_dir, j['file'])):
            results[j['id']] = r
    todo = [j for j in jobs if j['id'] not in results]
    log(f'{len(jobs)} việc · đã xong từ trước {len(results)} · cần đọc {len(todo)}')
    t0 = time.time(); q = QwenONNX(model_dir, device); log(f'Nạp model {time.time() - t0:.1f} s trên {q.device}')
    if require_gpu and not q.gpu:
        raise RuntimeError('Không chạy được trên GPU. Kaggle: Settings → Accelerator = GPU T4 x2 / P100, rồi chạy lại.')
    # TỰ CHẤM (Whisper): có faster-whisper thì đọc tối đa so_ban bản cho từ / câu, giữ bản Whisper nghe ra đúng nhất (dừng sớm khi đúng hẳn)
    ch = None
    if cham and so_ban > 1:
        try: ch = CL.Cham(device='cuda' if q.gpu else 'cpu'); log('Tự chấm bằng Whisper: bật')
        except Exception as e: log(f'Tự chấm bằng Whisper: tắt ({type(e).__name__}: {e}) – mỗi việc đọc 1 bản')
    NGUONG = {'tu': 0.0, 'cau': 0.1}

    def save():
        out = {'model': MODEL_NAME, 'tao_luc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
               'nguon': os.path.basename(jobs_path), 'center': doc.get('center'), 'thiet_bi': q.device,
               'ket_qua': [results.get(j['id']) or {'id': j['id'], 'trang_thai': 'loi', 'file': j['file'], 'loi': 'chưa chạy tới'} for j in jobs]}
        tmp = os.path.join(out_dir, 'ket-qua.json.tmp')
        with open(tmp, 'w', encoding='utf-8') as f: json.dump(out, f, ensure_ascii=False, indent=1)
        os.replace(tmp, os.path.join(out_dir, 'ket-qua.json'))

    times = []
    for n, j in enumerate(todo):
        if hours and time.time() - t_start > hours * 3600:
            log(f'Hết giờ ({hours} h): dừng, {len(todo) - n} việc để lần sau'); break
        t1 = time.time()
        try:
            tot, n_ban = None, (so_ban if ch and j.get('loai') in ('tu', 'cau') else 1)
            for k in range(n_ban):
                seed = int(hashlib.sha1((j['id'] + ('' if k == 0 else f'#{k}')).encode()).hexdigest()[:8], 16)   # bản 0 = như trước
                kana = j.get('lang') == 'ja' and k == n_ban - 1 and k > 0                                         # bản cuối tiếng Nhật: đọc theo kana
                segs = job_segments(j, kana)
                wav = synthesize(q, j['lang'], segs, np.random.default_rng(seed), j.get('loai'))
                if not ch or n_ban == 1: tot = (None, '', wav, k); break
                cer, nghe = ch.cer(wav, segs[0][0], j['lang'])
                if tot is None or cer < tot[0]: tot = (cer, nghe, wav, k)
                if cer <= NGUONG.get(j.get('loai'), 0.1): break
            cer, nghe, wav, k = tot
            to_mp3(wav, os.path.join(out_dir, j['file']), kbps)
            dur = round(len(wav) / SAMPLE_RATE, 2)
            results[j['id']] = {'id': j['id'], 'trang_thai': 'xong', 'file': j['file'], 'giong': job_label(j), 'thoi_luong': dur,
                                'giay_tao': round(time.time() - t1, 2)}
            if cer is not None: results[j['id']].update(cer=cer, nghe=nghe, ban=k + 1, can_nghe=cer > NGUONG.get(j.get('loai'), 0.1))
            times.append(time.time() - t1)
            log(f'[{n + 1}/{len(todo)}] {j["loai"]:9s} {time.time() - t1:5.1f}s → {dur:5.2f}s âm thanh  {j["file"]}  {clean(j.get("text"))[:30]}')
        except Exception as e:
            results[j['id']] = {'id': j['id'], 'trang_thai': 'loi', 'file': j['file'], 'loi': f'{type(e).__name__}: {e}'}
            log(f'[{n + 1}/{len(todo)}] LỖI {j["file"]}: {e}')
        if n % 10 == 9: save()
    save()
    ok = sum(1 for r in results.values() if r.get('trang_thai') == 'xong')
    if times:
        log(f'Xong {ok}/{len(jobs)} · trung bình {np.mean(times):.2f} s/việc · tổng {sum(times) / 60:.1f} phút · '
            f'talker {q.stats["talker"] / max(1, q.stats["tn"]) * 1000:.1f} ms/bước, cp {q.stats["cp"] / max(1, q.stats["cn"]) * 1000:.1f} ms/khung, '
            f'giải mã {q.stats["dec"]:.1f} s')
    return results


def main():
    ap = argparse.ArgumentParser(description='Đọc hàng loạt jobs.json bằng Qwen3-TTS (model của giong-noi)')
    ap.add_argument('--jobs', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--model', help='thư mục chứa file model (mặc định: tải về tools/doc-hang-loat/model)')
    ap.add_argument('--gioi-han', type=int, help='chỉ đọc N việc đầu (thử tốc độ)')
    ap.add_argument('--het-gio', type=float, help='dừng sau N giờ (Kaggle tối đa 12 h/phiên)')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--so-ban', type=int, default=3, help='số bản đọc tối đa cho mỗi từ / câu khi có Whisper tự chấm (mặc định 3)')
    ap.add_argument('--khong-cham', action='store_true', help='tắt tự chấm bằng Whisper (mỗi việc 1 bản như trước)')
    ap.add_argument('--kbps', type=int, default=128)
    a = ap.parse_args()
    try: sys.stdout.reconfigure(encoding='utf-8')     # Windows: cửa sổ lệnh mặc định cp1252
    except Exception: pass
    run(a.jobs, a.out, a.model, a.gioi_han, a.het_gio, 'cpu' if a.cpu else 'auto', so_ban=a.so_ban, cham=not a.khong_cham, kbps=a.kbps)


if __name__ == '__main__':
    main()
