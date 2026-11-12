# -*- coding: utf-8 -*-
"""CHẤT LƯỢNG ĐỌC (13/11/2026) – các bước quanh model, đo trên ~1.150 mẫu audio thật của 6 trung tâm (PLS Classroom docs/am-thanh.md mục 9–11):

  lam_sach(text, lang, loai)  bỏ ký hiệu từ điển máy hay đọc thành tiếng ("der Opa, -s" → "Der Opa Ness"): đuôi số nhiều tiếng Đức,
                              "(e)" tiếng Pháp, "+ Dat.", ghi chú trong ngoặc, nhãn vai "A:"; từ đơn thêm dấu kết thúc (ngữ điệu trần thuật)
  tach_ve(text, lang)         tách câu → vế (dấu phẩy, chấm phẩy…) để TỰ chèn quãng nghỉ đúng chuẩn của tiếng, không để model tự quyết
  NHIP[lang]                  quãng nghỉ (hết câu / dấu phẩy / đổi vai), lặng đầu – cuối file, tốc độ chuẩn (đơn vị/giây) – trung vị audio thật
  hoan_thien(wav, lang, ...)  kéo chậm phần có tiếng về tốc độ chuẩn (WSOLA, giữ cao độ, quãng nghỉ giữ nguyên), cân âm lượng, fade, đệm lặng
  Cham                        Whisper nghe lại → CER theo cách đọc (zh: pinyin có thanh, ja: kana, ko: hangul, khác: chữ cái) để chọn bản tốt nhất
"""
import re, unicodedata
import numpy as np

SR = 24000

# ---- nhịp chuẩn: trung vị audio thật của kho trung tâm (giây; tốc độ = âm tiết zh/ko, mora ja, chữ cái en/de/fr trên giây đọc) ----
NHIP = {
    'zh': dict(cau=0.67, phay=0.50, doi_vai=0.38, dau=0.23, cuoi=0.39, toc_cau=4.0, toc_tu=2.8),
    'ja': dict(cau=0.66, phay=0.45, doi_vai=0.27, dau=0.20, cuoi=0.34, toc_cau=6.8, toc_tu=5.8),
    'ko': dict(cau=0.35, phay=0.43, doi_vai=0.50, dau=0.14, cuoi=0.21, toc_cau=5.5, toc_tu=4.3),
    'en': dict(cau=0.37, phay=0.15, doi_vai=0.30, dau=0.20, cuoi=0.27, toc_cau=14.2, toc_tu=12.5),
    'de': dict(cau=0.47, phay=0.26, doi_vai=0.35, dau=0.14, cuoi=0.46, toc_cau=14.5, toc_tu=12.8),
    'fr': dict(cau=0.34, phay=0.32, doi_vai=0.33, dau=0.09, cuoi=0.37, toc_cau=14.6, toc_tu=11.5),
}

# ---- làm sạch chữ (cùng luật với PLS Classroom tools/chu-doc.js) ----
_RUBY = re.compile(r'\{([^{}|]+)\|([^{}]*)\}')
def lam_sach(text, lang, loai='cau', kana=False):
    s = unicodedata.normalize('NFC', str(text or ''))
    s = _RUBY.sub((lambda m: m.group(2)) if kana else (lambda m: m.group(1)), s)          # furigana: kanji (mặc định) hoặc kana
    s = re.sub(r'\{\{(.*?)\}\}', r'\1', s); s = re.sub(r'\[\[(.*?)\]\]', r'\1', s); s = re.sub(r'<[^<>]*>', '', s)
    s = re.sub(r',\s*[-–—](?:\s*[̈¨"])*[^\s,;()]*', '', s)                          # "der Tisch, -e" · "das Dorf, -¨er"
    s = re.sub(r'\+\s*(?:Dat|Akk|Gen|Nom|V-?ing|V|N|Adj|qn|qch|sb|sth)\b\.?', '', s, flags=re.I)
    s = re.sub(r'[（(]\s*(?:Sg|Pl|jdn|jdm|etw|qn|qch|sb|sth|sich)\.?\s*[)）]', '', s, flags=re.I)
    if lang == 'fr': s = re.sub(r'(\w)\((e|ne|le|se|ve|euse|rice|ère|ienne|ive)\)', r'\1', s, flags=re.I)
    if lang in ('de', 'fr', 'en'): s = re.sub(r'\s*/\s*', ', ', s)
    if lang == 'ja':
        s = re.sub(r'[（(]\s*[＝=][^）)]*[）)]', '', s); s = re.sub(r'[（(]([ぁ-ゟ]{1,3})[）)]', r'\1', s)
        s = re.sub(r'[＝=]', '、', s); s = re.sub(r'^[～〜]', '', s)
    if lang in ('zh', 'ko'): s = re.sub(r'[（(][^）)]*[）)]', '', s)
    s = re.sub(r'^\s*[A-Z가-힣]{1,3}\s*[:：]\s*', '', s)                                  # nhãn vai "A:" đầu câu
    s = re.sub(r'\s+', ' ', s); s = re.sub(r'\s+([,.!?。、，！？])', r'\1', s); s = s.strip(' ,，、')
    if loai == 'tu' and s and not re.search(r'[.!?。！？…]$', s): s += '。' if lang in ('zh', 'ja') else '.'
    return s

# ---- tách vế: câu → [(vế, loại nghỉ sau vế: 'cau' | 'phay' | None)] ----
_HET_CAU, _PHAY = '.!?。！？…', ',，、;；:：'
def tach_ve(text, lang):
    cjk = lang in ('zh', 'ja', 'ko'); ngan = 4 if cjk else 12        # vế quá ngắn thì ghép với vế sau (đọc rời nghe cụt)
    out, cur = [], ''
    for i, c in enumerate(text):
        cur += c
        nxt = text[i + 1] if i + 1 < len(text) else ''
        if c in _HET_CAU and nxt not in _HET_CAU: out.append([cur.strip(), 'cau']); cur = ''
        elif c in _PHAY: out.append([cur.strip(), 'phay']); cur = ''
    if cur.strip(): out.append([cur.strip(), None])
    gop = []
    for v in out:
        if not v[0]: continue
        if gop and len(re.sub(r'\W', '', gop[-1][0])) < ngan and gop[-1][1] == 'phay': gop[-1] = [gop[-1][0] + (' ' if not cjk else '') + v[0], v[1]]
        else: gop.append(v)
    if gop: gop[-1][1] = None
    return [tuple(v) for v in gop]

# ---- đơn vị đọc để đo tốc độ / chấm ----
_kks = None
def don_vi(s, lang):
    s = unicodedata.normalize('NFKC', str(s or '')).lower()
    if lang == 'zh':
        try:
            from pypinyin import lazy_pinyin, Style
            return lazy_pinyin(''.join(re.findall(r'[㐀-鿿]', s)), style=Style.TONE3, neutral_tone_with_five=True)
        except ImportError: return re.findall(r'[㐀-鿿]', s)
    if lang == 'ja':
        global _kks
        try:
            import pykakasi
            _kks = _kks or pykakasi.kakasi(); s = ''.join(x['hira'] for x in _kks.convert(s))
        except ImportError: pass
        return list(re.sub(r'[^ぁ-ゟ一-鿿]', '', s))
    if lang == 'ko': return list(re.sub(r'[^가-힣]', '', s))
    if lang == 'fr': s = ' '.join(re.sub(r'(?:ers|er|ez|ées|és|ée|é)$', 'e', re.sub(r'(?<=\w)[sxt]$', '', w)) for w in re.findall(r"[\w'’-]+", s))
    return list(re.sub(r'[^\w]|_|\d', '', s))

def lev(a, b):
    if not a: return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

# ---- hoàn thiện: tốc độ, âm lượng, fade, lặng đầu – cuối ----
def _nang(x, w=240):
    return np.sqrt(np.convolve(x.astype(np.float32) ** 2, np.ones(w) / w, 'same'))

def _wsola(x, k, n=960, tim=240):
    if abs(k - 1) < 0.02 or len(x) < n * 2: return x
    h_ra = n // 2; h_vao = h_ra / k; win = np.hanning(n).astype(np.float32)
    so = int((len(x) - n - tim) / h_vao)
    if so < 2: return x
    y = np.zeros(so * h_ra + n, np.float32); wt = np.zeros_like(y); truoc = None
    for i in range(so):
        p = int(i * h_vao)
        if truoc is not None:
            lo, hi = max(0, p - tim), min(len(x) - n, p + tim); mau = truoc[h_ra:h_ra + n // 2]; best, bv = p, -1e9
            for q in range(lo, hi, 8):
                v = float(np.dot(mau, x[q:q + n // 2]))
                if v > bv: bv, best = v, q
            p = best
        f = x[p:p + n]
        if len(f) < n: break
        y[i * h_ra:i * h_ra + n] += f * win; wt[i * h_ra:i * h_ra + n] += win; truoc = f
    wt[wt < 1e-3] = 1
    return y / wt

def keo_cham(x, k):
    """chỉ kéo các đoạn có tiếng; quãng lặng ≥ 120 ms giữ nguyên độ dài"""
    if k <= 1.02: return x
    e = _nang(x)[::240]; noi = e > max(np.percentile(e, 95) * 0.06, 1e-4); doan, i = [], 0
    while i < len(noi):
        j = i
        while j < len(noi) and noi[j] == noi[i]: j += 1
        doan.append([bool(noi[i]) or (j - i) < 12, i * 240, min(len(x), j * 240)]); i = j
    gop = []
    for d in doan:
        if gop and gop[-1][0] == d[0]: gop[-1][2] = d[2]
        else: gop.append(d)
    return np.concatenate([_wsola(x[a:b], k) if co else x[a:b] for co, a, b in gop]).astype(np.float32)

def thoi_gian_noi(x):
    e = _nang(x)[::240]; return float((e > max(np.percentile(e, 95) * 0.06, 1e-4)).sum()) * 240 / SR

def hoan_thien(x, lang, loai, text, toc_toi_da=1.6):
    """x: audio đã ghép (đã cắt lặng hai đầu). → kéo về tốc độ chuẩn (nhanh hơn > 8 %), −20 dBFS, đỉnh ≤ −1 dBFS, fade 10 ms, đệm lặng"""
    nh = NHIP.get(lang)
    if not nh or not len(x): return x
    n, t = len(don_vi(text, lang)), thoi_gian_noi(x); chuan = nh['toc_tu'] if loai == 'tu' else nh['toc_cau']
    if n and t > 0.2 and n / t > chuan * 1.08: x = keo_cham(x, min(toc_toi_da if loai != 'tu' else 1.3, (n / t) / chuan))
    e = _nang(x, 480); co = x[e > e.max() * 0.1]; rms = float(np.sqrt(np.mean(co ** 2))) if len(co) else 1e-3
    x = x * (0.1 / max(rms, 1e-4)); dinh = float(np.abs(x).max())
    if dinh > 0.891: x = x * (0.891 / dinh)
    m = min(len(x) // 4, SR // 100)
    if m > 1: r = np.linspace(0, 1, m, dtype=np.float32); x[:m] *= r; x[-m:] *= r[::-1]
    return np.concatenate([np.zeros(int(SR * max(0.05, nh['dau'] - 0.06)), np.float32), x.astype(np.float32), np.zeros(int(SR * max(0.1, nh['cuoi'] - 0.04)), np.float32)])

# ---- chấm bằng Whisper (tuỳ chọn: pip install faster-whisper pypinyin pykakasi) ----
class Cham:
    def __init__(self, model='large-v3-turbo', device='cuda', download_root=None):
        import glob, os, sys
        if os.name == 'nt':   # Windows: cuBLAS / cuDNN cài bằng pip (nvidia-*) nằm trong site-packages/nvidia/*/bin
            for d in glob.glob(os.path.join(sys.prefix, 'Lib', 'site-packages', 'nvidia', '*', 'bin')):
                os.add_dll_directory(d); os.environ['PATH'] = d + os.pathsep + os.environ['PATH']
        from faster_whisper import WhisperModel
        self.m = WhisperModel(model, device=device, compute_type='float16' if device == 'cuda' else 'int8', download_root=download_root)
    def cer(self, wav24k, text, lang):
        x = wav24k.astype(np.float32); idx = np.arange(0, len(x), 1.5)                  # 24 kHz → 16 kHz
        x16 = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
        seg, _ = self.m.transcribe(x16, language=lang, beam_size=5, condition_on_previous_text=False, without_timestamps=True, vad_filter=False, temperature=0)
        nghe = ''.join(s.text for s in seg).strip()
        r, h = don_vi(text, lang), don_vi(nghe, lang)
        return round(lev(r, h) / max(1, len(r)), 3), nghe
