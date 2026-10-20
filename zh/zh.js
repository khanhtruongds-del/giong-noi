// Mandarin grapheme-to-phoneme for Kokoro-82M v1.0, matching misaki ZHG2P(version=None) exactly:
//   cn2an.transform(text, 'an2cn') -> punctuation mapping -> jieba.lcut (HMM) per Han run
//   -> pypinyin lazy_pinyin(TONE3, neutral 5) per word -> pinyin_to_ipa (precomputed table).
// Sources ported: misaki (Apache-2.0), pypinyin (MIT), cn2an (MIT); jieba via jieba-wasm (MIT).
import initJieba, { cut } from './jieba_rs_wasm.js';

let CH, PH, PREFIX, IPA, ready = null;

export function loadZh(base) {
  if (!ready) ready = (async () => {
    const [c, p, i] = await Promise.all(['chars.json', 'phrases.json', 'ipa.json'].map(f => fetch(base + f).then(r => r.json())));
    CH = c; PH = p; IPA = i;
    PREFIX = new Set();
    for (const k of Object.keys(PH)) { const a = [...k]; for (let n = 1; n <= a.length; n++) PREFIX.add(a.slice(0, n).join('')); }
    await initJieba({ module_or_path: base + 'jieba_rs_wasm_bg.wasm' });
  })();
  return ready;
}

// ---- cn2an an2cn (mode "low" / "direct") -------------------------------------------------------
const NUM = '零一二三四五六七八九';
const UNIT = ['', '十', '百', '千', '万', '十', '百', '千', '亿', '十', '百', '千', '万', '十', '百', '千'];
function intConvert(s) {
  s = String(BigInt(s));
  const n = s.length;
  if (n > UNIT.length) throw new Error('too long');
  let out = '';
  for (let i = 0; i < n; i++) {
    const d = +s[i];
    if (d) out += NUM[d] + UNIT[n - i - 1];
    else {
      if (!((n - i - 1) % 4)) out += NUM[d] + UNIT[n - i - 1];
      if (i > 0 && out[out.length - 1] !== '零') out += NUM[d];
    }
  }
  out = out.replace(/零零/g, '零').replace(/零万/g, '万').replace(/零亿/g, '亿').replace(/亿万/g, '亿').replace(/^零+|零+$/g, '');
  out = out.replace(/([万亿])零([一二三四五六七八九][千])/g, '$1$2');
  if (out.startsWith('一十')) out = out.slice(1);
  return out || '零';
}
function an2cn(s, mode = 'low') {
  let sign = '';
  if (s[0] === '-') { sign = '负'; s = s.slice(1); }
  if (mode === 'direct') return sign + [...s].map(d => d === '.' ? '点' : NUM[+d]).join('');
  const parts = s.split('.');
  let out = intConvert(parts[0]);
  if (parts.length === 2) out += (parts[1] ? '点' : '') + [...parts[1].slice(0, 16)].map(d => NUM[+d]).join('');
  return sign + out;
}
const MEASURE = '斤|克|千克|公斤|吨|米|厘米|毫米|公里|升|毫升|元|角|分|个|只|条|张|块|瓶|杯|份|本|辆|台|匹|头|位|亩|小时|分钟|秒|天|半';
const safe = (f) => (m) => { try { return f(m); } catch { return m; } };
export function transformAn2Cn(text) {
  text = text.replace(/[０-９]/g, c => String.fromCharCode(c.charCodeAt(0) - 0xFEE0)).replace(/．/g, '.');
  text = text.replace(new RegExp(`\\d+(\\.\\d+)?-\\d+(\\.\\d+)?(?=(${MEASURE}))`, 'g'),
    safe(m => { const [a, b] = m.split('-'); return `${an2cn(a)}到${an2cn(b)}`; }));
  text = text.replace(/(\d{2,4}年)?(\d{1,2}月)?(\d{1,2}日)?/g,
    safe(m => m ? m.replace(/\d+(?=年)/g, x => an2cn(x, 'direct')).replace(/\d+/g, x => an2cn(x)) : m));
  text = text.replace(/\d+\/\d+/g, safe(m => { const [a, b] = m.split('/'); return `${an2cn(b)}分之${an2cn(a)}`; }));
  text = text.replace(/-?(\d+\.)?\d+%/g, safe(m => '百分之' + an2cn(m.slice(0, -1))));
  text = text.replace(/\d+℃/g, safe(m => an2cn(m.slice(0, -1)) + '摄氏度'));
  return text.replace(/-?(\d+\.)?\d+/g, safe(m => an2cn(m)));
}

// ---- pypinyin lazy_pinyin(style=TONE3, neutral_tone_with_five=True) ----------------------------
const MARKS = { 'ā': 'a1', 'á': 'a2', 'ǎ': 'a3', 'à': 'a4', 'ē': 'e1', 'é': 'e2', 'ě': 'e3', 'è': 'e4', 'ī': 'i1', 'í': 'i2', 'ǐ': 'i3', 'ì': 'i4',
  'ō': 'o1', 'ó': 'o2', 'ǒ': 'o3', 'ò': 'o4', 'ū': 'u1', 'ú': 'u2', 'ǔ': 'u3', 'ù': 'u4', 'ǖ': 'v1', 'ǘ': 'v2', 'ǚ': 'v3', 'ǜ': 'v4', 'ü': 'v0',
  'ń': 'n2', 'ň': 'n3', 'ǹ': 'n4', 'ḿ': 'm2' };
function tone3(py) {
  let tone = '5', out = '';
  for (const c of py.normalize('NFC')) { const m = MARKS[c]; if (m) { out += m[0]; if (m[1] !== '0') tone = m[1]; } else out += c; }
  return out + tone;
}
function* mmseg(text) {   // pypinyin.seg.mmseg.Seg(no_non_phrases=True).cut
  let remain = [...text];
  while (remain.length) {
    let lastValid = '', lastIdx = 0, broke = false;
    for (let i = 0; i < remain.length; i++) {
      const w = remain.slice(0, i + 1).join('');
      if (PREFIX.has(w)) { if (PH[w]) { lastValid = w; lastIdx = i + 1; } }
      else {
        if (lastValid) { yield lastValid; remain = remain.slice(lastIdx); } else { yield remain[0]; remain = remain.slice(1); }
        broke = true; break;
      }
    }
    if (!broke) {
      if (lastValid) { yield lastValid; remain = remain.slice(lastIdx); }
      else { const r = remain.join(''); if (PH[r]) yield r; else yield* remain; remain = []; }
    }
  }
}
function lazyPinyin(word) {
  const out = [];
  for (const piece of mmseg(word)) {
    if (PH[piece]) out.push(...PH[piece]); else for (const c of piece) out.push(CH[c] || c);
  }
  return out.map(tone3);
}

// ---- misaki ZHG2P(version=None) ------------------------------------------------------------------
function mapPunctuation(t) {
  return t.replaceAll('、', ', ').replaceAll('，', ', ').replaceAll('。', '. ').replaceAll('．', '. ').replaceAll('！', '! ')
    .replaceAll('：', ': ').replaceAll('；', '; ').replaceAll('？', '? ').replaceAll('«', ' “').replaceAll('»', '” ')
    .replaceAll('《', ' “').replaceAll('》', '” ').replaceAll('「', ' “').replaceAll('」', '” ').replaceAll('【', ' “')
    .replaceAll('】', '” ').replaceAll('（', ' (').replaceAll('）', ') ').trim();
}
export function g2pZh(text) {
  if (!text.trim()) return '';
  const t = mapPunctuation(transformAn2Cn(text));
  if (!t) return '';
  let isZh = /[一-鿿]/.test(t[0]), out = '';
  for (const seg of t.match(/[一-鿿]+|[^一-鿿]+/g)) {
    out += isZh ? cut(seg, true).map(w => lazyPinyin(w).map(p => IPA[p] ?? '').join('')).join(' ') : seg;
    isZh = !isZh;
  }
  return out.replaceAll('̯', '');
}
