// Syllable drill for Chinese: every syllable is a pre-recorded clip (am-tiet/{voice}/{base}{tone}.mp3, Qwen3-TTS 1.7B,
// Serena / Aiden) played one after another with a pause. Needs no WebGPU. Hanzi → pinyin with pinyin-pro (citation tones,
// no tone sandhi). Pinyin may carry tone numbers (ni3hao3) or tone marks (nǐ hǎo); a syllable without a tone plays all of its tones.
const SR = 24000, ROOT = new URL('am-tiet/', import.meta.url);
let index = null, pinyinPro = null;
const clips = new Map();   // "voice/base+tone" -> Float32Array

async function loadIndex() { return (index ??= await (await fetch(new URL('index.json', ROOT))).json()); }
function loadPinyin() {
  if (pinyinPro) return pinyinPro;
  return (pinyinPro = new Promise((ok, fail) => {
    const s = document.createElement('script');
    s.src = new URL('vendor/pinyin-pro.min.js?v=3.26.0', import.meta.url);
    s.onload = () => ok(window.pinyinPro); s.onerror = () => fail(new Error('Không tải được bộ chuyển chữ Hán sang pinyin.'));
    document.head.append(s);
  }));
}

// Split a run of letters ("nihao") into known bases with the fewest pieces; null if impossible.
function segment(run, tones) {
  const norm = (p) => (tones[p] ? p : /^[jqxy]v/.test(p) && tones[p.replace('v', 'u')] ? p.replace('v', 'u') : null);
  const best = [[]];
  for (let i = 1; i <= run.length; i++) {
    for (let j = Math.max(0, i - 6); j < i; j++) {
      const b = best[j] && norm(run.slice(j, i));
      if (b && (!best[i] || best[j].length + 1 < best[i].length)) best[i] = [...best[j], b];
    }
  }
  return best[run.length] || null;
}

export const hasHan = (s) => /\p{Script=Han}/u.test(s);

// → { items: [{ base, tone }], unknown: ['xx', ...] }
export async function parse(src) {
  const { tones } = await loadIndex();
  let s = src;
  const MARK = /[āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]/i;
  if (hasHan(s) || MARK.test(s)) {
    const P = await loadPinyin();
    if (hasHan(s)) s = s.replace(/\p{Script=Han}+/gu, (run) => ' ' + P.pinyin(run, { toneType: 'num', type: 'array', v: true, toneSandhi: false }).join(' ') + ' ');
    // Only words that carry a tone mark: convert() would also give unmarked pinyin a neutral tone ("ma" → "ma0").
    s = s.replace(/[a-zü:āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ]+/gi, (w) => (MARK.test(w) ? P.convert(w, { format: 'symbolToNum' }) : w));
  }
  s = s.toLowerCase().replace(/ü|u:/g, 'v');
  const items = [], unknown = [];
  for (const [, run, t] of s.matchAll(/([a-z]+)([0-5]?)/g)) {
    const bases = segment(run, tones);
    if (!bases) { unknown.push(run + t); continue; }
    bases.forEach((base, i) => {
      const have = tones[base], want = i === bases.length - 1 ? t : '';
      if (!want) for (const x of have) items.push({ base, tone: x });
      else if (have.includes(want)) items.push({ base, tone: want });
      else if (want === '0' || want === '5') items.push({ base, tone: have[0] });   // neutral tone: no recording, use the first tone there is
      else unknown.push(base + want);
    });
  }
  return { items, unknown };
}

async function clip(voice, key) {
  const id = `${voice}/${key}`;
  if (clips.has(id)) return clips.get(id);
  const res = await fetch(new URL(`${id}.mp3`, ROOT));
  if (!res.ok) throw new Error(`Không tải được âm ${key} (lỗi ${res.status}).`);
  const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const buf = await new Ctx(1, SR, SR).decodeAudioData(await res.arrayBuffer());
  const data = buf.getChannelData(0).slice();
  clips.set(id, data);
  return data;
}

// Fetch and decode every clip the items need (cached). onProgress(done, total); returns false if cancelled.
export async function load(items, gender, onProgress, cancelled) {
  const { voices } = await loadIndex(), voice = voices[gender] || voices.female;
  const keys = [...new Set(items.map((x) => x.base + x.tone))];
  let done = 0;
  for (let i = 0; i < keys.length; i += 8) {
    await Promise.all(keys.slice(i, i + 8).map((k) => clip(voice, k).then(() => onProgress?.(++done, keys.length))));
    if (cancelled?.()) return false;
  }
  return true;
}

// Loaded items → one Float32Array at 24 kHz: 0.3 s lead-in, clips (each passed through `shape`, e.g. a speed change)
// separated by `gap` seconds.
export function build(items, gender, gap, shape = (c) => c) {
  const voice = index.voices[gender] || index.voices.female, shaped = new Map();
  const parts = items.map((x) => {
    const k = x.base + x.tone;
    if (!shaped.has(k)) shaped.set(k, shape(clips.get(`${voice}/${k}`)));
    return shaped.get(k);
  });
  const pause = Math.round(gap * SR), lead = Math.round(0.3 * SR);
  const out = new Float32Array(lead + parts.reduce((n, p) => n + p.length, 0) + pause * Math.max(0, parts.length - 1) + lead);
  let at = lead;
  parts.forEach((p, i) => { out.set(p, at); at += p.length + (i < parts.length - 1 ? pause : 0); });
  return out;
}
export const SAMPLE_RATE = SR;
