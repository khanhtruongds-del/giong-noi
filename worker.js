// Inference worker: Supertonic 3 (en, fr, de, ja, ko) and Kokoro-82M-v1.1-zh (zh) with ONNX Runtime Web.
// Everything runs on the user's machine; model files are fetched once and kept in the Cache API.
import * as ort from './vendor/ort.wasm.min.mjs';
import { loadZh11, g2pZh11 } from './zh/zh11.js';
import { sentenceChunks } from './text.js';

const BASE = new URL('./', import.meta.url).href;
const CACHE = 'giong-noi-models-v1';
ort.env.wasm.wasmPaths = BASE + 'vendor/';
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

let manifest = null;
const engines = {};            // name -> loaded engine
let cancelled = false;

const post = (type, data = {}) => self.postMessage({ type, ...data });

async function getManifest() {
  if (!manifest) {
    manifest = await (await fetch(BASE + 'models/manifest.json', { cache: 'no-cache' })).json();
    pruneCache(manifest).catch(() => {});
  }
  return manifest;
}

// Drop cached parts of engines that are no longer published (e.g. the old Chinese model).
async function pruneCache(m) {
  const keep = new Set();
  for (const [name, e] of Object.entries(m.engines))
    for (const f of Object.values(e.files)) for (const part of f.parts) keep.add(BASE + 'models/' + name + '/' + part);
  const cache = await caches.open(CACHE);
  for (const req of await cache.keys()) if (!keep.has(req.url)) await cache.delete(req);
}

// ---- download with cache + progress ---------------------------------------------------------------
async function fetchPart(cache, url) {
  let res = await cache.match(url);
  if (!res) {
    res = await fetch(url);
    if (!res.ok) throw new Error(`Không tải được dữ liệu giọng (lỗi ${res.status}). Kiểm tra kết nối mạng rồi thử lại.`);
    await cache.put(url, res.clone());
  }
  return res;
}

async function loadFiles(engine, onBytes) {
  const m = await getManifest();
  const spec = m.engines[engine];
  const cache = await caches.open(CACHE);
  const out = {};
  for (const [name, file] of Object.entries(spec.files)) {
    const buf = new Uint8Array(file.size);
    let off = 0;
    for (const part of file.parts) {
      const res = await fetchPart(cache, BASE + 'models/' + engine + '/' + part);
      const b = new Uint8Array(await res.arrayBuffer());
      buf.set(b, off); off += b.length; onBytes(b.length);
    }
    out[name] = buf;
  }
  return out;
}

async function cachedBytes(engine) {
  const m = await getManifest();
  const cache = await caches.open(CACHE);
  let have = 0;
  for (const file of Object.values(m.engines[engine].files))
    for (const part of file.parts) if (await cache.match(BASE + 'models/' + engine + '/' + part)) have += 1;
  const total = Object.values(m.engines[engine].files).reduce((n, f) => n + f.parts.length, 0);
  return { complete: have === total, parts: have, total, bytes: m.engines[engine].bytes };
}

const session = (bytes) => ort.InferenceSession.create(bytes, { executionProviders: ['wasm'], graphOptimizationLevel: 'all' });

async function ensureEngine(name) {
  if (engines[name]) return engines[name];
  const m = await getManifest();
  const total = m.engines[name].bytes;
  let got = 0;
  const st = await cachedBytes(name);
  post('status', { message: st.complete ? 'Đang nạp giọng…' : 'Đang tải dữ liệu giọng lần đầu…' });
  const files = await loadFiles(name, (n) => { got += n; post('download', { engine: name, got, total }); });
  post('status', { message: 'Đang nạp giọng…' });
  engines[name] = name === 'supertonic-3' ? await makeSupertonic(files) : await makeKokoro(files, name);
  return engines[name];
}

// ---- Supertonic 3 (port of supertonic/core.py, MIT) ---------------------------------------------
const EMOJI = /[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F700}-\u{1F77F}\u{1F780}-\u{1F7FF}\u{1F800}-\u{1F8FF}\u{1F900}-\u{1F9FF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}☀-⛿✀-➿\u{1F1E6}-\u{1F1FF}]+/gu;
const SYMBOLS = [['–', '-'], ['‑', '-'], ['—', '-'], ['¯', ' '], ['_', ' '], ['“', '"'], ['”', '"'],
  ['‘', "'"], ['’', "'"], ['´', "'"], ['`', "'"], ['[', ' '], [']', ' '], ['|', ' '], ['/', ' '], ['#', ' '], ['→', ' '], ['←', ' ']];
function stPreprocess(text, lang) {
  text = text.normalize('NFKD').replace(EMOJI, '');
  for (const [a, b] of SYMBOLS) text = text.replaceAll(a, b);
  text = text.replace(/[♥☆♡©\\]/g, '');
  text = text.replaceAll('@', ' at ').replaceAll('e.g.,', 'for example, ').replaceAll('i.e.,', 'that is, ');
  for (const p of [',', '.', '!', '?', ';', ':', "'"]) text = text.replaceAll(' ' + p, p);
  text = text.replace(/(["'`])\1+/g, '$1').replace(/\s+/g, ' ').trim();
  if (!/[.!?;:,'"')\]}…。」』】〉》›»]$/.test(text)) text += '.';
  return `<${lang}>${text}</${lang}>`;
}

function randn(n) {   // Box-Muller
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u = 1 - Math.random(), v = Math.random(), r = Math.sqrt(-2 * Math.log(u));
    out[i] = r * Math.cos(2 * Math.PI * v); if (i + 1 < n) out[i + 1] = r * Math.sin(2 * Math.PI * v);
  }
  return out;
}

async function makeSupertonic(files) {
  const dec = new TextDecoder();
  const cfg = JSON.parse(dec.decode(files['tts.json']));
  const indexer = JSON.parse(dec.decode(files['unicode_indexer.json']));
  const s = {
    kind: 'supertonic', sr: cfg.ae.sample_rate, chunk: cfg.ae.base_chunk_size * cfg.ttl.chunk_compress_factor,
    ldim: cfg.ttl.latent_dim * cfg.ttl.chunk_compress_factor, indexer, styles: {},
    dp: await session(files['duration_predictor.onnx']), enc: await session(files['text_encoder.onnx']),
    est: await session(files['vector_estimator.onnx']), voc: await session(files['vocoder.onnx']),
  };
  return s;
}

async function stStyle(s, id) {
  if (!s.styles[id]) {
    const j = await (await fetch(BASE + 'models/supertonic-3/voice_styles/' + id + '.json')).json();
    const t = (o) => new ort.Tensor('float32', Float32Array.from(o.data.flat(Infinity)), o.dims);
    s.styles[id] = { ttl: t(j.style_ttl), dp: t(j.style_dp) };
  }
  return s.styles[id];
}

function stUnsupported(s, text) {
  const bad = new Set();
  for (const ch of new Set(text)) {
    const p = stPreprocess(ch, 'na');
    for (const c of p.slice(4, -5)) { const code = c.charCodeAt(0); if (code >= s.indexer.length || s.indexer[code] < 0) bad.add(ch); }
  }
  return [...bad];
}

async function stSynth(s, text, lang, style, steps, speed) {
  const t = stPreprocess(text, lang);
  const ids = new BigInt64Array(t.length);
  for (let i = 0; i < t.length; i++) { const v = s.indexer[t.charCodeAt(i)]; ids[i] = BigInt(v === undefined || v < 0 ? 0 : v); }
  const L = t.length;
  const textIds = new ort.Tensor('int64', ids, [1, L]);
  const textMask = new ort.Tensor('float32', new Float32Array(L).fill(1), [1, 1, L]);
  const dur = (await s.dp.run({ text_ids: textIds, style_dp: style.dp, text_mask: textMask })).duration.data[0] / speed;
  const textEmb = (await s.enc.run({ text_ids: textIds, style_ttl: style.ttl, text_mask: textMask })).text_emb;
  const wavLen = Math.floor(dur * s.sr);
  const T = Math.floor((wavLen + s.chunk - 1) / s.chunk);
  let xt = new ort.Tensor('float32', randn(s.ldim * T), [1, s.ldim, T]);
  const latentMask = new ort.Tensor('float32', new Float32Array(T).fill(1), [1, 1, T]);
  const total = new ort.Tensor('float32', new Float32Array([steps]), [1]);
  for (let k = 0; k < steps; k++) {
    if (cancelled) throw new Error('cancelled');
    xt = (await s.est.run({ noisy_latent: xt, text_emb: textEmb, style_ttl: style.ttl, text_mask: textMask,
      latent_mask: latentMask, current_step: new ort.Tensor('float32', new Float32Array([k]), [1]), total_step: total })).denoised_latent;
  }
  const wav = (await s.voc.run({ latent: xt })).wav_tts.data;
  return wav.slice(0, Math.min(wav.length, wavLen));
}

// ---- Kokoro-82M-v1.1-zh (ONNX exported by the project; G2P = misaki ZHG2P v1.1 port) -------------
async function makeKokoro(files, name) {
  const cfg = JSON.parse(new TextDecoder().decode(files['config.json']));
  await loadZh11(BASE + 'zh/');
  return { kind: 'kokoro', name, sr: cfg.sample_rate || 24000, vocab: cfg.vocab, styles: {}, sess: await session(files['kokoro.onnx']) };
}

async function kkStyle(k, id) {
  if (!k.styles[id]) k.styles[id] = new Float32Array(await (await fetch(BASE + 'models/' + k.name + '/voices/' + id + '.bin')).arrayBuffer());
  return k.styles[id];
}

function kkFit(chunk) {   // keep the phoneme string within Kokoro's 510-symbol context
  if (g2pZh11(chunk).length <= 510 || chunk.length < 8) return [chunk];
  const half = sentenceChunks(chunk, Math.max(4, chunk.length >> 1));
  return half.length > 1 ? half.flatMap(kkFit) : [chunk];
}

async function kkSynth(k, text, style, speed) {
  const ps = g2pZh11(text).slice(0, 510);
  const ids = [...ps].map(c => k.vocab[c]).filter(v => v !== undefined);
  if (!ids.length) return new Float32Array(0);
  const row = Math.min(ps.length, 510) - 1;
  const ref = style.subarray(row * 256, row * 256 + 256);
  const input = new BigInt64Array(ids.length + 2);
  ids.forEach((v, i) => { input[i + 1] = BigInt(v); });
  const out = await k.sess.run({ input_ids: new ort.Tensor('int64', input, [1, input.length]),
    ref_s: new ort.Tensor('float32', ref, [1, 256]), speed: new ort.Tensor('float32', new Float32Array([speed]), []) });
  return out.waveform.data;
}

// ---- jobs -----------------------------------------------------------------------------------------
// Kokoro pads every chunk with silence; trim it so the configured pauses alone set the rhythm.
function trimSilence(x, sr, margin = 0.04, head = true, tail = true) {
  let peak = 0; for (const v of x) peak = Math.max(peak, Math.abs(v));
  if (!peak) return x;
  const th = peak * 0.02, win = Math.round(sr * 0.01);
  const loud = (i) => { for (let j = i; j < Math.min(x.length, i + win); j++) if (Math.abs(x[j]) > th) return true; return false; };
  let a = 0; while (a < x.length && !loud(a)) a += win;
  let b = x.length; while (b > a && !loud(Math.max(0, b - win))) b -= win;
  const m = Math.round(sr * margin);
  return x.subarray(head ? Math.max(0, a - m) : 0, tail ? Math.min(x.length, b + m) : x.length);
}

async function synthesize({ engine, lang, segments, voice, text, speed, steps, commaBreak = false, pauses = { sentence: 0.3, comma: 0.15, turn: 0.6 } }) {
  cancelled = false;
  const e = await ensureEngine(engine);
  if (!segments) segments = [{ text, voice }];
  // Each segment is one speaker turn (or the whole text in single-voice mode).
  const plan = [];
  for (const seg of segments) {
    let chunks;
    if (e.kind === 'supertonic') {
      const bad = stUnsupported(e, seg.text);
      if (bad.length) throw new Error('Giọng này không đọc được các ký tự: ' + bad.join(' ') + '. Hãy xoá hoặc thay chúng rồi thử lại.');
      chunks = sentenceChunks(seg.text, lang === 'ko' ? 120 : 300, { group: false, commaBreak });
    } else chunks = sentenceChunks(seg.text, 120, { group: false, commaBreak }).flatMap(kkFit);
    if (chunks.length) plan.push({ chunks, voice: seg.voice });
  }
  const total = plan.reduce((n, p) => n + p.chunks.length, 0);
  if (!total) throw new Error('Không có đoạn văn bản hợp lệ.');
  const silence = (s) => new Float32Array(Math.round(s * e.sr));
  // A sentence gets the longer pause; a clause that was split at a comma gets the shorter one; a new speaker the longest.
  const pauseFor = (c) => silence(/[、，,]$/.test(c.trim()) ? pauses.comma : pauses.sentence);
  const pieces = [];
  let done = 0;
  for (let t = 0; t < plan.length; t++) {
    const { chunks, voice } = plan[t];
    const style = e.kind === 'supertonic' ? await stStyle(e, voice) : await kkStyle(e, voice);
    for (let i = 0; i < chunks.length; i++) {
      if (cancelled) throw new Error('cancelled');
      post('progress', { done: done++, total });
      const wav = e.kind === 'supertonic' ? await stSynth(e, chunks[i], lang, style, steps, speed) : await kkSynth(e, chunks[i], style, speed);
      // Kokoro: always trim. Supertonic: keep its own sentence rhythm, trim only where one speaker hands over to the next.
      const edge = plan.length > 1 && (i === 0 || i === chunks.length - 1);
      pieces.push(e.kind === 'kokoro' ? trimSilence(wav, e.sr) : edge ? trimSilence(wav, e.sr, 0.04, i === 0 && t > 0, i === chunks.length - 1 && t < plan.length - 1) : wav);
      if (i < chunks.length - 1) pieces.push(pauseFor(chunks[i]));
    }
    if (t < plan.length - 1) pieces.push(silence(pauses.turn ?? 0.6));
  }
  const n = pieces.reduce((a, p) => a + p.length, 0);
  const audio = new Float32Array(n);
  let off = 0; for (const p of pieces) { audio.set(p, off); off += p.length; }
  post('done', { audio, sampleRate: e.sr, chunks: total });
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'synthesize') await synthesize(data);
    else if (data.type === 'status') {
      const m = await getManifest();
      const out = {};
      for (const name of Object.keys(m.engines)) out[name] = await cachedBytes(name);
      post('cache', { engines: out });
    } else if (data.type === 'clear') {
      await caches.delete(CACHE); for (const k of Object.keys(engines)) delete engines[k];
      post('cleared');
    } else if (data.type === 'cancel') cancelled = true;
  } catch (err) {
    post(err.message === 'cancelled' ? 'cancelled' : 'error', { message: String(err.message || err) });
  }
};
