// Qwen3-TTS 0.6B (Apache-2.0, Alibaba Qwen team) in the browser: talker on WebGPU (8-bit weights, KV cache kept on the GPU),
// code predictor and codec decoder in fp32 on WASM. Model files come from the Hugging Face repo in config.json and are kept
// in the Cache API after the first download. Same message protocol as worker.js.
import * as ort from './vendor/gpu/ort.webgpu.min.mjs';
import { Tokenizer } from './vendor/tokenizers.min.mjs';
import { QwenTTS, SAMPLE_RATE } from './qwen.js';
import { sentenceChunks } from './text.js';

const BASE = new URL('./', import.meta.url).href;
const CACHE = 'giong-noi-qwen-v1';
ort.env.wasm.wasmPaths = BASE + 'vendor/gpu/';
// Multi-threaded WASM inside a worker deadlocks session creation together with WebGPU (seen on Chromium 2026-10), so keep 1 thread
// and run the talker and decoder on WebGPU.
ort.env.wasm.numThreads = 1;
const LANG = { de: 'german', en: 'english', fr: 'french', zh: 'chinese', ja: 'japanese', ko: 'korean' };

const EP = { cp: 'wasm', dec: 'webgpu' };   // measured on M4: cp wasm 87 ms/frame vs 400 ms on WebGPU; decoder faster on WebGPU
let spec = null, tts = null, voices = null, cancelled = false;
const post = (type, data = {}) => self.postMessage({ type, ...data });
const url = (f) => spec.base + f;

async function getSpec() {
  if (!spec) spec = (await (await fetch(BASE + 'config.json', { cache: 'no-cache' })).json()).qwen;
  return spec;
}

// ---- download with cache + progress --------------------------------------------------------------
async function fetchFile(cache, f, onBytes) {
  const u = url(f);
  let res = await cache.match(u);
  if (res) { const b = new Uint8Array(await res.arrayBuffer()); onBytes(b.length); return b; }
  res = await fetch(u);
  if (!res.ok) throw new Error(`Không tải được dữ liệu giọng (lỗi ${res.status}). Kiểm tra kết nối mạng rồi thử lại.`);
  const size = +res.headers.get('content-length') || spec.files[f] || 0;
  const buf = new Uint8Array(size), rd = res.body.getReader();
  let off = 0, chunks = null;
  for (;;) {
    const { done, value } = await rd.read();
    if (done) break;
    if (!chunks && off + value.length <= buf.length) buf.set(value, off); else (chunks ||= [buf.subarray(0, off)]).push(value);
    off += value.length; onBytes(value.length);
  }
  let out = buf.subarray(0, off);
  if (chunks) { out = new Uint8Array(off); let o = 0; for (const c of chunks) { out.set(c, o); o += c.length; } }
  await cache.put(u, new Response(out, { headers: { 'content-type': 'application/octet-stream' } }));
  return out;
}

async function cacheStatus() {
  await getSpec();
  const cache = await caches.open(CACHE), names = Object.keys(spec.files);
  let have = 0; for (const f of names) if (await cache.match(url(f))) have++;
  return { complete: have === names.length, parts: have, total: names.length, bytes: spec.bytes };
}

async function hasWebGPU() {
  try { return !!(navigator.gpu && await navigator.gpu.requestAdapter()); } catch { return false; }
}

async function ensure() {
  if (tts) return tts;
  await getSpec();
  if (!(await hasWebGPU())) throw new Error('Trình duyệt này chưa bật WebGPU nên chưa chạy được giọng mới. Hãy dùng Chrome hoặc Edge bản mới trên máy tính.');
  const st = await cacheStatus();
  post('status', { message: st.complete ? 'Đang nạp giọng…' : 'Đang tải dữ liệu giọng lần đầu…' });
  const cache = await caches.open(CACHE);
  let got = 0; const onBytes = (n) => { got += n; if (!st.complete) post('download', { engine: 'qwen3', got, total: spec.bytes }); };
  const get = (f) => fetchFile(cache, f, onBytes);
  const dec = new TextDecoder(), json = async (f) => JSON.parse(dec.decode(await get(f)));
  const tok = new Tokenizer(await json('tokenizer.json'), await json('tokenizer_config.json'));
  voices = await json('voices.json');
  const text = await get('text.f16'), codec = await get('codec.f32');
  const tables = { text: { f16: new Uint16Array(text.buffer, text.byteOffset, text.byteLength >> 1) }, codec: new Float32Array(codec.buffer, codec.byteOffset, codec.byteLength >> 2) };
  const kv = {}; for (let i = 0; i < 28; i++) { kv[`present_key_${i}`] = 'gpu-buffer'; kv[`present_value_${i}`] = 'gpu-buffer'; }
  const sessions = {};
  sessions.talker = await ort.InferenceSession.create(await get('talker_decode_q8nb.onnx'), { executionProviders: ['webgpu'], graphOptimizationLevel: 'all', preferredOutputLocation: kv });
  post('status', { message: 'Đang nạp giọng…' });
  sessions.cpf = await ort.InferenceSession.create(await get('cp_fused2_f32.onnx'), { executionProviders: [EP.cp], graphOptimizationLevel: 'all' });
  sessions.dec = await ort.InferenceSession.create(await get('decoder_f32.onnx'), { executionProviders: [EP.dec], graphOptimizationLevel: 'all' });
  tts = new QwenTTS(ort, sessions, tables, tok);
  return tts;
}

// ---- jobs -------------------------------------------------------------------------------------------
function trimSilence(x, sr, margin = 0.05) {
  let peak = 0; for (const v of x) peak = Math.max(peak, Math.abs(v));
  if (!peak) return x;
  const th = peak * 0.02, win = Math.round(sr * 0.01);
  const loud = (i) => { for (let j = i; j < Math.min(x.length, i + win); j++) if (Math.abs(x[j]) > th) return true; return false; };
  let a = 0; while (a < x.length && !loud(a)) a += win;
  let b = x.length; while (b > a && !loud(Math.max(0, b - win))) b -= win;
  const m = Math.round(sr * margin);
  return x.subarray(Math.max(0, a - m), Math.min(x.length, b + m));
}
const CJK = /[぀-ヿ㐀-鿿가-힯]/u;
// Upper bound on frames for a chunk (12.5 frames per second), so a rare run-on generation stops instead of looping.
const maxFrames = (t) => Math.min(900, 40 + Math.ceil([...t].reduce((n, c) => n + (CJK.test(c) ? 0.6 : /\s/.test(c) ? 0.05 : 0.17), 0) * 12.5 * 1.6));

async function speak(q, text, lang, voice) {
  const persona = voices[voice];
  if (!persona) throw new Error('Không tìm thấy giọng ' + voice + '.');
  const lim = maxFrames(text);
  for (let attempt = 0; ; attempt++) {
    const frames = await q.generate(text, LANG[lang], persona, { maxFrames: lim });
    if (cancelled) throw new Error('cancelled');
    if (frames.length < lim || attempt) return trimSilence(await q.decode(frames), SAMPLE_RATE);
  }
}

async function synthesize({ lang, segments, voice, text, pauses = { sentence: 0.3, comma: 0.15, turn: 0.6 } }) {
  cancelled = false;
  const q = await ensure();
  if (!segments) segments = [{ text, voice }];
  const limit = ['zh', 'ja', 'ko'].includes(lang) ? 60 : 160;   // a chunk stays short enough for one natural breath group
  const plan = segments.map(s => ({ chunks: sentenceChunks(s.text, limit, { group: true }), voice: s.voice })).filter(p => p.chunks.length);
  const total = plan.reduce((n, p) => n + p.chunks.length, 0);
  if (!total) throw new Error('Không có đoạn văn bản hợp lệ.');
  const silence = (s) => new Float32Array(Math.round(s * SAMPLE_RATE));
  const pieces = []; let done = 0;
  for (let t = 0; t < plan.length; t++) {
    const { chunks, voice } = plan[t];
    for (let i = 0; i < chunks.length; i++) {
      if (cancelled) throw new Error('cancelled');
      post('progress', { done: done++, total });
      pieces.push(await speak(q, chunks[i], lang, voice));
      if (i < chunks.length - 1) pieces.push(silence(pauses.sentence));
    }
    if (t < plan.length - 1) pieces.push(silence(pauses.turn ?? 0.6));
  }
  const audio = new Float32Array(pieces.reduce((a, p) => a + p.length, 0));
  let off = 0; for (const p of pieces) { audio.set(p, off); off += p.length; }
  post('done', { audio, sampleRate: SAMPLE_RATE, chunks: total, stats: q.stats });
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'synthesize') { if (data.ep) Object.assign(EP, data.ep); await synthesize(data); }
    else if (data.type === 'status') { caches.delete('giong-noi-models-v1').catch(() => {}); post('cache', { engines: { qwen3: { ...(await cacheStatus()), webgpu: await hasWebGPU() } } }); }   // the old engines' cache is no longer used
    else if (data.type === 'clear') { await caches.delete(CACHE); tts = null; post('cleared'); }
    else if (data.type === 'cancel') cancelled = true;
  } catch (err) {
    post(err.message === 'cancelled' ? 'cancelled' : 'error', { message: String(err.message || err) });
  }
};
