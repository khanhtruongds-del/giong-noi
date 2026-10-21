// Qwen3-TTS 12Hz 0.6B Base (voice-clone personas) on ONNX Runtime Web.
// Port of qwen_tts Qwen3TTSForConditionalGeneration.generate (Apache-2.0, Alibaba Qwen team).
// Parts: talker (28-layer LM with KV cache), code predictor (5 layers, 15 code groups per frame),
// 12 Hz speech tokenizer decoder (16 code groups -> 24 kHz audio). Embedding tables are plain arrays.
const D = 1024, KVH = 8, HD = 128, LAYERS = 28, GROUPS = 16, CP_VOCAB = 2048;
const CFG = {
  im_start: 151644, assistant: 77091, tts_bos: 151672, tts_eos: 151673, tts_pad: 151671,
  codec_bos: 2149, codec_eos: 2150, codec_pad: 2148, think: 2154, nothink: 2155, think_bos: 2156, think_eos: 2157,
  vocab: 3072,
  lang: { chinese: 2055, english: 2050, german: 2053, spanish: 2054, japanese: 2058, korean: 2064, french: 2061, russian: 2069, italian: 2070, portuguese: 2071 },
};
export const SAMPLE_RATE = 24000, FRAME_SAMPLES = 1920;
const F16 = (() => {   // float16 bit pattern -> float32
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    t[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return t;
})();

export class QwenTTS {
  // tables: { text: {q: Int8Array, s: Float32Array}, codec: Float32Array, cpemb: {q, s} }
  // sessions: { talker, cp, dec } created by the caller (so it can pick WASM or WebGPU)
  constructor(ort, sessions, tables, tokenizer) { Object.assign(this, { ort, ...sessions, tables, tokenizer, stats: { talker: 0, tn: 0, cp: 0, cn: 0 } }); }

  // ---- embeddings --------------------------------------------------------------------------------
  rowI8(t, row, out, add = false, off = 0) {
    const s = t.s[row], base = row * D;
    if (add) for (let i = 0; i < D; i++) out[off + i] += t.q[base + i] * s;
    else for (let i = 0; i < D; i++) out[off + i] = t.q[base + i] * s;
  }
  textEmb(id, out, add = false, off = 0) {
    const t = this.tables.text;
    if (!t.f16) return this.rowI8(t, id, out, add, off);
    const L = F16, b = id * D;                        // float16 table, decoded through a lookup table
    if (add) for (let i = 0; i < D; i++) out[off + i] += L[t.f16[b + i]]; else for (let i = 0; i < D; i++) out[off + i] = L[t.f16[b + i]];
  }
  codecEmb(id, out, add = false, off = 0) {
    const c = this.tables.codec, b = id * D;
    if (add) for (let i = 0; i < D; i++) out[off + i] += c[b + i]; else for (let i = 0; i < D; i++) out[off + i] = c[b + i];
  }
  cpEmb(group, id, out, add = false, off = 0) { this.rowI8(this.tables.cpemb, group * CP_VOCAB + id, out, add, off); }

  // ---- prompt (generate(): role + codec tags + speaker + text / ICL reference) --------------------
  buildPrompt(text, lang, persona, icl) {
    const enc = (s) => this.tokenizer.encode(s, { add_special_tokens: false }).ids;
    const ids = enc(`<|im_start|>assistant\n${text}<|im_end|>\n<|im_start|>assistant\n`);
    const rows = [];                 // each row: Float32Array(D)
    const vec = () => new Float32Array(D);
    const T = (id) => { const v = vec(); this.textEmb(id, v); return v; };
    const C = (id) => { const v = vec(); this.codecEmb(id, v); return v; };
    const add = (a, b) => { for (let i = 0; i < D; i++) a[i] += b[i]; return a; };
    const pad = T(CFG.tts_pad), bos = T(CFG.tts_bos), eos = T(CFG.tts_eos);
    for (const id of ids.slice(0, 3)) rows.push(T(id));                      // <|im_start|>assistant\n
    const lid = CFG.lang[lang.toLowerCase()];
    const codec = (lid == null ? [CFG.nothink, CFG.think_bos, CFG.think_eos] : [CFG.think, CFG.think_bos, lid, CFG.think_eos]).map(C);
    codec.push(Float32Array.from(persona.spk_emb), C(CFG.codec_pad), C(CFG.codec_bos));
    for (let i = 0; i < codec.length - 1; i++) rows.push(add(Float32Array.from(i < codec.length - 2 ? pad : bos), codec[i]));
    const textIds = ids.slice(3, -5);
    let trailing;
    if (icl) {
      const refIds = enc(`<|im_start|>assistant\n${persona.ref_text}<|im_end|>\n`).slice(3, -2);
      const tEmb = [...refIds, ...textIds].map(T); tEmb.push(eos);
      const cEmb = [C(CFG.codec_bos)];
      for (const frame of persona.ref_code) {
        const v = vec(); this.codecEmb(frame[0], v);
        for (let g = 1; g < GROUPS; g++) this.cpEmb(g - 1, frame[g], v, true);
        cEmb.push(v);
      }
      if (tEmb.length > cEmb.length) { cEmb.forEach((c, i) => rows.push(add(tEmb[i], c))); trailing = tEmb.slice(cEmb.length); }
      else { cEmb.forEach((c, i) => rows.push(add(Float32Array.from(i < tEmb.length ? tEmb[i] : pad), c))); trailing = []; }
    } else {
      rows.push(add(T(textIds[0]), codec[codec.length - 1]));
      trailing = [...textIds.slice(1).map(T), eos];
    }
    return { rows, trailing, pad };
  }

  // ---- sampling (HF processors: repetition penalty, suppress, min length, temperature, top-k, top-p) --
  sample(logits, { greedy = false, temperature = 0.9, topK = 50, topP = 1.0, penalty = 1.0, history = null, suppress = null, rng = Math.random }) {
    const l = Float32Array.from(logits);
    if (history && penalty !== 1) for (const t of new Set(history)) l[t] = l[t] < 0 ? l[t] * penalty : l[t] / penalty;
    if (suppress) suppress(l);
    if (greedy) { let b = 0; for (let i = 1; i < l.length; i++) if (l[i] > l[b]) b = i; return b; }
    for (let i = 0; i < l.length; i++) l[i] /= temperature;
    const idx = Array.from(l.keys()).filter(i => l[i] > -Infinity).sort((a, b) => l[b] - l[a]).slice(0, topK);
    const m = l[idx[0]]; let p = idx.map(i => Math.exp(l[i] - m)); const z = p.reduce((a, b) => a + b, 0); p = p.map(x => x / z);
    if (topP < 1) { let c = 0, n = 0; while (n < p.length && c < topP) c += p[n++]; p = p.slice(0, n); }
    let r = rng() * p.reduce((a, b) => a + b, 0);
    for (let i = 0; i < p.length; i++) { r -= p[i]; if (r <= 0) return idx[i]; }
    return idx[p.length - 1];
  }

  // ---- talker step (decode graph also used for prefill with an empty cache) ------------------------
  async talkerRun(rows, past, pastLen) {
    const ort = this.ort, n = rows.length, x = new Float32Array(n * D);
    rows.forEach((r, i) => x.set(r, i * D));
    const feeds = { inputs_embeds: new ort.Tensor('float32', x, [1, n, D]),
      attention_mask: new ort.Tensor('int64', new BigInt64Array(pastLen + n).fill(1n), [1, pastLen + n]) };
    for (let i = 0; i < LAYERS; i++) {
      feeds[`past_key_${i}`] = past ? past[2 * i] : new ort.Tensor('float32', new Float32Array(0), [1, KVH, 0, HD]);
      feeds[`past_value_${i}`] = past ? past[2 * i + 1] : new ort.Tensor('float32', new Float32Array(0), [1, KVH, 0, HD]);
    }
    const t0 = performance.now(); const out = await this.talker.run(feeds); this.stats.talker += performance.now() - t0; this.stats.tn++;
    const kv = []; for (let i = 0; i < LAYERS; i++) kv.push(out[`present_key_${i}`], out[`present_value_${i}`]);
    if (past) for (const t of past) t.dispose?.();
    const lg = out.logits.data, V = CFG.vocab;
    return { logits: lg.subarray(lg.length - V), hidden: Float32Array.from(out.last_hidden.data), past: kv };
  }

  async cpRun(rows, step) {
    const ort = this.ort, x = new Float32Array(rows.length * D);
    rows.forEach((r, i) => x.set(r, i * D));
    const t0 = performance.now(); const out = await this.cp.run({ inputs_embeds: new ort.Tensor('float32', x, [1, rows.length, D]),
      generation_step: new ort.Tensor('int64', BigInt64Array.from([BigInt(step)]), [1]) });
    this.stats.cp += performance.now() - t0; this.stats.cn++;
    return out.logits.data;
  }

  // ---- full generation ---------------------------------------------------------------------------
  async generate(text, lang, persona, { icl = false, greedy = false, maxFrames = 600, rng = Math.random, onFrame = null } = {}) {
    const { rows, trailing, pad } = this.buildPrompt(text, lang, persona, icl);
    const opts = { greedy, rng };
    const suppress = (minDone) => (l) => { for (let i = CFG.vocab - 1024; i < CFG.vocab; i++) if (i !== CFG.codec_eos) l[i] = -Infinity; if (!minDone) l[CFG.codec_eos] = -Infinity; };
    let r = await this.talkerRun(rows, null, 0), pastLen = rows.length;
    const frames = [], history = [];
    let tok = this.sample(r.logits, { ...opts, penalty: 1.05, history, suppress: suppress(false) });
    for (let step = 0; step < maxFrames && tok !== CFG.codec_eos; step++) {
      history.push(tok);
      // code predictor: 15 more code groups for this frame
      const last = new Float32Array(D); this.codecEmb(tok, last);
      let codes = [tok], sum;
      if (this.cpf) {          // fused graph: all 15 groups in one run, sampled in-graph (Gumbel-max over top-k)
        const ort = this.ort, gum = new Float32Array(15 * CP_VOCAB);
        if (!greedy) for (let i = 0; i < gum.length; i++) gum[i] = -Math.log(-Math.log(Math.max(rng(), 1e-12)));
        const t0 = performance.now();
        const o = await this.cpf.run({ past_hidden: new ort.Tensor('float32', r.hidden, [1, 1, D]), first_embed: new ort.Tensor('float32', last, [1, 1, D]),
          gumbel: new ort.Tensor('float32', gum, [15, CP_VOCAB]), inv_temp: new ort.Tensor('float32', Float32Array.from([1 / 0.9]), [1]) });
        this.stats.cp += performance.now() - t0; this.stats.cn++;
        codes = codes.concat(Array.from(o.codes.data, Number)); sum = Float32Array.from(o.sum_embed.data);
      } else {
        const seq = [r.hidden, last]; sum = Float32Array.from(last);
        for (let g = 0; g < GROUPS - 1; g++) {
          const c = this.sample(await this.cpRun(seq, g), { ...opts });
          codes.push(c);
          const e = new Float32Array(D); this.cpEmb(g, c, e); seq.push(e);
          for (let i = 0; i < D; i++) sum[i] += e[i];
        }
      }
      frames.push(codes); onFrame?.(frames.length);
      const tt = step < trailing.length ? trailing[step] : pad;
      for (let i = 0; i < D; i++) sum[i] += tt[i];
      r = await this.talkerRun([sum], r.past, pastLen); pastLen += 1;
      tok = this.sample(r.logits, { ...opts, penalty: 1.05, history, suppress: suppress(history.length >= 2) });
    }
    for (const t of r.past) t.dispose?.();
    return frames;
  }

  // The codec decoder graph was exported for at most 100 frames per call, so long chunks are decoded in
  // windows: each window re-reads CTX frames of left context and keeps only the audio of its new frames.
  async decodeWindow(frames) {
    const ort = this.ort, T = frames.length, codes = new BigInt64Array(GROUPS * T);
    for (let t = 0; t < T; t++) for (let g = 0; g < GROUPS; g++) codes[t * GROUPS + g] = BigInt(frames[t][g]);   // [1, T, 16]
    const out = await this.dec.run({ audio_codes: new ort.Tensor('int64', codes, [1, T, GROUPS]) });
    let wav = out.audio_values.data;
    if (out.lengths) wav = wav.subarray(0, Number(out.lengths.data[0]));
    return wav;
  }
  async decode(frames, persona = null, icl = false, { win = 100, ctx = 25 } = {}) {
    const all = icl && persona ? [...persona.ref_code, ...frames] : frames, T = all.length;
    let wav;
    if (T <= win) wav = Float32Array.from(await this.decodeWindow(all));
    else {
      const parts = [];
      const step = win - ctx;
      for (let s = 0; s < T; s += step) {
        const a = Math.max(0, s - ctx), b = Math.min(T, s + step), w = await this.decodeWindow(all.slice(a, b));
        parts.push(w.slice(Math.round(w.length * (s - a) / (b - a))));
      }
      wav = new Float32Array(parts.reduce((n, p) => n + p.length, 0)); let o = 0; for (const p of parts) { wav.set(p, o); o += p.length; }
    }
    if (icl && persona) wav = wav.subarray(Math.floor(wav.length * persona.ref_code.length / T));
    return Float32Array.from(wav);
  }
}
