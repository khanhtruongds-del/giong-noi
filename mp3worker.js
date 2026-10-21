// MP3 encoder worker (classic worker so lamejs can load with importScripts). lamejs: LGPL-3.0.
importScripts('vendor/lame.min.js');
self.onmessage = ({ data: { samples, sampleRate, kbps = 128 } }) => {
  try {
    const enc = new lamejs.Mp3Encoder(1, sampleRate, kbps);
    const pcm = new Int16Array(samples.length);
    for (let i = 0; i < samples.length; i++) { const s = Math.max(-1, Math.min(1, samples[i])); pcm[i] = s < 0 ? s * 0x8000 : s * 0x7fff; }
    const out = [], step = 1152 * 20;
    for (let i = 0; i < pcm.length; i += step) {
      const b = enc.encodeBuffer(pcm.subarray(i, i + step)); if (b.length) out.push(new Uint8Array(b));
      if ((i / step) % 20 === 0) self.postMessage({ type: 'progress', done: i, total: pcm.length });
    }
    const end = enc.flush(); if (end.length) out.push(new Uint8Array(end));
    self.postMessage({ type: 'done', blob: new Blob(out, { type: 'audio/mpeg' }) });
  } catch (e) { self.postMessage({ type: 'error', message: String(e && e.message || e) }); }
};
