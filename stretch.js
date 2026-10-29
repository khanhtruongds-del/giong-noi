// Change speaking speed without changing pitch (WSOLA). rate > 1 = faster, < 1 = slower.
// 32 ms Hann frames at 50 % overlap; each next frame is taken within ±8 ms of its ideal place where it best continues the previous one.
export function stretch(x, rate, sr = 24000) {
  if (!rate || Math.abs(rate - 1) < 1e-3 || x.length < 2) return x;
  const L = Math.round(sr * 0.032) & ~1, H = L >> 1, tol = Math.round(sr * 0.008);
  const n = Math.ceil(x.length / rate), out = new Float32Array(n + L), wsum = new Float32Array(n + L);
  const win = new Float32Array(L);
  for (let i = 0; i < L; i++) win[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / L);
  const at = (i) => (i >= 0 && i < x.length ? x[i] : 0);
  let prev = 0;
  for (let k = 0, o = 0; o < n; k++, o += H) {
    const ideal = Math.round(k * H * rate);
    let best = ideal;
    if (k > 0) {
      // compare the first half of each candidate with what naturally follows the previous frame
      const nat = prev + H;
      let bestScore = -Infinity;
      for (let d = -tol; d <= tol; d += 2) {
        const c = ideal + d;
        let s = 0;
        for (let i = 0; i < H; i += 4) s += at(c + i) * at(nat + i);
        if (s > bestScore) { bestScore = s; best = c; }
      }
    }
    for (let i = 0; i < L; i++) { out[o + i] += at(best + i) * win[i]; wsum[o + i] += win[i]; }
    prev = best;
  }
  const y = new Float32Array(n);
  for (let i = 0; i < n; i++) y[i] = wsum[i] > 1e-3 ? out[i] / wsum[i] : 0;
  return y;
}
