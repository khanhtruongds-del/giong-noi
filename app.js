// UI: settings, text, job control, WAV export. Heavy work happens in worker.js.
const $ = (id) => document.getElementById(id);
const cfg = await (await fetch('config.json')).json();
const manifest = await (await fetch('models/manifest.json')).json();
const ORDER = ['de', 'en', 'fr', 'zh', 'ja', 'ko'];
const store = { get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
                set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } } };

let lang = ORDER.includes(store.get('lang')) ? store.get('lang') : 'de';
let gender = store.get('gender') === 'male' ? 'male' : 'female';
let placeholder = true, busy = false, cache = {}, lastUrl = null;
let worker = null;

// ---- header waveform ------------------------------------------------------------------------
const BARS = [0.35, 0.6, 0.9, 0.55, 0.75, 0.4];
const wave = $('wave');
const bars = [];
function buildWave() {
  const mk = (i, strong) => { const s = document.createElement('span'); if (strong) s.className = 's'; s.style.height = BARS[i] * 52 + 'px'; bars.push([s, BARS[i]]); return s; };
  BARS.forEach((_, i) => wave.append(mk(i, i >= 4)));
  const mic = document.createElement('div'); mic.className = 'mic'; mic.innerHTML = '<svg class="i"><use href="#i-mic"/></svg>';
  wave.append(mic);
  [...BARS.keys()].reverse().forEach(i => wave.append(mk(i, i >= 4)));
}
let anim = null;
function animate(on) {
  cancelAnimationFrame(anim);
  if (!on) { bars.forEach(([s, b]) => (s.style.height = b * 52 + 'px')); return; }
  const t0 = performance.now();
  const tick = (t) => { bars.forEach(([s, b], i) => (s.style.height = b * 52 * (0.55 + 0.45 * Math.abs(Math.sin((t - t0) / 220 + i * 0.9))) + 'px')); anim = requestAnimationFrame(tick); };
  anim = requestAnimationFrame(tick);
}
buildWave();

// ---- settings --------------------------------------------------------------------------------
for (const code of ORDER) {
  const b = document.createElement('button');
  b.className = 'pill'; b.textContent = cfg.languages[code].name; b.dataset.lang = code;
  b.onclick = () => { lang = code; store.set('lang', code); refresh(); };
  $('langs').append(b);
}
document.querySelectorAll('#genders .pill').forEach(b => (b.onclick = () => { gender = b.dataset.g; store.set('gender', gender); refresh(); }));
for (const q of Object.keys(cfg.quality)) $('quality').append(new Option(q, q));
$('quality').value = cfg.quality[store.get('quality')] ? store.get('quality') : 'Cân bằng';
$('quality').onchange = () => store.set('quality', $('quality').value);
for (const sp of Object.keys(cfg.speed)) $('speed').append(new Option(sp, sp));
$('speed').value = cfg.speed[store.get('speed')] ? store.get('speed') : cfg.default_speed;
$('speed').onchange = () => store.set('speed', $('speed').value);

const mb = (b) => (b / 1e6).toFixed(0) + ' MB';
function refresh() {
  document.querySelectorAll('#langs .pill').forEach(b => b.setAttribute('aria-pressed', b.dataset.lang === lang));
  document.querySelectorAll('#genders .pill').forEach(b => b.setAttribute('aria-pressed', b.dataset.g === gender));
  const L = cfg.languages[lang];
  $('quality').disabled = L.engine === 'kokoro-82m';
  $('quality').title = L.engine === 'kokoro-82m' ? 'Giọng tiếng Trung không có tuỳ chọn chất lượng' : '';
  const st = cache[L.engine];
  const hint = $('hint');
  if (st && st.complete) {
    hint.className = 'hint ok';
    $('hintText').textContent = `Giọng ${gender === 'male' ? 'nam' : 'nữ'} · ${L.full} · sẵn sàng, dùng được khi không có mạng`;
  } else {
    hint.className = 'hint';
    $('hintText').textContent = `Lần đầu dùng ${L.full}: trình duyệt sẽ tải khoảng ${mb(manifest.engines[L.engine].bytes)} dữ liệu giọng rồi giữ lại trên máy.`;
  }
  if (placeholder) setPlaceholder();
  $('go').disabled = busy;
}

// ---- text --------------------------------------------------------------------------------------
const text = $('text');
function setPlaceholder() { text.value = ''; text.placeholder = `Nhập hoặc dán văn bản vào đây, ví dụ:\n${cfg.languages[lang].sample}`; placeholder = true; count(); }
function count() { const n = text.value.length; $('count').textContent = n ? n.toLocaleString('vi-VN') + ' ký tự' : ''; }
text.addEventListener('input', () => { placeholder = !text.value; count(); });
$('clear').onclick = () => { text.value = ''; setPlaceholder(); text.focus(); };
$('open').onclick = () => $('file').click();
$('file').onchange = async () => {
  const f = $('file').files[0]; $('file').value = '';
  if (!f) return;
  if (f.size > 1_000_000) return setStatus('File quá lớn. Chọn file dưới 1 MB.', true);
  const t = (await f.text()).replace(/^﻿/, '');
  if (t.length > 100_000) return setStatus('Mỗi lần tối đa 100.000 ký tự.', true);
  if (text.value.trim() && !confirm('Thay nội dung đang nhập bằng nội dung file?')) return;
  text.value = t; placeholder = !t; count();
};

// ---- job ---------------------------------------------------------------------------------------
const bar = $('bar');
function setStatus(msg, warn = false) { $('status').textContent = msg; $('status').className = warn ? 'warn' : ''; }
function progress(frac) { bar.hidden = false; bar.classList.toggle('indet', frac == null); bar.firstElementChild.style.width = frac == null ? '' : Math.round(frac * 100) + '%'; }

function newWorker() {
  worker = new Worker(new URL('worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = ({ data }) => handlers[data.type]?.(data);
  worker.onerror = (e) => { finish(); setStatus('Lỗi khi nạp phần đọc giọng: ' + (e.message || 'không rõ'), true); };
  worker.postMessage({ type: 'status' });
}
const handlers = {
  status: (d) => setStatus(d.message),
  download: (d) => { setStatus(`Đang tải dữ liệu giọng (chỉ lần đầu)… ${mb(d.got)} / ${mb(d.total)}`); progress(d.got / d.total); },
  progress: (d) => { setStatus(`Đang đọc đoạn ${d.done + 1}/${d.total}…`); progress(d.done / d.total); },
  done: (d) => { bar.hidden = true; setStatus('Xong.'); showResult(d.audio, d.sampleRate); finish(); worker.postMessage({ type: 'status' }); },
  error: (d) => { finish(); bar.hidden = true; setStatus(d.message, true); },
  cancelled: () => { finish(); bar.hidden = true; setStatus('Đã hủy.'); },
  cache: (d) => { cache = d.engines; refresh(); renderData(); },
  cleared: () => { $('dataMsg').textContent = 'Đã xoá.'; worker.postMessage({ type: 'status' }); },
};
function finish() { busy = false; $('cancel').disabled = true; animate(false); refresh(); }

$('go').onclick = () => {
  const content = text.value.trim();
  if (!content) { setStatus('Nhập hoặc mở file văn bản trước.', true); text.focus(); return; }
  const L = cfg.languages[lang];
  busy = true; refresh(); $('cancel').disabled = false; animate(true);
  $('result').classList.remove('show'); setStatus('Đang chuẩn bị…'); progress(null);
  worker.postMessage({ type: 'synthesize', engine: L.engine, lang, voice: L[gender], speed: cfg.speed[$('speed').value] || 1,
                       steps: cfg.quality[$('quality').value] || 8, text: content });
};
$('cancel').onclick = () => {
  // Terminating is the only way to stop a running ONNX call; the next job starts a fresh worker (models stay cached on disk).
  worker.terminate(); newWorker(); finish(); bar.hidden = true; setStatus('Đã hủy.');
};

// ---- result / WAV ----------------------------------------------------------------------------
function wav(samples, sr) {
  const buf = new ArrayBuffer(44 + samples.length * 2), v = new DataView(buf);
  const w = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  w(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); w(8, 'WAVE'); w(12, 'fmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, sr, true);
  v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, 'data'); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, samples[i])) * 32767, true);
  return new Blob([buf], { type: 'audio/wav' });
}
function filename() {
  const words = (text.value.match(/[\p{L}\p{N}]+/gu) || []).slice(0, 5).join('-').slice(0, 40) || 'giong-noi';
  return `${lang}_${gender === 'male' ? 'nam' : 'nu'}_${words}.wav`;
}
function showResult(samples, sr) {
  const blob = wav(samples, sr);
  if (lastUrl) URL.revokeObjectURL(lastUrl);
  lastUrl = URL.createObjectURL(blob);
  const secs = Math.round(samples.length / sr);
  $('player').src = lastUrl; $('save').href = lastUrl; $('save').download = filename();
  $('rname').textContent = filename();
  $('rmeta').textContent = `WAV · ${(blob.size / 1e6).toFixed(1)} MB · ${String(Math.floor(secs / 60)).padStart(2, '0')}:${String(secs % 60).padStart(2, '0')}`;
  $('result').classList.add('show'); $('playLabel').textContent = 'Nghe';
}
$('play').onclick = () => { const a = $('player'); if (a.paused) { a.currentTime = a.ended ? 0 : a.currentTime; a.play(); } else a.pause(); };
$('player').onplay = () => ($('playLabel').textContent = 'Dừng');
$('player').onpause = $('player').onended = () => ($('playLabel').textContent = 'Nghe');

// ---- data dialog -----------------------------------------------------------------------------
const NAMES = { 'supertonic-3': 'Đức · Anh · Pháp · Nhật · Hàn (Supertonic 3)', 'kokoro-82m': 'Trung (Kokoro-82M)' };
function renderData() {
  const list = $('engList'); list.textContent = '';
  for (const [name, e] of Object.entries(manifest.engines)) {
    const st = cache[name] || {};
    const row = document.createElement('div'); row.className = 'eng';
    row.innerHTML = `<div class="grow"><b></b><div class="sub"></div></div><span class="chip"></span>`;
    row.querySelector('b').textContent = NAMES[name] || name;
    row.querySelector('.sub').textContent = mb(e.bytes);
    const chip = row.querySelector('.chip');
    chip.textContent = st.complete ? 'Đã tải' : st.parts ? `Đã tải ${st.parts}/${st.total} phần` : 'Chưa tải';
    if (st.complete) chip.classList.add('ok');
    list.append(row);
  }
}
$('data').onclick = () => { $('dataMsg').textContent = ''; renderData(); $('dataDlg').showModal(); };
$('closeDlg').onclick = () => $('dataDlg').close();
$('wipe').onclick = () => { if (confirm('Xoá dữ liệu giọng đã tải? Lần sau dùng sẽ phải tải lại.')) worker.postMessage({ type: 'clear' }); };

newWorker();
setPlaceholder();
refresh();
setStatus('');
