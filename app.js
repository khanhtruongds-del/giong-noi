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
  // Kokoro (Chinese) has no quality steps: hide the control instead of showing a dead dropdown.
  const kokoro = L.engine.startsWith('kokoro');
  $('qualityBox').hidden = kokoro;
  // Languages with a "choices" list (Chinese, while teachers pick) let the user try every voice of that gender.
  const pool = (L.choices || {})[gender];
  $('voiceBox').hidden = !pool;
  if (pool) {
    const sel = $('voice'), saved = store.get(`voice_${lang}_${gender}`);
    sel.textContent = '';
    pool.forEach((id, i) => sel.append(new Option(`${gender === 'male' ? 'Nam' : 'Nữ'} ${id.slice(3)}${id === L[gender] ? ' (mặc định)' : ''}`, id)));
    sel.value = pool.includes(saved) ? saved : L[gender];
    sel.onchange = () => store.set(`voice_${lang}_${gender}`, sel.value);
  }
  $('opts').classList.toggle('one', kokoro && !pool);
  const st = cache[L.engine];
  const hint = $('hint');
  const who = `${L.full} · giọng ${gender === 'male' ? 'nam' : 'nữ'}`;
  if (st && st.complete) {
    hint.className = 'hint ok';
    $('hintIcon').innerHTML = '<use href="#i-check"/>';
    $('hintText').textContent = `${who} đã sẵn sàng, dùng được cả khi không có mạng.`;
  } else {
    hint.className = 'hint';
    $('hintIcon').innerHTML = '<use href="#i-download"/>';
    const shared = kokoro ? '' : ' (dùng chung cho Đức, Anh, Pháp, Nhật, Hàn)';
    $('hintText').textContent = `Lần đầu dùng ${L.full}, trang sẽ tải khoảng ${mb(manifest.engines[L.engine].bytes)} dữ liệu giọng${shared}. Chỉ tải một lần, sau đó dùng được cả khi không có mạng.`;
  }
  if (placeholder) setPlaceholder();
  $('go').disabled = busy;
  $('goLabel').textContent = busy ? 'Đang tạo…' : 'Tạo giọng đọc';
  $('cancel').hidden = !busy;
  if (!busy && !$('status').textContent) idle();
}
const MAC = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
function idle() { setStatus('Nhập văn bản rồi bấm “Tạo giọng đọc”.'); }
$('kbd').innerHTML = `Mẹo: <kbd>${MAC ? '⌘' : 'Ctrl'}</kbd> + <kbd>Enter</kbd> để tạo nhanh`;

// ---- text --------------------------------------------------------------------------------------
const text = $('text');
function setPlaceholder() { text.value = ''; text.placeholder = `Nhập hoặc dán văn bản vào đây, ví dụ:\n${cfg.languages[lang].sample}`; placeholder = true; count(); }
function count() { const n = text.value.length; $('count').textContent = n ? `${n.toLocaleString('vi-VN')} / 100.000 ký tự` : ''; }
text.addEventListener('input', () => { placeholder = !text.value; count(); });
text.addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); if (!busy) $('go').click(); } });
$('clear').onclick = () => { text.value = ''; setPlaceholder(); text.focus(); };
$('open').onclick = () => $('file').click();
$('file').onchange = async () => {
  const f = $('file').files[0]; $('file').value = '';
  if (!f) return;
  if (f.size > 1_000_000) return setStatus('Tệp lớn hơn 1 MB. Hãy chọn tệp nhỏ hơn.', true);
  const t = (await f.text()).replace(/^﻿/, '');
  if (t.length > 100_000) return setStatus('Văn bản dài hơn 100.000 ký tự. Hãy chia thành nhiều lần.', true);
  if (text.value.trim() && !confirm('Thay văn bản đang có bằng nội dung tệp?')) return;
  text.value = t; placeholder = !t; count();
};

// ---- job ---------------------------------------------------------------------------------------
const bar = $('bar');
function setStatus(msg, warn = false) { $('status').textContent = msg; $('status').className = warn ? 'warn' : ''; }
function progress(frac) { bar.hidden = false; bar.classList.toggle('indet', frac == null); bar.firstElementChild.style.width = frac == null ? '' : Math.round(frac * 100) + '%'; }

function newWorker() {
  worker = new Worker(new URL('worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = ({ data }) => handlers[data.type]?.(data);
  worker.onerror = (e) => { finish(); bar.hidden = true; setStatus('Không chạy được phần đọc giọng trên trình duyệt này. Hãy thử Chrome, Edge hoặc Safari bản mới. (' + (e.message || 'không rõ lỗi') + ')', true); };
  worker.postMessage({ type: 'status' });
}
const handlers = {
  status: (d) => setStatus(d.message),
  download: (d) => { setStatus(`Đang tải dữ liệu giọng lần đầu · ${mb(d.got)} / ${mb(d.total)} (${Math.round(100 * d.got / d.total)}%)`); progress(d.got / d.total); },
  progress: (d) => { setStatus(d.total > 1 ? `Đang tạo giọng đọc… đoạn ${d.done + 1}/${d.total}` : 'Đang tạo giọng đọc…'); progress(d.total > 1 ? d.done / d.total : null); },
  done: (d) => { bar.hidden = true; showResult(d.audio, d.sampleRate); setStatus(`Đã xong · ${clock(d.audio.length / d.sampleRate)} âm thanh. Bấm “Nghe” hoặc tải file về.`); finish(); worker.postMessage({ type: 'status' }); },
  error: (d) => { finish(); bar.hidden = true; setStatus(d.message, true); },
  cancelled: () => { finish(); bar.hidden = true; setStatus('Đã huỷ.'); },
  cache: (d) => { cache = d.engines; refresh(); renderData(); },
  cleared: () => { $('dataMsg').textContent = 'Đã xoá dữ liệu giọng.'; worker.postMessage({ type: 'status' }); },
};
function finish() { busy = false; animate(false); refresh(); }
const clock = (secs) => { secs = Math.round(secs); return `${String(Math.floor(secs / 60)).padStart(2, '0')}:${String(secs % 60).padStart(2, '0')}`; };

$('go').onclick = () => {
  const content = text.value.trim();
  if (!content) { setStatus('Chưa có văn bản để đọc. Hãy nhập, dán hoặc mở tệp .txt.', true); text.focus(); return; }
  const L = cfg.languages[lang];
  busy = true; animate(true);
  $('result').classList.remove('show'); setStatus('Đang chuẩn bị…'); progress(null); refresh();
  worker.postMessage({ type: 'synthesize', engine: L.engine, lang, voice: (L.choices || {})[gender] ? $('voice').value : L[gender], speed: cfg.speed[$('speed').value] || 1,
                       steps: cfg.quality[$('quality').value] || 8, text: content,
                       commaBreak: (cfg.pause_at_comma || []).includes(lang), pauses: cfg.pauses });
};
$('cancel').onclick = () => {
  // Terminating is the only way to stop a running ONNX call; the next job starts a fresh worker (models stay cached on disk).
  worker.terminate(); newWorker(); finish(); bar.hidden = true; setStatus('Đã huỷ.');
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
  const L = cfg.languages[lang], who = (L.choices || {})[gender] ? $('voice').value : (gender === 'male' ? 'nam' : 'nu');
  return `${lang}_${who}_${words}.wav`;
}
function showResult(samples, sr) {
  const blob = wav(samples, sr);
  if (lastUrl) URL.revokeObjectURL(lastUrl);
  lastUrl = URL.createObjectURL(blob);
  const secs = Math.round(samples.length / sr);
  $('player').src = lastUrl; $('save').href = lastUrl; $('save').download = filename();
  $('rname').textContent = filename();
  $('rmeta').textContent = `WAV · ${(blob.size / 1e6).toFixed(1)} MB · ${clock(secs)}`;
  $('result').classList.add('show'); $('playLabel').textContent = 'Nghe';
}
$('play').onclick = () => { const a = $('player'); if (a.paused) { a.currentTime = a.ended ? 0 : a.currentTime; a.play(); } else a.pause(); };
$('player').onplay = () => ($('playLabel').textContent = 'Dừng');
$('player').onpause = $('player').onended = () => ($('playLabel').textContent = 'Nghe');

// ---- data dialog -----------------------------------------------------------------------------
const NAMES = { 'supertonic-3': 'Đức · Anh · Pháp · Nhật · Hàn (Supertonic 3)', 'kokoro-82m': 'Trung (Kokoro-82M)', 'kokoro-zh': 'Trung (Kokoro-82M v1.1-zh)' };
function renderData() {
  const list = $('engList'); list.textContent = '';
  for (const [name, e] of Object.entries(manifest.engines)) {
    const st = cache[name] || {};
    const row = document.createElement('div'); row.className = 'eng';
    row.innerHTML = `<div class="grow"><b></b><div class="sub"></div></div><span class="chip"></span>`;
    row.querySelector('b').textContent = NAMES[name] || name;
    row.querySelector('.sub').textContent = mb(e.bytes);
    const chip = row.querySelector('.chip');
    chip.textContent = st.complete ? 'Đã lưu trên máy' : st.parts ? `Tải dở (${st.parts}/${st.total} phần)` : 'Chưa tải';
    if (st.complete) chip.classList.add('ok');
    list.append(row);
  }
}
$('data').onclick = () => { $('dataMsg').textContent = ''; renderData(); $('dataDlg').showModal(); };
$('closeDlg').onclick = () => $('dataDlg').close();
$('wipe').onclick = () => { if (confirm('Xoá dữ liệu giọng đã lưu trên máy này? Lần dùng sau sẽ phải tải lại.')) worker.postMessage({ type: 'clear' }); };

newWorker();
setPlaceholder();
refresh();
idle();
