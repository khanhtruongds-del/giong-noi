// UI: settings, text, job control, WAV export. Heavy work happens in worker.js.
const $ = (id) => document.getElementById(id);
const cfg = await (await fetch('config.json', { cache: 'no-cache' })).json();
const ORDER = ['de', 'en', 'fr', 'zh', 'ja', 'ko'];
// Qwen3-TTS runs on the graphics card through WebGPU; browsers without it cannot use the page.
const GPU = await (async () => { try { return !!(navigator.gpu && await navigator.gpu.requestAdapter()); } catch { return false; } })();
const NO_GPU = 'Trình duyệt này chưa có WebGPU nên không chạy được giọng đọc. Hãy mở trang bằng Chrome hoặc Edge bản mới trên máy tính.';
const store = { get(k, d) { try { return localStorage.getItem(k) ?? d; } catch { return d; } },
                set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } } };

let lang = ORDER.includes(store.get('lang')) ? store.get('lang') : 'de';
let gender = store.get('gender') === 'male' ? 'male' : 'female';
let placeholder = true, busy = false, cache = {}, lastUrl = null, mp3Url = null, lastDialogue = false, mp3w = null;
let worker = null;
let dialogue = store.get('dialogue') === '1';
let sylMode = false;   // syllable drill was Kokoro-only; removed with the old engines
let lastSyl = false;
const spkVoices = {};   // lang -> { speakerName: voiceId } chosen by the user this session

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
  $('qualityBox').hidden = true;
  $('speedBox').hidden = true;   // Qwen speaks at its own natural pace; there is no speed control in the model
  // Languages with a "choices" list (Chinese, while teachers pick) let the user try every voice of that gender.
  const pool = (L.choices || {})[gender];
  $('voiceBox').hidden = !pool;
  if (pool) {
    const sel = $('voice'), saved = store.get(`voice_${lang}_${gender}`);
    sel.textContent = '';
    pool.forEach((id, i) => sel.append(new Option(`${(cfg.voice_labels || {})[id] || (gender === 'male' ? 'Nam ' : 'Nữ ') + id.slice(3)}${id === L[gender] ? ' (mặc định)' : ''}`, id)));
    sel.value = pool.includes(saved) ? saved : L[gender];
    sel.onchange = () => store.set(`voice_${lang}_${gender}`, sel.value);
  }
  $('opts').classList.toggle('one', true);
  const st = cache.qwen3;
  const hint = $('hint');
  const who = `${L.full} · giọng ${gender === 'male' ? 'nam' : 'nữ'}`;
  if (st && st.complete) {
    hint.className = 'hint ok';
    $('hintIcon').innerHTML = '<use href="#i-check"/>';
    $('hintText').textContent = `${who} đã sẵn sàng, dùng được cả khi không có mạng.`;
  } else {
    hint.className = 'hint';
    $('hintIcon').innerHTML = '<use href="#i-download"/>';
    $('hintText').textContent = GPU ? `Lần đầu dùng, trang sẽ tải khoảng ${(cfg.qwen.bytes / 1e9).toFixed(1)} GB dữ liệu giọng (dùng chung cho cả 6 tiếng). Chỉ tải một lần, sau đó dùng được cả khi không có mạng. Nên dùng Wi-Fi.` : NO_GPU;
  }
  if (placeholder) setPlaceholder();
  renderSpeakers();
  $('go').disabled = busy || !GPU;
  $('goLabel').textContent = busy ? 'Đang tạo…' : 'Tạo giọng đọc';
  $('cancel').hidden = !busy;
  if (!busy && !$('status').textContent) idle();
}
const MAC = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
function idle() { if (GPU) setStatus('Nhập văn bản rồi bấm “Tạo giọng đọc”.'); else setStatus(NO_GPU, true); }
$('kbd').innerHTML = `Mẹo: <kbd>${MAC ? '⌘' : 'Ctrl'}</kbd> + <kbd>Enter</kbd> để tạo nhanh`;

// ---- text --------------------------------------------------------------------------------------
const text = $('text');
function setPlaceholder() { text.value = ''; text.placeholder = sylMode && lang === 'zh' ? 'Gõ chữ Hán hoặc pinyin có số thanh, ví dụ:\nma1 ma2 ma3 ma4\n你好' : `Nhập hoặc dán văn bản vào đây, ví dụ:\n${cfg.languages[lang].sample}`; placeholder = true; count(); }
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

// ---- dialogue (multi-speaker) -------------------------------------------------------------------
// A line "Name: text" starts a turn; lines without a name continue the previous speaker.
const TURN = /^\s*([^:：\n.。!?！？…"“”()（）]{1,24}?)\s*[:：]\s*(.*)$/u;
function parseDialogue(src) {
  const turns = [];
  for (const line of src.split(/\r?\n/)) {
    if (!line.trim()) continue;
    const m = line.match(TURN);
    if (m && m[2].trim()) turns.push({ speaker: m[1].trim(), text: m[2].trim() });
    else if (turns.length) turns[turns.length - 1].text += '\n' + line.trim();
    else turns.push({ speaker: '', text: line.trim() });
  }
  return turns;
}
function voicePool(L, g) {
  return L.dialogue_voices[g];
}
const voiceLabel = (id, L) => `${(cfg.voice_labels || {})[id] || `${genderOf(id) === 'male' ? 'Nam' : 'Nữ'} ${id.replace(/^z[mf]_/, '')}`}${id === L.male || id === L.female ? ' (chính)' : ''}`;
const genderOf = (id) => { const d = (cfg.languages[lang].dialogue_voices || {}); if (d.male?.includes(id)) return 'male'; if (d.female?.includes(id)) return 'female'; return id.startsWith('zm_') || id.startsWith('M') ? 'male' : 'female'; };
function speakerVoices(names) {
  const L = cfg.languages[lang], map = (spkVoices[lang] ||= {});
  const used = new Set(names.map(n => map[n]).filter(Boolean));
  names.forEach((n, i) => {
    if (map[n]) return;
    const g = i % 2 === 0 ? 'female' : 'male';
    map[n] = voicePool(L, g).find(v => !used.has(v)) || L[g];
    used.add(map[n]);
  });
  return map;
}
const sylOn = () => sylMode && lang === 'zh';
function renderSpeakers() {
  $('sylMode').hidden = true;
  $('sylMode').setAttribute('aria-pressed', sylOn());
  $('sylOpts').hidden = !sylOn();
  $('dlgMode').setAttribute('aria-pressed', dialogue && !sylOn());
  $('speakers').hidden = !dialogue || sylOn();
  if (sylOn()) return;
  if (!dialogue) return;
  const L = cfg.languages[lang];
  const turns = parseDialogue(placeholder ? '' : text.value);
  const names = [...new Set(turns.map(t => t.speaker || '(không tên)'))];
  const list = $('spkList'); list.textContent = '';
  if (!names.length) { list.innerHTML = '<div class="sub">Chưa có lượt thoại nào.</div>'; return; }
  const map = speakerVoices(names);
  for (const n of names) {
    const row = document.createElement('div'); row.className = 'spk';
    const who = document.createElement('div'); who.className = 'who'; who.textContent = n;
    const k = turns.filter(t => (t.speaker || '(không tên)') === n).length;
    const small = document.createElement('small'); small.textContent = `${k} lượt`; who.append(small);
    const pills = document.createElement('div'); pills.className = 'pills';
    const sel = document.createElement('select'); sel.setAttribute('aria-label', 'Giọng của ' + n);
    const fill = (g) => { sel.textContent = ''; voicePool(L, g).forEach(v => sel.append(new Option(voiceLabel(v, L), v))); sel.value = map[n]; };
    for (const [g, label] of [['female', 'Nữ'], ['male', 'Nam']]) {
      const b = document.createElement('button'); b.className = 'pill'; b.textContent = label;
      b.setAttribute('aria-pressed', genderOf(map[n]) === g);
      b.onclick = () => { if (genderOf(map[n]) !== g) { map[n] = voicePool(L, g)[0]; renderSpeakers(); } };
      pills.append(b);
    }
    fill(genderOf(map[n]));
    sel.onchange = () => { map[n] = sel.value; };
    row.append(who, pills, sel); list.append(row);
  }
}
let spkTimer = null;
text.addEventListener('input', () => { if (dialogue) { clearTimeout(spkTimer); spkTimer = setTimeout(renderSpeakers, 250); } });
$('dlgMode').onclick = () => { dialogue = !(dialogue && !sylOn()); if (dialogue) sylMode = false; store.set('dialogue', dialogue ? '1' : '0'); store.set('sylMode', sylMode ? '1' : '0'); renderSpeakers(); if (placeholder) setPlaceholder(); };
$('sylMode').onclick = () => { sylMode = !sylOn(); if (sylMode) dialogue = false; store.set('sylMode', sylMode ? '1' : '0'); store.set('dialogue', dialogue ? '1' : '0'); renderSpeakers(); if (placeholder) setPlaceholder(); };
for (const [id, key] of [['sylGap', 'sylGap'], ['sylWord', 'sylWord']]) {
  const el = $(id), v = store.get(key);
  if (v != null) { if (el.type === 'checkbox') el.checked = v === '1'; else el.value = v; }
  el.onchange = () => store.set(key, el.type === 'checkbox' ? (el.checked ? '1' : '0') : el.value);
}

// ---- job ---------------------------------------------------------------------------------------
const bar = $('bar');
function setStatus(msg, warn = false) { $('status').textContent = msg; $('status').className = warn ? 'warn' : ''; }
function progress(frac) { bar.hidden = false; bar.classList.toggle('indet', frac == null); bar.firstElementChild.style.width = frac == null ? '' : Math.round(frac * 100) + '%'; }

function spawn(file) {
  const w = new Worker(new URL(file, import.meta.url), { type: 'module' });
  w.onmessage = ({ data }) => handlers[data.type]?.(data);
  w.onerror = (e) => { finish(); bar.hidden = true; setStatus('Không chạy được phần đọc giọng trên trình duyệt này. Hãy thử Chrome, Edge hoặc Safari bản mới. (' + (e.message || 'không rõ lỗi') + ')', true); };
  w.postMessage({ type: 'status' });
  return w;
}
function newWorker() { worker = GPU ? spawn('qwen-worker.js?v=20261021b') : null; }
const handlers = {
  status: (d) => setStatus(d.message),
  download: (d) => { setStatus(`Đang tải dữ liệu giọng lần đầu · ${mb(d.got)} / ${mb(d.total)} (${Math.round(100 * d.got / d.total)}%)`); progress(d.got / d.total); },
  progress: (d) => { setStatus(d.total > 1 ? `Đang tạo giọng đọc… đoạn ${d.done + 1}/${d.total}` : 'Đang tạo giọng đọc…'); progress(d.total > 1 ? d.done / d.total : null); },
  done: (d) => { bar.hidden = true; showResult(d.audio, d.sampleRate); setStatus(`Đã xong · ${clock(d.audio.length / d.sampleRate)} âm thanh. Bấm “Nghe” hoặc tải file về.`); finish(); worker?.postMessage({ type: 'status' }); },
  error: (d) => { finish(); bar.hidden = true; setStatus(d.message, true); },
  cancelled: () => { finish(); bar.hidden = true; setStatus('Đã huỷ.'); },
  cache: (d) => { cache = { ...cache, ...d.engines }; refresh(); renderData(); },
  cleared: () => { $('dataMsg').textContent = 'Đã xoá dữ liệu giọng.'; worker?.postMessage({ type: 'status' }); },
};
function finish() { busy = false; animate(false); refresh(); }
const clock = (secs) => { secs = Math.round(secs); return `${String(Math.floor(secs / 60)).padStart(2, '0')}:${String(secs % 60).padStart(2, '0')}`; };

$('go').onclick = () => {
  const content = text.value.trim();
  if (!content) { setStatus('Chưa có văn bản để đọc. Hãy nhập, dán hoặc mở tệp .txt.', true); text.focus(); return; }
  const L = cfg.languages[lang];
  busy = true; animate(true);
  $('result').classList.remove('show'); setStatus('Đang chuẩn bị…'); progress(null); refresh();
  const single = (L.choices || {})[gender] ? $('voice').value : L[gender];
  let segments = [{ text: content, voice: single }];
  if (dialogue && !sylOn()) {
    const turns = parseDialogue(content);
    const map = speakerVoices([...new Set(turns.map(t => t.speaker || '(không tên)'))]);
    segments = turns.map(t => ({ text: t.text, voice: map[t.speaker || '(không tên)'] }));
  }
  lastDialogue = dialogue && !sylOn() && segments.length > 1;
  lastSyl = sylOn();
  worker.postMessage({ type: 'synthesize', engine: L.engine, lang, segments, text: content, pauses: { ...cfg.pauses, ...((cfg.pauses_by_lang || {})[lang] || {}) } });
};
$('cancel').onclick = () => {
  // Terminating is the only way to stop a running ONNX call; the next job starts a fresh worker (models stay cached on disk).
  worker?.terminate(); newWorker(); finish(); bar.hidden = true; setStatus('Đã huỷ.');
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
  const src = lastDialogue ? parseDialogue(text.value).map(t => t.text).join(' ') : text.value;
  const words = (src.match(/[\p{L}\p{N}]+/gu) || []).slice(0, 5).join('-').slice(0, 40) || 'giong-noi';
  const L = cfg.languages[lang], who = (L.choices || {})[gender] ? $('voice').value : (gender === 'male' ? 'nam' : 'nu');
  return `${lang}_${lastSyl ? 'amtiet' : lastDialogue ? 'hoithoai' : who}_${words}`;
}
function showResult(samples, sr) {
  const blob = wav(samples, sr);
  if (lastUrl) URL.revokeObjectURL(lastUrl);
  lastUrl = URL.createObjectURL(blob);
  const secs = Math.round(samples.length / sr);
  const name = filename();
  $('player').src = lastUrl; $('save').href = lastUrl; $('save').download = name + '.wav';
  $('rname').textContent = name + '.mp3';
  $('rmeta').textContent = `${clock(secs)} · WAV ${(blob.size / 1e6).toFixed(1)} MB`;
  $('result').classList.add('show'); $('playLabel').textContent = 'Nghe';
  makeMp3(samples, sr, name, secs, blob.size);
}
// MP3 is encoded in its own worker after the WAV is ready, so listening is never delayed.
function makeMp3(samples, sr, name, secs, wavBytes) {
  if (mp3w) mp3w.terminate();
  if (mp3Url) { URL.revokeObjectURL(mp3Url); mp3Url = null; }
  const a = $('saveMp3'); a.removeAttribute('href'); a.setAttribute('aria-disabled', 'true'); $('mp3Label').textContent = 'Đang tạo MP3…';
  mp3w = new Worker(new URL('mp3worker.js', import.meta.url));
  mp3w.onmessage = ({ data }) => {
    if (data.type === 'progress') $('mp3Label').textContent = `Đang tạo MP3… ${Math.round(100 * data.done / data.total)}%`;
    else if (data.type === 'done') {
      mp3Url = URL.createObjectURL(data.blob);
      a.href = mp3Url; a.download = name + '.mp3'; a.removeAttribute('aria-disabled'); $('mp3Label').textContent = 'Tải MP3';
      $('rmeta').textContent = `${clock(secs)} · MP3 ${(data.blob.size / 1e6).toFixed(1)} MB · WAV ${(wavBytes / 1e6).toFixed(1)} MB`;
      mp3w.terminate(); mp3w = null;
    } else { $('mp3Label').textContent = 'Không tạo được MP3'; mp3w.terminate(); mp3w = null; }
  };
  mp3w.onerror = () => { $('mp3Label').textContent = 'Không tạo được MP3'; };
  mp3w.postMessage({ samples, sampleRate: sr, kbps: 128 });
}
$('play').onclick = () => { const a = $('player'); if (a.paused) { a.currentTime = a.ended ? 0 : a.currentTime; a.play(); } else a.pause(); };
$('player').onplay = () => ($('playLabel').textContent = 'Dừng');
$('player').onpause = $('player').onended = () => ($('playLabel').textContent = 'Nghe');

// ---- data dialog -----------------------------------------------------------------------------
const NAMES = { qwen3: 'Cả 6 tiếng (Qwen3-TTS 0.6B)' };
function renderData() {
  const list = $('engList'); list.textContent = '';
  const all = { qwen3: { bytes: cfg.qwen.bytes } };
  for (const [name, e] of Object.entries(all)) {
    const st = cache[name] || {};
    const row = document.createElement('div'); row.className = 'eng';
    row.innerHTML = `<div class="grow"><b></b><div class="sub"></div></div><span class="chip"></span>`;
    row.querySelector('b').textContent = NAMES[name] || name;
    row.querySelector('.sub').textContent = e.bytes > 1e9 ? `${(e.bytes / 1e9).toFixed(1)} GB` : mb(e.bytes);
    const chip = row.querySelector('.chip');
    chip.textContent = st.complete ? 'Đã lưu trên máy' : st.parts ? `Tải dở (${st.parts}/${st.total} phần)` : 'Chưa tải';
    if (st.complete) chip.classList.add('ok');
    list.append(row);
  }
}
$('data').onclick = () => { $('dataMsg').textContent = ''; renderData(); $('dataDlg').showModal(); };
$('closeDlg').onclick = () => $('dataDlg').close();
$('wipe').onclick = () => { if (confirm('Xoá dữ liệu giọng đã lưu trên máy này? Lần dùng sau sẽ phải tải lại.')) worker?.postMessage({ type: 'clear' }); };

newWorker();
setPlaceholder();
refresh();
idle();
