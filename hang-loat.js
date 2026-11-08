// Đọc hàng loạt trong trình duyệt: jobs.json (định dạng docs/am-thanh-tts.md của PLS Classroom) → MP3 + ket-qua.json → zip.
// Dùng lại qwen-worker.js (tạo âm thanh) và mp3worker.js (lamejs). Mỗi việc xong được lưu ngay vào IndexedDB,
// nên trang bị đóng giữa chừng vẫn chạy tiếp từ việc chưa xong. Cùng quy tắc với tools/doc-hang-loat/doc_hang_loat.py.
const $ = (id) => document.getElementById(id);
const cfg = await (await fetch('config.json', { cache: 'no-cache' })).json();
const MODEL = 'Qwen3-TTS 0.6B (Khanhpls/qwen3-tts-web)';

// ---- IndexedDB: kết quả từng việc ----------------------------------------------------------------
const db = await new Promise((ok, no) => {
  const r = indexedDB.open('giong-noi-hang-loat', 1);
  r.onupgradeneeded = () => r.result.createObjectStore('kq');
  r.onsuccess = () => ok(r.result); r.onerror = () => no(r.error);
});
const tx = (mode, fn) => new Promise((ok, no) => { const t = db.transaction('kq', mode), s = t.objectStore('kq'), r = fn(s); t.oncomplete = () => ok(r?.result); t.onerror = () => no(t.error); });
const dbGet = (k) => tx('readonly', (s) => s.get(k));
const dbPut = (k, v) => tx('readwrite', (s) => s.put(v, k));
const dbKeys = () => tx('readonly', (s) => s.getAllKeys());
const dbDel = (k) => tx('readwrite', (s) => s.delete(k));

// ---- giọng, chữ ------------------------------------------------------------------------------------
function voiceCode(label, lang) {
  if (cfg.voice_labels[label]) return label;
  let hits = Object.keys(cfg.voice_labels).filter(k => cfg.voice_labels[k] === label);
  if (!hits.length) throw new Error(`Không biết giọng "${label}"`);
  const L = cfg.languages[lang]?.choices, all = L ? [...L.female, ...L.male] : [];
  return hits.find(k => all.includes(k)) || hits[0];      // tiếng Nhật: Cô → co_ja (như config.json)
}
const clean = (t) => String(t || '').replace(/\{([^{}|]+)\|[^{}]*\}/g, '$1').trim();   // {漢字|かんじ} → 漢字
const defVoice = (lang) => cfg.languages[lang]?.female || 'nu1';
function segments(job) {
  if (job.loai === 'hoi-thoai' && job.lines?.length)
    return job.lines.map(l => ({ text: clean(l.text), voice: voiceCode((job.vai || {})[l.vai] || job.giong || defVoice(job.lang), job.lang) })).filter(s => s.text);
  return [{ text: clean(job.text), voice: voiceCode(job.giong || defVoice(job.lang), job.lang) }];
}

// ---- worker ----------------------------------------------------------------------------------------
let worker = null, pending = null;
function newWorker() {
  worker?.terminate();
  worker = new Worker(new URL('qwen-worker.js?v=20261108a', import.meta.url), { type: 'module' });
  worker.onmessage = ({ data: d }) => {
    if (d.type === 'download') status(`Đang tải dữ liệu giọng lần đầu · ${(d.got / 1e6).toFixed(0)} / ${(d.total / 1e6).toFixed(0)} MB`);
    else if (d.type === 'status' && d.message) status(d.message);
    else if (d.type === 'done') { pending?.ok(d); pending = null; }
    else if (d.type === 'error' || d.type === 'cancelled') { pending?.no(new Error(d.message || 'đã huỷ')); pending = null; }
  };
  worker.onerror = (e) => { pending?.no(new Error(e.message || 'worker lỗi')); pending = null; };
}
const synth = (job) => new Promise((ok, no) => {
  pending = { ok, no };
  worker.postMessage({ type: 'synthesize', lang: job.lang, segments: segments(job), pauses: { ...cfg.pauses, ...((cfg.pauses_by_lang || {})[job.lang] || {}) } });
});
let mp3w = null;
const toMp3 = (samples, sampleRate) => new Promise((ok, no) => {
  mp3w ||= new Worker('mp3worker.js');
  mp3w.onmessage = ({ data: d }) => { if (d.type === 'done') ok(d.blob); else if (d.type === 'error') no(new Error(d.message)); };
  mp3w.postMessage({ samples, sampleRate, kbps: 128 });
});

// ---- zip (không nén – mp3 vốn đã nén) -------------------------------------------------------------
const CRC = (() => { const t = new Uint32Array(256); for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; } return t; })();
const crc32 = (b) => { let c = 0xffffffff; for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };
async function makeZip(files) {               // files: [{ name, data: Uint8Array }]
  const enc = new TextEncoder(), parts = [], central = []; let off = 0;
  const now = new Date(), dosT = (now.getHours() << 11) | (now.getMinutes() << 5) | (now.getSeconds() >> 1), dosD = ((now.getFullYear() - 1980) << 9) | ((now.getMonth() + 1) << 5) | now.getDate();
  for (const f of files) {
    const name = enc.encode(f.name), crc = crc32(f.data), n = f.data.length;
    const h = new DataView(new ArrayBuffer(30));
    h.setUint32(0, 0x04034b50, true); h.setUint16(4, 20, true); h.setUint16(6, 0x0800, true); h.setUint16(8, 0, true);
    h.setUint16(10, dosT, true); h.setUint16(12, dosD, true); h.setUint32(14, crc, true); h.setUint32(18, n, true); h.setUint32(22, n, true);
    h.setUint16(26, name.length, true); h.setUint16(28, 0, true);
    parts.push(h.buffer, name, f.data);
    const c = new DataView(new ArrayBuffer(46));
    c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint16(8, 0x0800, true); c.setUint16(10, 0, true);
    c.setUint16(12, dosT, true); c.setUint16(14, dosD, true); c.setUint32(16, crc, true); c.setUint32(20, n, true); c.setUint32(24, n, true);
    c.setUint16(28, name.length, true); c.setUint32(42, off, true);
    central.push(c.buffer, name);
    off += 30 + name.length + n;
  }
  const cSize = central.reduce((s, p) => s + p.byteLength, 0), e = new DataView(new ArrayBuffer(22));
  e.setUint32(0, 0x06054b50, true); e.setUint16(8, files.length, true); e.setUint16(10, files.length, true); e.setUint32(12, cSize, true); e.setUint32(16, off, true);
  return new Blob([...parts, ...central, e.buffer], { type: 'application/zip' });
}

// ---- giao diện --------------------------------------------------------------------------------------
let doc = null, key = '', running = false, stopFlag = false;
const status = (m, cls = 'muted') => { $('status').className = cls; $('status').textContent = m; };
const log = (m, cls = '') => { const d = document.createElement('div'); if (cls) d.className = cls; d.textContent = m; $('log').prepend(d); };
const k = (id) => key + '/' + id;
window.__hangLoat = { times: [], audio: [] };   // để đo tốc độ từ bảng điều khiển

async function summary() {
  if (!doc) return;
  let ok = 0, bad = 0;
  for (const j of doc.jobs) { const r = await dbGet(k(j.id)); if (r?.kq.trang_thai === 'xong') ok++; else if (r) bad++; }
  $('info').textContent = `${doc.center || ''} · ${doc.jobs.length} việc (${doc.lang || ''}) · đã xong ${ok}${bad ? ` · lỗi ${bad}` : ''}`;
  $('bar').style.width = (100 * ok / doc.jobs.length) + '%';
  $('zip').disabled = !ok || running; $('reset').disabled = !(ok + bad) || running; $('start').disabled = running; $('stop').disabled = !running;
  return { ok, bad };
}

async function loadJobs(name, textPromise) {
  try {
    const f = { name };
    doc = JSON.parse(await textPromise);
    if (!Array.isArray(doc.jobs)) throw new Error('không có mục "jobs"');
    key = `${doc.center || f.name}|${doc.jobs.length}|${doc.jobs[0]?.id || ''}|${doc.jobs.at(-1)?.id || ''}`;
    $('log').textContent = ''; await summary(); status('Đã nạp. Bấm “Bắt đầu / chạy tiếp”.');
  } catch (e) { doc = null; status('Không đọc được jobs.json: ' + e.message, 'err'); }
}
$('file').onchange = () => { const f = $('file').files[0]; if (f) loadJobs(f.name, f.text()); };
const qs = new URLSearchParams(location.search).get('jobs');          // hang-loat.html?jobs=<đường dẫn cùng trang>
if (qs) loadJobs(qs, fetch(new URL(qs, location.href)).then(r => { if (!r.ok) throw new Error('HTTP ' + r.status); return r.text(); }));

$('start').onclick = async () => {
  if (!doc || running) return;
  running = true; stopFlag = false; await summary();
  if (!worker) newWorker();
  const redo = $('redo').checked, t0all = performance.now();
  let n = 0;
  const todo = [];
  for (const j of doc.jobs) { const r = await dbGet(k(j.id)); if (!r || (redo && r.kq.trang_thai !== 'xong')) todo.push(j); }
  for (const j of todo) {
    if (stopFlag) break;
    n++;
    const t0 = performance.now();
    status(`Đang đọc ${n}/${todo.length}: ${j.file}`);
    try {
      const d = await synth(j);
      const blob = await toMp3(d.audio, d.sampleRate);
      const secs = (performance.now() - t0) / 1000, dur = +(d.audio.length / d.sampleRate).toFixed(2);
      await dbPut(k(j.id), { mp3: blob, kq: { id: j.id, trang_thai: 'xong', file: j.file, giong: j.loai === 'hoi-thoai' ? j.vai : j.giong, thoi_luong: dur, giay_tao: +secs.toFixed(2) } });
      window.__hangLoat.times.push(secs); window.__hangLoat.audio.push(dur);
      const avg = window.__hangLoat.times.reduce((a, b) => a + b, 0) / window.__hangLoat.times.length;
      log(`[${n}/${todo.length}] ${secs.toFixed(1)} s → ${dur.toFixed(2)} s âm thanh · ${j.file} · ${clean(j.text).slice(0, 30)}`);
      status(`Trung bình ${avg.toFixed(2)} s/việc · còn khoảng ${Math.ceil(avg * (todo.length - n) / 60)} phút`);
    } catch (e) {
      if (stopFlag) break;
      if (!window.__hangLoat.times.length) {         // chưa việc nào xong: thường là lỗi nạp model / WebGPU, không phải lỗi của việc này
        log(`Dừng: ${e.message}`, 'err'); newWorker(); break;
      }
      await dbPut(k(j.id), { kq: { id: j.id, trang_thai: 'loi', file: j.file, loi: String(e.message || e) } });
      log(`[${n}/${todo.length}] LỖI ${j.file}: ${e.message}`, 'err');
      newWorker();
    }
    if (n % 5 === 0) await summary();
  }
  running = false;
  const s = await summary();
  status(`${stopFlag ? 'Đã dừng' : 'Xong lượt này'} · ${s.ok}/${doc.jobs.length} việc xong · ${((performance.now() - t0all) / 60000).toFixed(1)} phút.`, s.bad ? 'err' : 'ok');
};

$('stop').onclick = () => { stopFlag = true; worker?.terminate(); worker = null; pending?.no(new Error('đã dừng')); pending = null; };

$('zip').onclick = async () => {
  status('Đang gói zip…');
  const files = [], ket_qua = [];
  for (const j of doc.jobs) {
    const r = await dbGet(k(j.id));
    if (r?.kq.trang_thai === 'xong' && r.mp3) { files.push({ name: j.file, data: new Uint8Array(await r.mp3.arrayBuffer()) }); ket_qua.push(r.kq); }
    else ket_qua.push(r?.kq || { id: j.id, trang_thai: 'loi', file: j.file, loi: 'chưa chạy tới' });
  }
  const kq = { model: MODEL, tao_luc: new Date().toISOString(), center: doc.center, thiet_bi: 'webgpu', ket_qua };
  files.push({ name: 'ket-qua.json', data: new TextEncoder().encode(JSON.stringify(kq, null, 1)) });
  const a = document.createElement('a'); a.href = URL.createObjectURL(await makeZip(files));
  a.download = `ket-qua-${doc.center || 'jobs'}.zip`; a.click();
  status(`Đã tạo zip: ${files.length - 1} file mp3 + ket-qua.json.`, 'ok');
};

$('reset').onclick = async () => {
  if (!doc || !confirm('Xoá mọi kết quả đã lưu của danh sách này trong trình duyệt?')) return;
  for (const kk of await dbKeys()) if (String(kk).startsWith(key + '/')) await dbDel(kk);
  $('log').textContent = ''; await summary(); status('Đã xoá.');
};
