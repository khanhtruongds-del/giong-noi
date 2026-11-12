// Tạo kaggle/doc-hang-loat.ipynb từ doc_hang_loat.py (notebook tự chứa mã, không cần clone repo).
//   node tools/doc-hang-loat/tao-notebook.js
const fs = require('fs'), path = require('path');
const here = __dirname, root = path.join(here, '..', '..');
const src = fs.readFileSync(path.join(here, 'doc_hang_loat.py'), 'utf8').replace(/\r\n/g, '\n');
const srcCL = fs.readFileSync(path.join(here, 'chat_luong.py'), 'utf8').replace(/\r\n/g, '\n');   // làm sạch chữ, nhịp theo tiếng, hậu kỳ, chấm Whisper
const lines = (s) => s.replace(/\r\n/g, '\n').split('\n').map((l, i, a) => (i < a.length - 1 ? l + '\n' : l)).filter((l, i, a) => !(i === a.length - 1 && l === ''));
const md = (s) => ({ cell_type: 'markdown', metadata: {}, source: lines(s) });
const code = (s) => ({ cell_type: 'code', metadata: {}, execution_count: null, outputs: [], source: lines(s) });

const cells = [
  md(`# Giọng nói – đọc hàng loạt (jobs.json → mp3)

Chạy Qwen3-TTS 0.6B bằng **đúng các file model của bản web giong-noi** (huggingface.co/Khanhpls/qwen3-tts-web, giọng trong \`voices.json\`).
Đầu vào: \`jobs.json\` do PLS Classroom xuất (\`tools/xuat-can-doc.js\`). Đầu ra: \`/kaggle/working/ket-qua/\` (mp3 theo đường dẫn \`file\` + \`ket-qua.json\`) và \`ket-qua.zip\`.

**Cách dùng** (chi tiết: \`tools/doc-hang-loat/README.md\` trong repo giong-noi)
1. Tạo Dataset **riêng tư** chứa \`jobs.json\`, gắn vào notebook (Add Input).
2. Settings: Accelerator = **GPU T4 x2** hoặc **P100**; Internet = **On** (tải model ~1,9 GB và thư viện).
3. Bấm **Save Version → Save & Run All (Commit)** để chạy nền; tắt trình duyệt được.
4. Xong: mở phiên bản đó → tab **Output** → tải \`ket-qua.zip\`, giải nén rồi chạy \`node tools/nhan-audio.js --center … --ket-qua <thư mục>\` trong Classroom.

Chạy lại phần lỗi / chưa xong: gắn thêm **Output của phiên bản trước** làm Input rồi Save Version lần nữa – việc đã \`xong\` được giữ, chỉ đọc việc \`loi\`.`),
  code(`# Tuỳ chỉnh
GIOI_HAN = None      # số việc đầu tiên cần đọc (vd 20 để thử tốc độ); None = tất cả
HET_GIO = 11.3       # dừng an toàn trước giới hạn 12 giờ của Kaggle; phần còn lại ghi "loi" để lần sau chạy tiếp
OUT = '/kaggle/working/ket-qua'
MODEL_DIR = '/tmp/qwen3-tts-web'   # ngoài /kaggle/working để model không lọt vào Output
SO_BAN = 3           # số bản đọc cho mỗi từ / câu; Whisper chọn bản nghe đúng nhất (1 = nhanh, không chấm)
CHAM = True         # tự chấm bằng Whisper (faster-whisper large-v3-turbo); False = chỉ đọc
KBPS = 128          # chất lượng mp3`),
  code(`!pip install -q faster-whisper       # tự chấm; cài TRƯỚC vì nó kéo theo onnxruntime bản CPU
!pip uninstall -y -q onnxruntime   # bản CPU che mất bản GPU
!pip install -q --force-reinstall "onnxruntime-gpu[cuda,cudnn]" tokenizers lameenc
!nvidia-smi --query-gpu=name,memory.total --format=csv`),
  code('%%writefile /tmp/chat_luong.py\n' + srcCL),
  code('%%writefile /tmp/doc_hang_loat.py\n' + src),
  code(`import glob, os, shutil, sys
sys.path.insert(0, '/tmp')
import doc_hang_loat as dhl

jobs = sorted(glob.glob('/kaggle/input/**/jobs.json', recursive=True))
assert jobs, 'Chưa gắn dataset chứa jobs.json (Add Input)'
if len(jobs) > 1: print('Có nhiều jobs.json, dùng file đầu:', jobs)
JOBS = jobs[0]
os.makedirs(OUT, exist_ok=True)
# Output của lần chạy trước (nếu được gắn làm Input): chép sang để giữ việc đã xong
for kq in glob.glob('/kaggle/input/**/ket-qua/ket-qua.json', recursive=True):
    print('Dùng lại kết quả cũ:', kq)
    shutil.copytree(os.path.dirname(kq), OUT, dirs_exist_ok=True)

dhl.download_model(MODEL_DIR)
res = dhl.run(JOBS, OUT, MODEL_DIR, limit=GIOI_HAN, hours=HET_GIO, require_gpu=True, so_ban=SO_BAN, cham=CHAM, kbps=KBPS)`),
  code(`import json, shutil
kq = json.load(open(OUT + '/ket-qua.json', encoding='utf-8'))['ket_qua']
loi = [r for r in kq if r['trang_thai'] != 'xong']
print(f"xong {len(kq) - len(loi)} / {len(kq)} · lỗi/chưa làm {len(loi)}")
for r in loi[:20]: print(' ', r['id'], r['file'], r.get('loi'))
shutil.make_archive('/kaggle/working/ket-qua', 'zip', OUT)
print('→ /kaggle/working/ket-qua.zip')`),
];

const nb = {
  cells,
  metadata: {
    kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' },
    language_info: { name: 'python' },
    kaggle: { accelerator: 'gpu', isInternetEnabled: true, isGpuEnabled: true, dataSources: [] },
  },
  nbformat: 4, nbformat_minor: 4,
};
const out = path.join(root, 'kaggle', 'doc-hang-loat.ipynb');
fs.mkdirSync(path.dirname(out), { recursive: true });
fs.writeFileSync(out, JSON.stringify(nb, null, 1) + '\n');
console.log('→', path.relative(root, out));
