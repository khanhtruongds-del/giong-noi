# Đọc hàng loạt (jobs.json → MP3 + ket-qua.json)

Trạm đọc cho PLS Classroom: Classroom xuất danh sách cần đọc (`tools/xuat-can-doc.js` → `out/can-doc/<trung tâm>/jobs.json`),
giong-noi đọc thành MP3, Classroom nạp lại (`tools/nhan-audio.js --ket-qua <thư mục>`). Định dạng trao đổi: `docs/am-thanh-tts.md` của Classroom.

Cả ba cách dưới đây dùng **đúng các file model của bản web** (huggingface.co/Khanhpls/qwen3-tts-web: Qwen3-TTS-12Hz-0.6B-Base,
talker 8-bit, giọng = vector người nói trong `voices.json`) và cùng quy tắc: chia câu (`text.js`), giới hạn khung, đọc lại khi rỗng /
lan man, cắt khoảng lặng, nghỉ 0,3 s giữa câu (tiếng Trung 0,5 s), 0,6 s giữa lượt hội thoại, MP3 128 kbps mono 24 kHz.

| Cách | Khi nào | Tốc độ đo được (8/11/2026) |
|---|---|---|
| **Kaggle** – `kaggle/doc-hang-loat.ipynb` | danh sách dài, chạy nền, máy nhà tắt được | xem mục Tốc độ |
| **Máy có GPU NVIDIA** – `doc_hang_loat.py` | máy RTX rảnh | RTX 3070: 1,8 s/câu tiếng Trung |
| **Trình duyệt** – `hang-loat.html` | vài chục việc, không cài gì | xem mục Tốc độ |

Việc lỗi ghi `"trang_thai": "loi"`; chạy lại thì việc đã `xong` được giữ, chỉ đọc lại việc lỗi / chưa làm.
Mỗi việc có hạt giống ngẫu nhiên lấy từ `id` (bản Python) nên chạy lại cho cùng kết quả.

## 1. Kaggle (GPU miễn phí, chạy nền)

Lần đầu (một lần):
1. Tài khoản Kaggle đã **xác minh số điện thoại** (mới bật được GPU và Internet).
2. **Tạo dataset riêng tư** chứa `jobs.json`: kaggle.com → Datasets → New Dataset → kéo thả `out/can-doc/<trung tâm>/jobs.json`
   → đặt tên (vd `can-doc-dac-nhan`) → để **Private** → Create.
3. **Tạo notebook**: kaggle.com → Code → New Notebook → File → Import Notebook → chọn `kaggle/doc-hang-loat.ipynb` trong repo này.
4. Bên phải notebook: **Add Input** → Your Datasets → chọn dataset vừa tạo. **Settings**: Accelerator = *GPU T4 x2* (hoặc *GPU P100*),
   Internet = *On*.

Mỗi lần đọc:
1. (Nếu jobs.json đổi) dataset → New Version → tải file mới lên.
2. Notebook → **Save Version** → *Save & Run All (Commit)* → Save. Kaggle chạy nền, tắt trình duyệt được; xong có email.
   Muốn thử nhanh trước: sửa `GIOI_HAN = 20` ở ô đầu.
3. Mở phiên bản vừa chạy → tab **Output** → tải `ket-qua.zip` → giải nén → trong Classroom:
   ```bash
   node tools/nhan-audio.js --center 3_Dac_Nhan --ket-qua <thư mục đã giải nén>
   ```
4. Còn việc lỗi / chưa xong (hết 12 giờ): Add Input → *Notebook Output Files* → chọn chính notebook này (phiên bản vừa chạy)
   → Save Version lần nữa. Notebook chép kết quả cũ sang rồi chỉ đọc phần còn thiếu.

Hạn mức Kaggle: 1 phiên GPU tối đa 12 giờ, khoảng 30 giờ GPU / tuần. Notebook tự dừng ở 11,3 giờ (`HET_GIO`) để kịp lưu.
Model (~1,9 GB) tải vào `/tmp`, không lọt vào Output.

Tự động sau này (Kaggle API + GitHub Actions): `kaggle datasets version -p <thư mục jobs>` rồi `kaggle kernels push -p <thư mục notebook>`,
`kaggle kernels status` / `kaggle kernels output`. Khoá API (`KAGGLE_USERNAME`, `KAGGLE_KEY`) do người dùng tự đặt vào GitHub Secrets –
không ghi vào repo.

Sửa mã: sửa `doc_hang_loat.py` rồi tạo lại notebook: `node tools/doc-hang-loat/tao-notebook.js`.

## 2. Máy có GPU NVIDIA (Windows / Linux)

```bash
python -m pip install "onnxruntime-gpu[cuda,cudnn]" tokenizers lameenc numpy
python tools/doc-hang-loat/doc_hang_loat.py --jobs <jobs.json> --out <thư mục kết quả> [--gioi-han 20] [--het-gio 3]
```
Lần đầu tự tải model vào `tools/doc-hang-loat/model/` (đã có trong `.gitignore`); có sẵn thì thêm `--model <thư mục>`.
Không cần cài CUDA riêng (gói `[cuda,cudnn]` kèm thư viện). Không có GPU thì chạy CPU (`--cpu`), chậm hơn nhiều.

## 3. Trình duyệt (`hang-loat.html`)

Mở `…/giong-noi/hang-loat.html` bằng Chrome/Edge có WebGPU → *Chọn jobs.json* → *Bắt đầu / chạy tiếp* → *Tải ket-qua.zip*.
Mỗi việc xong được lưu ngay trong trình duyệt (IndexedDB): đóng trang, mở lại, chọn đúng file jobs.json rồi bấm *Bắt đầu / chạy tiếp*
là đọc tiếp. Đánh dấu *đọc lại cả việc lỗi* để thử lại việc lỗi.

## Tốc độ (đo 8/11/2026)

Đo trên máy RTX 3070 8 GB (Windows, driver 621). Thời gian tính từ lúc bắt đầu một việc đến khi có file MP3; nạp model mất thêm 13–20 s mỗi lần chạy.

| Cách chạy | Từ (1 từ) | Câu tiếng Trung (~18 chữ, ~3,5 s âm thanh) | Hội thoại tiếng Nhật |
|---|---|---|---|
| Python + CUDA (`doc_hang_loat.py`) | 0,5–1,3 s | **1,8 s** (20 việc đầu Đắc Nhân: trung bình 1,77 s/việc) | ~0,09 s/ký tự (đoạn 30 s âm thanh mất ~17 s) |
| Trình duyệt WebGPU (`hang-loat.html`) | 5–7 s | ~24 s (khoảng 7 s cho mỗi giây âm thanh) | – |

Python chạy talker 13 ms/bước, bộ đoán mã 23 ms/khung (12,5 khung = 1 giây âm thanh), tức nhanh hơn thời gian thực khoảng 2 lần.
Trình duyệt chậm hơn ~13 lần vì bộ đoán mã chạy trên CPU (WASM 1 luồng).

Ước thời gian đọc (Python + CUDA, RTX 3070):
- **Đắc Nhân** – 185 việc (4 từ, 181 câu): **~6 phút**.
- **SAKUMI** – 1.756 việc (922 từ ≈ 15 phút, 711 câu ≈ 23 phút – đo 30 câu tiếng Nhật: 1,8 s/câu, 123 hội thoại / 29.770 ký tự ≈ 46 phút – đo 10 bài: ~0,5 s cho mỗi giây âm thanh): **~1,5 giờ**.

Kaggle (T4 / P100) chưa đo được ở đây (cần tài khoản Kaggle của người dùng). Card T4 đọc từng bước chậm hơn RTX 3070, nên dự kiến
chậm hơn khoảng 1,5–2,5 lần: Đắc Nhân ~10–15 phút, SAKUMI ~2,5–4 giờ. Như vậy vẫn xong trong một phiên 12 giờ.
Notebook in ra `trung bình … s/việc` ở cuối lần chạy để cập nhật bảng này.

Kiểm chất lượng: nhận dạng lại 20 file Python bằng Whisper small. Câu tiếng Trung khớp gần như nguyên văn (chỉ khác ở chỗ Whisper viết chữ
phồn thể). Riêng 他虽然失败了很多次，然而… thì Whisper nghe 然而 thành “神兒”, nên cần giáo viên nghe lại. Từ đơn tiếng Nhật có vài chỗ
nhầm âm (許可 → 教科), đúng như lý do mọi file đều ở trạng thái `duyet: false`.

Giọng Python và giọng web là một: cùng vector người nói, cùng model. Đo 3 câu Nữ 4: khoảng cách MFCC Python ↔ web là 10–13, trong khi
Nữ 4 ↔ Nữ 1 / Nữ 2 / Nam 1 là 25–58. Cao độ trung vị Python 205 Hz, web 216–219 Hz; Nữ 1 khoảng 270 Hz, Nữ 2 khoảng 180 Hz.
Mỗi lần đọc có lấy mẫu ngẫu nhiên nên hai bản không trùng từng mẫu âm thanh.

## Giấy phép (kiểm 8/11/2026)

- **Qwen3-TTS-12Hz-0.6B-Base** (model đọc), **Qwen3-TTS-Tokenizer-12Hz** (bộ giải mã âm thanh), **Qwen3-TTS-12Hz-1.7B-VoiceDesign**
  (dùng để thiết kế giọng), **Qwen3-TTS-12Hz-1.7B-CustomVoice** (giọng có sẵn Serena / Aiden) đều theo **Apache 2.0** (thẻ model trên
  Hugging Face; mã nguồn github.com/QwenLM/Qwen3-TTS cũng Apache 2.0). Thẻ model không có điều khoản cấm thương mại hay giới hạn số
  người dùng. Apache 2.0 cho phép dùng thương mại, sửa đổi, phân phối; khi phân phối lại model hoặc mã thì phải kèm giấy phép và thông báo
  bản quyền (đã có ở `LICENSES/qwen3-tts-Apache-2.0.txt`). Giấy phép này không cấp quyền dùng tên hay thương hiệu Qwen / Alibaba.
- **Bản ONNX `Khanhpls/qwen3-tts-web`**: Apache 2.0 (bản phái sinh, có ghi nguồn).
- **Các giọng** (`voices.json`) là vector người nói rút từ âm thanh do chính Qwen3-TTS tạo, không thu từ người thật.
  Nữ 2–5, Nam 2–5, Cô, Bác gái, Bà, Chú, Bác trai, Ông được thiết kế bằng VoiceDesign từ mô tả chữ. **Nữ 1 = giọng Serena, Nam 1 = Aiden**
  (giọng có sẵn của CustomVoice). Về giấy phép, đầu ra của model Apache 2.0 không bị hạn chế. Tuy vậy Serena / Aiden là giọng
  có tên trong tài liệu của Qwen, ai dùng CustomVoice cũng tạo được. Nếu muốn giọng riêng không trùng với sản phẩm khác thì có thể đổi
  Nữ 1 / Nam 1 sang giọng VoiceDesign.
- **Âm thanh tạo ra** là giọng máy: nên ghi rõ “giọng do máy tạo” khi đưa cho học viên (như `licenses.html`). Mọi file vào Classroom
  đều ở `duyet: false`, giáo viên duyệt trước khi dùng.
- Đây là đọc điều khoản giấy phép, **không phải ý kiến pháp lý**. Nếu dùng ở quy mô thương mại lớn, nên để bộ phận pháp lý xem lại.
- Thư viện dùng thêm khi chạy: ONNX Runtime (MIT), tokenizers (Apache 2.0), lameenc / LAME (LGPL; chỉ dùng như thư viện, không phân phối
  kèm), numpy (BSD).

## Lưu ý

- Đừng cài gói `onnxruntime` (bản CPU) cạnh `onnxruntime-gpu`: gói nào cài sau sẽ che gói kia, model lặng lẽ chạy CPU
  (~77 s/câu thay vì 1,8 s). Ví dụ `faster-whisper` kéo theo `onnxruntime`. Script in **CẢNH BÁO** khi có CUDA mà không dùng được;
  notebook Kaggle gỡ `onnxruntime` trước khi cài và dừng ngay nếu không chạy được trên GPU.
- Một số câu trong DATA APP có dạng `A : …\nB : …` (loại `cau`): nhãn “A”, “B” sẽ bị đọc thành tiếng. Nên để Classroom xuất những câu này
  thành `hoi-thoai` hoặc bỏ nhãn trước khi đọc.

## Chất lượng (chat_luong.py)

`doc_hang_loat.py` dùng `chat_luong.py` (notebook Kaggle tự chép kèm):
- **Làm sạch chữ** trước khi đọc: bỏ ký hiệu từ điển (`der Opa, -s`, `+ Dat`, `(qn)`, `～`, `（＝…）`…).
- **Đọc từng vế**, nghỉ ở dấu phẩy / hết câu theo **nhịp đo từ audio thật** của từng tiếng (bảng `NHIP`); từ chối bản "nói tràn" (dài bất thường).
- **Hậu kỳ**: kéo chậm phần nói về tốc độ chuẩn của tiếng (WSOLA), −20 dBFS, đỉnh ≤ −1 dBFS, fade 10 ms, lặng đầu / cuối.
- **Tự chấm bằng Whisper** (`faster-whisper` large-v3-turbo, CER theo đơn vị đọc: pinyin có thanh / kana / hangul / chữ cái): đọc tối đa `--so-ban` bản (mặc định 3) cho từ / câu, giữ bản nghe ra đúng nhất; tiếng Nhật bản cuối đọc bằng kana. `ket-qua.json` ghi `cer`, `nghe`, `ban`, `can_nghe` (mục nên nghe lại).
- Tắt chấm: `--khong-cham` (nhanh, chỉ đọc 1 bản). Notebook: `SO_BAN`, `CHAM`, `KBPS` ở ô Tuỳ chỉnh.

Đo (13/11/2026, 100 mục mỗi bộ): từ tiếng Trung đạt chặt 78% → 87%, từ tiếng Nhật 75% → 86%, câu tiếng Đức 99%.
