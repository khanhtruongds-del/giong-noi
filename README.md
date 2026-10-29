# Giọng nói offline — bản trình duyệt

Đọc văn bản thành giọng nói ngay trong trình duyệt cho 6 tiếng: Đức, Anh, Pháp, Trung, Nhật, Hàn, bằng Qwen3-TTS-12Hz-0.6B (Apache 2.0).
Model chạy trên máy người dùng bằng ONNX Runtime Web (WebGPU); văn bản không gửi lên máy chủ. Lần đầu dùng, trình duyệt tải
khoảng 1,9 GB từ https://huggingface.co/Khanhpls/qwen3-tts-web và giữ trong bộ nhớ đệm, sau đó dùng được khi không có mạng.
Cần trình duyệt có WebGPU (Chrome, Edge bản mới trên máy tính).

- `index.html`, `app.js` — giao diện; `qwen-worker.js` — tải và chạy model; `qwen.js` — dựng câu lệnh, lấy mẫu, giải mã âm thanh; `text.js` — chia câu
- `config.json` — giọng của từng tiếng (16 giọng: Nữ 1–5, Cô, Bác gái, Bà; Nam 1–5, Chú, Bác trai, Ông) và danh sách file model
- `vendor/gpu/` — ONNX Runtime Web 1.30 (WebGPU); `vendor/tokenizers.min.mjs` — Hugging Face tokenizers.js
- `licenses.html`, `LICENSES/` — giấy phép; giọng là giọng tổng hợp do máy tạo
- `syllables.js`, `am-tiet/` — chế độ **Âm tiết** (tiếng Trung): đọc từng âm tiết bằng bản thu sẵn (Qwen3-TTS 1.7B, giọng Serena nữ / Aiden nam),
  394 âm gốc × thanh = 1.121 file MP3 mỗi giọng (~5 KB/file, tải khi cần), không cần WebGPU. Chữ Hán → pinyin bằng `vendor/pinyin-pro.min.js` (MIT).
  Cách tạo bảng âm: `tools/am-tiet/` (WAV → MP3: `to_mp3.py <thư mục WAV> am-tiet`).
