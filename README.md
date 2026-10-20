# Giọng nói offline — bản trình duyệt

Đọc văn bản thành giọng nói ngay trong trình duyệt: Đức, Anh, Pháp, Nhật, Hàn (Supertonic 3) và Trung (Kokoro-82M-v1.1-zh).
Model chạy trên máy người dùng bằng ONNX Runtime Web; văn bản không gửi lên máy chủ. Lần đầu dùng mỗi bộ giọng, trình duyệt
tải dữ liệu (Supertonic ~200 MB, Kokoro ~163 MB) và giữ lại trong bộ nhớ đệm, sau đó dùng được khi không có mạng.

- `index.html`, `app.js` — giao diện; `worker.js` — chạy model; `text.js` — chia câu
- `zh/` — chữ Hán → âm vị cho Kokoro v1.1-zh (bản JavaScript của misaki ZHFrontend + jieba posseg + pypinyin/pypinyin-dict + cn2an, khớp bản Python trên 6.121 câu thử)
- `config.json` — giọng Nam/Nữ của từng tiếng (sửa ở đây để đổi giọng)
- `models/` — model chia thành phần ≤ 45 MB (`tools/build_models.py`)
- `licenses.html`, `LICENSES/` — giấy phép; giọng là giọng tổng hợp do máy tạo (OpenRAIL-M, Apache 2.0)
