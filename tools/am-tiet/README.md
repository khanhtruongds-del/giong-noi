# Bảng âm tiết pinyin (Qwen3-TTS 1.7B CustomVoice, giọng Serena + Aiden)

Chạy trên máy (không nằm trong trang web). 394 âm gốc × các thanh có trong `syl_table.json` (1.121 thanh) × 2 giọng.

- `chay_bang_pinyin.command` → `syl_full.py`: đọc mỗi âm gốc thành một chuỗi các thanh ("妈，麻，马，骂。"), cắt theo chỗ ngắt,
  chấm theo đường cao độ (pyin), thử tối đa 8 lần, giữ lần tốt nhất. Ra `ket-qua-full/{giọng}_{âm}{thanh}.wav` + `scores.json`.
- `syl_redo.py`: chạy lại các âm chưa đạt (bf16, 8 lượt một lần; `ROUNDS=1` = tối đa 8 lần thử). Chỉ giữ bản điểm cao hơn. Ra `ket-qua-redo/`.
- `syl_fix2.py`: 2 âm khó của Aiden — nan dùng 难/腩 thay 男/赧; wen đọc từng chữ lặp đôi ("温，温。") rồi lấy đoạn đầu.
- `gop_ket_qua.py <đường dẫn ket-qua-full>`: gộp `ket-qua-redo` vào `ket-qua-full` (cất bản cũ vào `_cu/`, `scores.before-redo.json`).

Kết quả 29/10/2026 (sau chạy lại): Serena 362/394 âm đúng hết thanh (1.086/1.121 thanh), Aiden 300/394 (1.021/1.121); đủ file cho cả 394 âm × 2 giọng.
WAV gốc ở máy (qwen-am-tiet/ket-qua-full). Bản dùng trên trang: `to_mp3.py <ket-qua-full> ../../am-tiet` → `am-tiet/{serena,aiden}/{âm}{thanh}.mp3`
(cân âm lượng RMS −20 dBFS, vuốt đầu/cuối 8 ms) + `am-tiet/index.json` (giọng theo giới, thanh có của từng âm).
