#!/bin/bash
# Tạo cả bảng âm tiết pinyin (Serena + Aiden). Chạy lại được: âm đã xong sẽ bỏ qua. Không tắt máy/không cho máy ngủ khi đang chạy.
cd "$(dirname "$0")/qwen-am-tiet" || exit 1
caffeinate -i .venv/bin/python -u syl_full.py 2>&1 | grep --line-buffered -v -i "warn\|pad_token" | tee log_full.txt
echo "XONG — có thể đóng cửa sổ này."
