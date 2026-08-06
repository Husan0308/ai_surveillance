#!/bin/bash
set -e

echo "=== AI Surveillance Auto-Installation & Setup ==="

# 1. Virtual environment yaratish
if [ ! -d "venv" ]; then
    echo "[1/3] Virtual Environment (venv) yaratilmoqda..."
    python3 -m venv venv
else
    echo "[1/3] Existing venv topildi."
fi

# 2. Pip va setup installyatorlarini yangilash
echo "[2/3] Kerakli kutubxonalar o'rnatilmoqda (requirements.txt)..."
./venv/bin/python3 -m pip install --upgrade pip setuptools wheel

# 3. Requirements.txt ni bitta buyruq bilan o'rnatish
./venv/bin/python3 -m pip install -r requirements.txt

echo "=== ✅ BARCHA KUTUBXONALAR MUVAFFAQIYATLI O'RNATILDI! ==="
echo "Dasturni ishga tushirish uchun: ./venv/bin/python3 main.py"
