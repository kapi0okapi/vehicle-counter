#!/bin/bash
# Pre-fetch YOLO weights and the fonts Ultralytics uses for plots. Run on a login node:
# compute nodes may not have internet access, and the jobs run with YOLO_OFFLINE=1.
set -euo pipefail
python - <<'PY'
from ultralytics import YOLO
from ultralytics.utils.checks import check_font

for name in ("yolo26n.pt", "yolo26s.pt", "yolo26m.pt"):
    YOLO(name)
    print("cached", name)
check_font("Arial.ttf")
check_font("Arial.Unicode.ttf")
PY
