#!/bin/bash
# Refresh docs/metrics.json and push it to GitHub Pages if it changed. Run by f5-progress.timer (login node).
set -euo pipefail
source /data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS/repro/env.sh >/dev/null
python repro/progress/export.py
git diff --quiet -- docs/metrics.json && { echo "no change"; exit 0; }
git add docs/metrics.json
git commit -q -m "progress: update $(python -c "import json;print(json.load(open('docs/metrics.json'))['progress']['update'])")" -- docs/metrics.json
git push -q fork HEAD:main
echo "pushed"
