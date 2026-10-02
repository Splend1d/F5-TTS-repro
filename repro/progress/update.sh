#!/bin/bash
# Refresh docs/metrics.json and push it to GitHub Pages if it changed. Run by f5-progress.timer (login node).
set -euo pipefail
source /data/group_data/UTD-NAS/chanjanh/speechgen/F5-TTS/repro/env.sh >/dev/null
python repro/progress/export.py
# ignore fields that change on every export, so an idle run doesn't add a commit every 6 h
strip='import json,sys;d=json.load(open(sys.argv[1]));d.pop("generated_at");d["progress"].pop("seconds_since_last_log");print(json.dumps(d,sort_keys=True))'
if git cat-file -e HEAD:docs/metrics.json 2>/dev/null &&
   [ "$(python -c "$strip" docs/metrics.json)" = "$(git show HEAD:docs/metrics.json > /tmp/f5m.$$ && python -c "$strip" /tmp/f5m.$$; rm -f /tmp/f5m.$$)" ]; then
  git checkout -- docs/metrics.json; echo "no change"; exit 0
fi
git add docs/metrics.json
git commit -q -m "progress: update $(python -c "import json;print(json.load(open('docs/metrics.json'))['progress']['update'])")" -- docs/metrics.json
git push -q fork HEAD:main
echo "pushed"
