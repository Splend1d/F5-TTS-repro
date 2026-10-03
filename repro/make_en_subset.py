"""Cut the EN-only dataset out of the prepared ZH+EN one (same text processing and official vocab).

data/Emilia_ZH_EN_pinyin/raw.arrow stores all ZH rows before all EN rows, so EN is one contiguous tail range.
Writes data/Emilia_EN_pinyin/{raw.arrow,duration.json,vocab.txt}; every row is checked to be under raw/EN/.
"""

import json
import os
import shutil

from datasets import Dataset
from datasets.arrow_writer import ArrowWriter
from tqdm import tqdm


SRC, DST = "data/Emilia_ZH_EN_pinyin", "data/Emilia_EN_pinyin"
d = Dataset.from_file(f"{SRC}/raw.arrow")
n = len(d)
lo, hi = 0, n  # first EN row
while lo < hi:
    m = (lo + hi) // 2
    if "/raw/EN/" in d[m]["audio_path"]:
        hi = m
    else:
        lo = m + 1
assert lo > 0 and "/raw/EN/" not in d[lo - 1]["audio_path"]
print(f"EN rows {lo}..{n - 1} ({n - lo})")

os.makedirs(DST, exist_ok=True)
durations = []
with ArrowWriter(path=f"{DST}/raw.arrow.tmp", writer_batch_size=100_000) as writer:
    for start in tqdm(range(lo, n, 100_000), mininterval=60):
        batch = d[start : min(start + 100_000, n)]
        assert all("/raw/EN/" in p for p in batch["audio_path"]), f"non-EN row in batch at {start}"
        writer.write_batch(batch)
        durations.extend(batch["duration"])
    writer.finalize()
os.replace(f"{DST}/raw.arrow.tmp", f"{DST}/raw.arrow")
with open(f"{DST}/duration.json", "w") as f:
    json.dump({"duration": durations}, f, ensure_ascii=False)
shutil.copy(f"{SRC}/vocab.txt", f"{DST}/vocab.txt")
print(f"wrote {DST}: {len(durations)} utts, {sum(durations) / 3600:.1f} h")
