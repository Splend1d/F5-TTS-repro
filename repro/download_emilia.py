"""Download Emilia ZH+EN (pinned rev fc71e07, the folder format prepare_emilia.py expects) and extract.
Idempotent: a marker per shard in raw/.done; safe to kill / requeue at any point."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import HfApi, hf_hub_download


REPO, REV = "amphion/Emilia-Dataset", "fc71e07e8572f5f3be1dbd02ed3172a4d298f152"
ROOT = "/data/group_data/UTD-NAS/chanjanh/speechgen/data_raw/Emilia"
TARS, RAW, DONE = f"{ROOT}/tars", f"{ROOT}/raw", f"{ROOT}/raw/.done"
LANGS = ("ZH", "EN")
os.makedirs(DONE, exist_ok=True)

files = [s.rfilename for s in HfApi().dataset_info(REPO, revision=REV).siblings]
shards = {}  # shard name -> list of parts
for f in files:
    lang = f.split("/")[0]
    if lang in LANGS:
        name = os.path.basename(f).split(".tar.gz")[0]
        shards.setdefault(name, []).append(f)


def run(cmd):
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash")


def fetch(f):
    return hf_hub_download(REPO, f, repo_type="dataset", revision=REV, local_dir=TARS)


def do_shard(name, parts):
    if os.path.exists(f"{DONE}/{name}"):
        return name, "skip"
    lang = name[:2]
    paths = [fetch(p) for p in sorted(parts)]
    os.makedirs(f"{RAW}/{lang}", exist_ok=True)
    run(f"rm -rf {RAW}/{lang}/{name}")  # clean partial extraction
    run(f"cat {' '.join(paths)} | tar -xz -C {RAW}/{lang}")
    open(f"{DONE}/{name}", "w").close()
    for p in paths:
        os.remove(p)  # tar no longer needed (re-downloadable)
    return name, "ok"


# metadata jsonl
if not os.path.exists(f"{DONE}/openemilia_all"):
    p = fetch("openemilia_all.tar.gz")
    run(f"tar -xzf {p} -C {RAW} --wildcards 'ZH/*.jsonl' 'EN/*.jsonl'")
    open(f"{DONE}/openemilia_all", "w").close()

nworkers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
todo = sorted(shards.items(), key=lambda kv: kv[0])
print(f"{len(todo)} shards, {sum(os.path.exists(f'{DONE}/{n}') for n, _ in todo)} already done", flush=True)
with ThreadPoolExecutor(nworkers) as ex:
    futs = [ex.submit(do_shard, n, p) for n, p in todo]
    for i, fu in enumerate(as_completed(futs)):
        try:
            print(i, *fu.result(), flush=True)
        except Exception as e:
            print("FAIL", repr(e), flush=True)
missing = [n for n, _ in todo if not os.path.exists(f"{DONE}/{n}")]
print("MISSING", missing, flush=True)
sys.exit(1 if missing else 0)
