"""NCCL sanity check: init + 1.36 GB allreduce (~ one F5 gradient sync). Launched by repro/diag/nccl_test.sbatch."""

import datetime
import os
import time

import torch
import torch.distributed as dist


r, w, lr = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(lr)
t = time.time()
dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=90), device_id=torch.device("cuda", lr))
x = torch.ones(340_000_000, device="cuda", dtype=torch.float32)
dist.all_reduce(x)
torch.cuda.synchronize()
t1 = time.time()
for _ in range(3):
    dist.all_reduce(x)
torch.cuda.synchronize()
print(
    f"rank {r}: init+first {t1 - t:.1f}s, allreduce 1.36GB {(time.time() - t1) / 3:.3f}s, ok={x[0].item() == w**4}",
    flush=True,
)
dist.destroy_process_group()
