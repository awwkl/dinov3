import os
import re
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = 'babyview/outputs/grad_accum_4/logs'
LOG_PATH = os.path.join(LOG_DIR, "log.txt")
BIN_SIZE = 5000

# Matches lines like:
# Training  [124999/125000] ... total_loss: 9.3104 (9.5936) ...
pat = re.compile(
    r"Training\s+\[\s*(\d+)\s*/\s*(\d+)\s*\].*?"
    r"total_loss:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*"
    r"\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)"
)

iters = []
losses = []

USE_RUNNING_AVG = False  # set True to use the value in parentheses

with open(LOG_PATH, "r", errors="ignore") as f:
    for line in f:
        m = pat.search(line)
        if not m:
            continue
        it = int(m.group(1))
        cur = float(m.group(3))
        avg = float(m.group(4))
        iters.append(it)
        losses.append(avg if USE_RUNNING_AVG else cur)
        print('', it, cur, avg)

iters = np.asarray(iters, dtype=np.int64)
losses = np.asarray(losses, dtype=np.float64)

# sort (just in case logs are interleaved)
order = np.argsort(iters)
iters, losses = iters[order], losses[order]

# bin by iter//BIN_SIZE
bin_ids = iters // BIN_SIZE
uniq = np.unique(bin_ids)

bin_centers = uniq * BIN_SIZE + (BIN_SIZE / 2.0)
bin_means = np.array([losses[bin_ids == k].mean() for k in uniq])

plt.figure()
plt.plot(bin_centers, bin_means, marker="o")
plt.xlabel(f"Iteration (binned, {BIN_SIZE} iters)")
plt.ylabel("total_loss (mean within bin)")
plt.title("DINO total_loss vs iteration")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(LOG_DIR, "loss_plot.png"))