"""Render observed F10 training logs with the bundled plotting runtime."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

base = Path(__file__).resolve().parent
rows = json.loads((base / "parsed_training.json").read_text(encoding="utf-8"))["F10"]
fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
series = [("Mean action std", "Exploration standard deviation"),
          ("Curriculum/terrain_levels", "Mean terrain level"),
          ("Episode_Termination/base_contact", "Training base-contact termination fraction"),
          ("Mean value loss", "Critic loss (log scale)")]
for ax, (tag, label) in zip(axes.flat, series):
    x = np.arange(1000, 4000)
    y = np.array([r[tag] for r in rows[1000:]])
    ax.plot(x, y, color="#4f76a6", alpha=.25, lw=.6, label="Iteration log")
    med = np.array([np.median(y[i:i+100]) for i in range(0, len(y), 100)])
    ax.plot(np.arange(1050, 4000, 100), med, color="#c34b38", lw=2, label="100-iteration median")
    ax.axvspan(2000, 2800, color="#cbe4d3", alpha=.25)
    ax.set_title(label)
    ax.set_xlabel("Training iteration")
    ax.grid(alpha=.18)
    if tag == "Mean value loss":
        ax.set_yscale("log")
axes[0, 0].legend(frameon=False)
fig.suptitle("F10: later training does not show uniform improvement", fontsize=15)
fig.savefig(base / "training_diagnostics.png", dpi=150)
print(base / "training_diagnostics.png")
