import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

schedules = ["guided", "static", "dynamic"]

fig, axes = plt.subplots(1, 3, figsize=(20, 8), sharey=True)

for ax, sched in zip(axes, schedules):
    subset = df[df["Schedule"] == sched]
    
    for chunk, group in subset.groupby("Chunk"):
        ax.plot(group["Threads"], group["Time"], marker='o', label=f"Chunk={chunk}")
    
    ax.set_title(f"Schedule: {sched}")
    ax.set_xlabel("Número de Threads")
    ax.grid(True)

axes[0].set_ylabel("Tiempo de ejecución (s)")

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Chunk", loc="upper center", ncol=6)

plt.tight_layout(rect=[0, 0, 1, 0.92])

plt.savefig("openmp_all_schedules.png", dpi=300)

plt.close()
