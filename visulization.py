import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# Configuration
# --------------------------------------------------------

benchmarks = [
    "QEC small\n(helper-intensive)",
    "Classical logic\nreversible",
    "Multi-control X\nCⁿ(X)",
    "LCU ZZX\nHamiltonian",
]

scales = ["Small", "Medium"]   # n≈10, n≈20
x = np.arange(len(scales))

algorithms = [
    "IBM Quantum",
    "HyperQ",
    "HALO(No Sharing)",
    "HALO(Shot-unaware)",
    "HALO",
]

num_algs = len(algorithms)
bar_width = 0.15

# --------------------------------------------------------
# >>> Replace these with your real numbers (2 values per list)
# --------------------------------------------------------

fidelity = {
    "QEC small\n(helper-intensive)": {
        "IBM Quantum":        [0.95, 0.93],
        "HyperQ":             [0.96, 0.94],
        "HALO(No Sharing)":   [0.96, 0.94],
        "HALO(Shot-unaware)": [0.97, 0.95],
        "HALO":               [0.97, 0.96],
    },
    "Classical logic\nreversible": {
        "IBM Quantum":        [0.99, 0.98],
        "HyperQ":             [0.99, 0.98],
        "HALO(No Sharing)":   [0.99, 0.98],
        "HALO(Shot-unaware)": [0.99, 0.98],
        "HALO":               [0.99, 0.98],
    },
    "Multi-control X\nCⁿ(X)": {
        "IBM Quantum":        [0.98, 0.96],
        "HyperQ":             [0.98, 0.96],
        "HALO(No Sharing)":   [0.98, 0.96],
        "HALO(Shot-unaware)": [0.98, 0.97],
        "HALO":               [0.99, 0.97],
    },
    "LCU ZZX\nHamiltonian": {
        "IBM Quantum":        [0.96, 0.94],
        "HyperQ":             [0.96, 0.94],
        "HALO(No Sharing)":   [0.96, 0.94],
        "HALO(Shot-unaware)": [0.97, 0.95],
        "HALO":               [0.97, 0.95],
    },
}

# Waiting time in seconds
waiting_time = {
    "QEC small\n(helper-intensive)": {
        "IBM Quantum":        [40.0, 70.0],
        "HyperQ":             [32.0, 55.0],
        "HALO(No Sharing)":   [28.0, 50.0],
        "HALO(Shot-unaware)": [25.0, 42.0],
        "HALO":               [20.0, 35.0],
    },
    "Classical logic\nreversible": {
        "IBM Quantum":        [25.0, 45.0],
        "HyperQ":             [20.0, 38.0],
        "HALO(No Sharing)":   [18.0, 35.0],
        "HALO(Shot-unaware)": [15.0, 30.0],
        "HALO":               [12.0, 25.0],
    },
    "Multi-control X\nCⁿ(X)": {
        "IBM Quantum":        [30.0, 60.0],
        "HyperQ":             [26.0, 52.0],
        "HALO(No Sharing)":   [24.0, 48.0],
        "HALO(Shot-unaware)": [21.0, 43.0],
        "HALO":               [18.0, 36.0],
    },
    "LCU ZZX\nHamiltonian": {
        "IBM Quantum":        [50.0, 90.0],
        "HyperQ":             [42.0, 78.0],
        "HALO(No Sharing)":   [38.0, 70.0],
        "HALO(Shot-unaware)": [34.0, 62.0],
        "HALO":               [28.0, 55.0],
    },
}

# --------------------------------------------------------
# Plotting helper
# --------------------------------------------------------

def plot_metric(metric_name, metric_data, ylabel, save_as=None):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3), sharey=True)
    fig.suptitle(metric_name, fontsize=18, y=1.05)

    for b_idx, bench in enumerate(benchmarks):
        ax = axes[b_idx]

        for a_idx, alg in enumerate(algorithms):
            offset = (a_idx - (num_algs - 1) / 2) * bar_width
            y_vals = metric_data[bench][alg]

            ax.bar(
                x + offset, y_vals, width=bar_width,
                label=alg if b_idx == 0 else None
            )

        ax.set_title(bench, fontsize=14, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(scales, fontsize=13)
        ax.tick_params(axis='y', labelsize=13)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        if b_idx == 0:
            ax.set_ylabel(ylabel, fontsize=14)

    # Shared legend for all subplots
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=5,
        fontsize=13,
        bbox_to_anchor=(0.5, 1.15)
    )

    fig.tight_layout()
    if save_as is not None:
        plt.savefig(save_as, bbox_inches="tight", dpi=300)

# --------------------------------------------------------
# Create both figures
# --------------------------------------------------------

plot_metric(
    "Fidelity vs. Problem Size",
    fidelity,
    ylabel="Fidelity",
    save_as="fidelity.pdf",
)

plot_metric(
    "Waiting Time vs. Problem Size",
    waiting_time,
    ylabel="Waiting time (s)",
    save_as="waiting_time.pdf",
)

# If running interactively:
# plt.show()
