import os
import glob
from vdrl.utils.results_saver import Results
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CURRENT_WORKING_DIR = os.path.dirname(os.path.realpath(__file__))

RUN_DIR = os.path.join(CURRENT_WORKING_DIR, "runs")
RESULTS_DIR = os.path.join(CURRENT_WORKING_DIR, "results")


def plot_line_graph(results):
    plt.figure(figsize=(10, 6))
    colors = ["blue", "green", "red"]

    fig, ax = plt.subplots()

    for index, result in enumerate(results):
        steps = result["steps"]
        lower_bounds = result["lower_bounds"]
        empirical = result["empirical"]
        color = colors[index]

        ax.plot(
            steps,
            lower_bounds,
            linestyle="-",
            color=color,
            label=f"Run {index + 1} - Lower Bound",
        )
        ax.plot(
            steps,
            empirical,
            linestyle="--",
            color=color,
            label=f"Run {index + 1} - Empirical Performance",
        )

    legend_elements = [
        Line2D([0], [0], color="blue", lw=2, label="Run 1"),
        Line2D([0], [0], color="green", lw=2, label="Run 2"),
        Line2D([0], [0], color="red", lw=2, label="Run 3"),
        Line2D(
            [0], [0], color="black", lw=2, linestyle="--", label="Empirical Performance"
        ),
        Line2D(
            [0], [0], color="black", lw=2, linestyle="-", label="Verified Lower Bound"
        ),
    ]

    ax.legend(handles=legend_elements, fontsize=10.5, frameon=True, framealpha=0.7)

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Performance", fontsize=12)
    ax.set_title("Emperical Performance vs Verified Lower Bound", fontsize=15)
    ax.grid(True)

    plt.savefig(os.path.join(RESULTS_DIR, f"emperical_vs_formal_traces.png"))


def main():
    assert os.path.exists(
        RUN_DIR
    ), f"The run directory {RUN_DIR} should exist and contain the trained policies for the experiment."

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    results_list = []

    for run_path in glob.glob(os.path.join(RUN_DIR, "*")):
        results: Results = Results(load_dir=run_path)

        steps_for_run = list(results.data["composition_policy_lower_bound"].keys())

        verifiable_lower_bounds = list(
            results.data["composition_policy_lower_bound"].values()
        )

        success_counts = [
            val[0] for val in list(results.data["composition_rollout_mean"].values())
        ]
        num_rollouts = list(results.data["composition_num_rollouts"].values())

        empirical_performance = [s / r for (s, r) in zip(success_counts, num_rollouts)]

        results_list.append(
            {
                "steps": steps_for_run,
                "lower_bounds": verifiable_lower_bounds,
                "empirical": empirical_performance,
            }
        )

    plot_line_graph(results=results_list)


if __name__ == "__main__":
    main()
