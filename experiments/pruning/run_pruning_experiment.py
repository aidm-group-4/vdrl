import sys
import os
import json
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

from vdrl.high_level import MetaController
from vdrl.config import ExperimentConfig
from vdrl.environment import MiniGridLabyrinth
from vdrl.controller import ReachAvoidSubController

CURRENT_WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
RESULTS_DIR = os.path.join(CURRENT_WORKING_DIR, "results")
CONFIG_FILE = os.path.join(CURRENT_WORKING_DIR, "config.json")


def plot_line_graph(x_axis, y_axis, stds, title, y_label, zero_value=None):
    plt.figure(figsize=(10, 6))

    # Plot the main data
    plt.plot(x_axis, y_axis, label=y_label, color="blue")

    upper_bound = [y + s for y, s in zip(y_axis, stds)]
    lower_bound = [y - s for y, s in zip(y_axis, stds)]

    if zero_value:
        upper_bound = list(map(lambda x: min(x, 1.0), upper_bound))
        plt.axhline(
            y=zero_value, color="red", linestyle=":", label=f"{y_label} without pruning"
        )

    # Plot the variance lines
    plt.plot(x_axis, upper_bound, color="green", linestyle="--")
    plt.plot(x_axis, lower_bound, color="green", linestyle="--")

    # Add labels and title
    plt.xlabel("Pruning Thresholds", fontsize=18)
    plt.ylabel(y_label, fontsize=18)
    plt.title(title, fontsize=20)

    plt.xscale("log")
    plt.xticks(x_axis, [f"{x:.0e}" for x in x_axis])

    if zero_value:
        plt.legend(fontsize=18)

    # Show the plot
    plt.savefig(os.path.join(RESULTS_DIR, f"{title}.png"))


def minigrid_obs_mapping_test(obs, hl_state_map):
    """Map high level states to minigrid states

    Inputs
    ------
    obs : np.array
        An observation from the environment.
    hl_state_map : dict
        A map from the observation to higher level state index.

    Outputs
    -------
    hl_state : int
        The high level state.
    """
    observation = tuple(obs)
    for i, states in enumerate(hl_state_map):
        if observation in states:
            return i

    return None


def load_test_problem(json_file):
    with open(json_file, "r") as f:
        exp_json = json.load(f)
    exp_config = ExperimentConfig(**exp_json)

    # Construct meta-environment
    env_config = exp_config.env_config

    # Construct the environment
    mapping_config = exp_config.mapping_config

    hl_state_map = []
    for hl_state in mapping_config.high_level_states:
        states = []
        for state in hl_state:
            states.append(tuple(state))
        hl_state_map.append(states)

    start_states = []
    for state in env_config.start_states:
        start_states += [
            tuple(minigrid_state) for minigrid_state in hl_state_map[state]
        ]

    goal_states = []
    for state in env_config.goal_states:
        goal_states += [tuple(minigrid_state) for minigrid_state in hl_state_map[state]]

    fail_states = []
    for state in env_config.avoid_states:
        fail_states += [tuple(minigrid_state) for minigrid_state in hl_state_map[state]]

    def obs_mapping(obs):
        return minigrid_obs_mapping_test(obs, hl_state_map)

    environment = MiniGridLabyrinth(
        start_states, goal_states, exp_config.problem_config, fail_states
    )

    return environment, obs_mapping


def main(args):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    pruning_thresholds = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0]
    environment, obs_mapping = load_test_problem(CONFIG_FILE)

    experiment_data = []

    for p_prune in pruning_thresholds:
        # load a meta-controller which has already been trained
        prune_data = {
            "p_prune": p_prune,
            "max_states": [],
            "transition_function_calls": [],
            "lower_bounds": [],
        }

        # Pass the path to a trained meta-policy
        load_dir = args[1]
        controller_load_fn = partial(ReachAvoidSubController.load, p_prune=p_prune)
        meta_controller = MetaController.load(
            load_dir, controller_load_fn, obs_mapping, environment
        )

        print(f"Verifying system with p_prune {p_prune}")
        reach_prob = meta_controller.verify_performance()
        print(f"Verified lower bound: {reach_prob:0.3f}\n")

        for controller in meta_controller.controller_list[:-1]:
            if controller.verification_data["reachability"]["max_states"] > 0:
                prune_data["max_states"].append(
                    controller.verification_data["reachability"]["max_states"]
                )
                prune_data["transition_function_calls"].append(
                    controller.verification_data["reachability"][
                        "transition_function_calls"
                    ]
                )
                prune_data["lower_bounds"].append(
                    float(controller.guarantees["reachability"])
                )

        experiment_data.append(prune_data)

    x_axis = []
    y_axis = []
    stds = []

    zero_pruning_data = experiment_data[-1]
    zero_values = (
        np.mean(zero_pruning_data["max_states"]),
        np.mean(zero_pruning_data["transition_function_calls"]),
        np.mean(zero_pruning_data["lower_bounds"]),
    )

    for prune_data in experiment_data[:-1]:
        p_prune = prune_data["p_prune"]
        max_states = prune_data["max_states"]
        transition_function_calls = prune_data["transition_function_calls"]
        lower_bounds = prune_data["lower_bounds"]

        x_axis.append(p_prune)
        y_axis.append(
            (
                np.mean(max_states),
                np.mean(transition_function_calls),
                np.mean(lower_bounds),
            )
        )
        stds.append(
            (
                np.std(max_states),
                np.std(transition_function_calls),
                np.std(lower_bounds),
            )
        )

    plot_line_graph(
        x_axis=x_axis,
        y_axis=[y[0] for y in y_axis],
        stds=[s[0] for s in stds],
        title="Space Consumption vs Pruning Threshold",
        y_label="Space Consumption",
    )

    plot_line_graph(
        x_axis=x_axis,
        y_axis=[y[1] for y in y_axis],
        stds=[s[1] for s in stds],
        title="Transition Function Calls vs Pruning Threshold",
        y_label="Transition Function Calls",
    )

    plot_line_graph(
        x_axis=x_axis,
        y_axis=[y[2] for y in y_axis],
        stds=[s[2] for s in stds],
        title="Verifiable Lower Bound vs Pruning Threshold",
        y_label="Verifiable Lower Bound",
        zero_value=zero_values[2],
    )


if __name__ == "__main__":
    main(sys.argv)
