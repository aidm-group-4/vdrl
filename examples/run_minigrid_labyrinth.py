import sys
import os
import json
import copy

import numpy as np

from vdrl.high_level import MetaController, MetaEnvironment
from vdrl.config import ExperimentConfig, LowerLevelPolicies, VerificationMethod
from vdrl.environment import MiniGridLabyrinth
from vdrl.controller import EmpericalSubController, ReachAvoidSubController
from vdrl.model import PPOModel, ViperModel

# High level decomposition of the example in the paper figure 2
minigrid_test_json_file = os.path.abspath("examples/minigrid_test.json")

# number of rollouts (episodes) for emperically testing the meta controller
meta_controller_n_steps_per_rollout = 200

# number of steps of each test episode
num_rollouts = 300

# Select if controllers should be emperically evaluated while training (can turn this off to make training faster).
do_emperical_eval = True


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
    # raise ValueError(f'Could not find a high level state matching the observation: {obs}')


# problem specific stuff
def load_test_problem(json_file):
    with open(json_file, "r") as f:
        exp_json = json.load(f)
    exp_config = ExperimentConfig(**exp_json)

    # Construct meta-environment
    env_config = exp_config.env_config
    meta_environment = MetaEnvironment.from_env_config(env_config)

    # Construct the environment
    mapping_config = exp_config.mapping_config

    # TODO: interpret these as tuples with pydantic
    # hl_state_map = mapping_config.high_level_states
    hl_state_map = []
    for hl_state in mapping_config.high_level_states:
        states = []
        for state in hl_state:
            states.append(tuple(state))
        hl_state_map.append(states)

    verification_config = exp_config.verification_config

    start_states = []
    for state in env_config.start_states:
        start_states += [
            tuple(minigrid_state) for minigrid_state in hl_state_map[state]
        ]

    # TODO: what is the difference between reach and goal states?
    goal_states = []
    for state in env_config.goal_states:
        goal_states += [tuple(minigrid_state) for minigrid_state in hl_state_map[state]]

    fail_states = []
    for state in env_config.avoid_states:
        fail_states += [tuple(minigrid_state) for minigrid_state in hl_state_map[state]]

    # Create function mapping observations to high-level state
    def obs_mapping(obs):
        return minigrid_obs_mapping_test(obs, hl_state_map)

    # Create list of controllers with PPO models
    # TODO: maybe move settings to a separate item in the config to avoid this awkward deletion?
    llp_config = exp_config.llp_config
    llp_name = llp_config.llp_name
    llp_config = llp_config.dict(exclude={"llp_name"})

    controller_list = []
    for edge in env_config.decompositions:
        init_states = []
        for state in edge.initial_states:
            init_states += [
                tuple(minigrid_state) for minigrid_state in hl_state_map[state]
            ]

        final_states = [
            tuple(minigrid_state) for minigrid_state in hl_state_map[edge.final_state]
        ]
        avoid_states = fail_states

        # Make a smaller environment just for this task
        # TODO: maybe move this into the controller definition?
        environment = MiniGridLabyrinth(
            init_states, final_states, exp_config.problem_config, avoid_states
        )

        if llp_name == LowerLevelPolicies.PPO:
            model = PPOModel(environment._env, **llp_config)

        else:
            raise NotImplementedError(
                f"The given lower level policy {llp_name} is currently not supported."
            )
        if verification_config.verification_method == VerificationMethod.EMPERICAL:
            controller = EmpericalSubController(
                environment=environment,
                model=model,
                init_states=init_states,
                final_states=final_states,
                avoid_states=avoid_states,
            )
        elif verification_config.verification_method == VerificationMethod.VIPER:
            viper_model = ViperModel(
                environment=environment, model=model, viper_config=verification_config
            )
            controller = ReachAvoidSubController(
                environment=environment,
                model=viper_model,
                init_states=init_states,
                final_states=final_states,
                avoid_states=avoid_states,
                p_prune=0.0000000001,
            )
        elif verification_config.verification_method == VerificationMethod.EXHAUSTIVE:
            controller = ReachAvoidSubController(
                environment=environment,
                model=model,
                init_states=init_states,
                final_states=final_states,
                avoid_states=avoid_states,
                p_prune=0.0000000001,
            )
        controller_list.append(controller)

    # Whole environment
    environment = MiniGridLabyrinth(
        start_states, goal_states, exp_config.problem_config, fail_states
    )

    return environment, meta_environment, controller_list, obs_mapping


# Run the labyrinth navigation experiment.
def main(args):
    # load problem (TODO: swap out the test function later)
    # TODO: Select the problem loading function based on the problem config type?
    environment, meta_environment, controller_list, obs_mapping = load_test_problem(
        minigrid_test_json_file
    )

    if len(args) > 1:
        # load a meta-controller which has already been trained
        load_dir = args[1]
        meta_controller = MetaController.load(
            load_dir, ReachAvoidSubController.load, obs_mapping, environment
        )
    else:
        # Construct a meta-controller and train it
        meta_controller = MetaController(meta_environment, controller_list, obs_mapping)
        meta_controller.learn(
            environment, evaluate_controllers_progress=do_emperical_eval
        )

    print(
        "\nStored performance results (empirical rollouts tend to use more timesteps)"
    )
    performance_results = meta_controller.get_performance_results()
    print(
        f"Stored final results: \n"
        f"Verified lower bound: {performance_results[-1][1]:0.3f}\n"
        f"Empirical estimate: {performance_results[-1][2]:0.3f}\n"
    )

    # Once learning has finished, manually run the formal and empirical performance again
    print("Verifying system...")
    reach_prob = meta_controller.verify_performance()
    print(f"Verified lower bound: {reach_prob:0.3f}\n")

    print("Evaluating performance of meta controller")
    success_count, avg_success_steps, total_steps = meta_controller.eval_performance(
        environment,
        n_episodes=num_rollouts,
        n_steps=meta_controller_n_steps_per_rollout,
    )
    print(
        f"Empirically measured success prob: {success_count / num_rollouts:0.3f}\n"
        f"Average steps to reach goal: {avg_success_steps:0.3f}"
    )

    # Visualize performance
    n_episodes = 5
    n_steps = 200
    render = False
    meta_controller.demonstrate_capabilities(
        environment, n_episodes=n_episodes, n_steps=n_steps, render=render
    )


if __name__ == "__main__":
    main(sys.argv)
