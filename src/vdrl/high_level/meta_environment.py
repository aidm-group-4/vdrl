import os
import pickle

import numpy as np
from typing import Any, List, Dict

from vdrl.environment import Environment
from vdrl.config import EnvironmentConfig


class MetaEnvironment(Environment):
    """
    Class representing the decomposition graph of the high-level decision
    making process.
    """

    def __init__(
        self,
        state_space: List[int],
        action_space: List[int],
        start_state: int,
        goal_state: int,
        fail_state: int,
        successor_map: Dict,
    ):
        """
        Inputs
        ------
        state_space : list[int]
            State space of the high-level MDP where each high-level state is
            represented by an integer. Must contain the start state, goal state,
            fail state, and the entry points of each controller.
        action_space : list[int]
            Action space of the high-level MDP where each high-level action
            (sub-task controller) is represented by an integer.
        start_state : int
            Integer representation of the initial state in the high-level MDP.
        goal_state : int
            Integer representation of the goal state in the high-level MDP.
        fail_state : int
            Integer representation of the abstract high-level failure state.
        successor_map : dict
            Dictionary mapping high-level state-action pairs to the next
            high-level state.
        """
        super().__init__(
            [start_state], [goal_state], [fail_state], state_space, action_space
        )

        self.successor = successor_map
        self.current_state = start_state

    def save(self, save_dir: str):
        """
        Save the MetaEnvironment object.

        Inputs
        ------
        save_dir : string
            Absolute path to the directory that will be used to save this MetaEnvironment.
        """
        # create save directory if it doesn't exist
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        meta_env_file = os.path.join(save_dir, "meta_environment.p")
        meta_env_data = {
            # TODO: add whatever data we think is important here
            "state_space": self.state_space,
            "action_space": self.action_space,
            "start_state": self.start_states[0],
            "goal_state": self.goal_states[0],
            "fail_state": self.fail_states[0],
            "successor_map": self.successor,
        }

        with open(meta_env_file, "wb") as pickle_file:
            pickle.dump(meta_env_data, pickle_file)

    def load(load_dir: str):
        """
        Load the MetaEnvironment object.

        Inputs
        ------
        load_dir : string
            Absolute path to the directory that will be used to load this MetaEnvironment.
        """
        meta_env_file = os.path.join(load_dir, "meta_environment.p")
        with open(meta_env_file, "rb") as pickle_file:
            meta_env_data = pickle.load(pickle_file)

        return MetaEnvironment(
            meta_env_data["state_space"],
            meta_env_data["action_space"],
            meta_env_data["start_state"],
            meta_env_data["goal_state"],
            meta_env_data["fail_state"],
            meta_env_data["successor_map"],
        )

    def step(self, action, slip_prob=0.0):
        """
        Simulates a step in the meta-environement with some probability of failure.

        Parameters
        ----------
        action : List[int]
            The controller that is used to get to the next sub task.

        slip_prob : float, optional
            A probability that the controller fails and the state transitions
            to a fail state. If not provided, this defaults to 0.

        Returns
        -------
        observation : [int]
            The next high level state.
        reward : SupportsFloat
            The reward as a result of taking the action. A reward of 1.0 if a
            goal state is reached, otherwise 0.0.
        terminated : bool
            Whether the agent reached a goal or fail state. If true, the user
            needs to call reset() before using the environment again.
        truncated : bool
            Unused, this always returns False.
        info : dict
            Unused for now, this always returns {}.
        """
        if np.random.uniform() < slip_prob:
            self.current_state = self.fail_states[0]
        else:
            self.current_state = self.successor[(self.current_state, action[0])]

        obs = self.current_state
        reward = 1.0 if self.current_state in self.goal_states else 0.0
        terminated = (
            self.current_state in self.goal_states
            or self.current_state in self.fail_states
        )

        # TODO: would we have a use for either of these?
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info

    def reset(self):
        """
        Reset the meta-environment.

        Outputs
        -------
        obs : Any
            Observation of the starting state
        info : {}
            Empty dict to match the expected Gymnasium reset output format.
        """
        self.current_state = self.start_states[0]
        return self.current_state, {}

    def render(self):
        """
        Render the MetaEnvironment as a directed graph.
        """
        raise (NotImplementedError)

    @classmethod
    def from_env_config(cls, env_config: EnvironmentConfig):
        """
        Create a MetaEnvironment from the environment configuration.
        """
        # TODO: what happens if these are not length 1? Maybe use set methods
        assert len(env_config.start_states) == 1
        start_state = env_config.start_states[0]

        assert len(env_config.goal_states) == 1
        goal_state = env_config.goal_states[0]

        assert len(env_config.avoid_states) == 1
        fail_state = env_config.avoid_states[0]

        state_space = {fail_state}
        action_space = np.arange(len(env_config.decompositions))
        successor_map = {}

        for index, action in enumerate(env_config.decompositions):
            # Add states to the state_space
            state_space.update(action.initial_states)
            state_space.add(action.final_state)

            # Collect states from decomposition
            edges = {
                (state, index): action.final_state for state in action.initial_states
            }
            successor_map.update(edges)

        # TODO: make sure the goal is reachable from the start in the decomposition?
        # Ensure the start and goal are members of the decomposition
        assert start_state in state_space
        assert goal_state in state_space

        return cls(
            state_space=state_space,
            action_space=action_space,
            start_state=start_state,
            goal_state=goal_state,
            fail_state=fail_state,
            successor_map=successor_map,
        )

    def close(self, **kwargs):
        """Closes the environment."""
        raise NotImplementedError

    def validate_state(self, state: Any):
        """Validates the format for a state for a particular environment."""
        return isinstance(state, int) and state in self.state_space
