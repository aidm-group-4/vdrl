from sklearn.tree import DecisionTreeClassifier
from vdrl.model.model import Model
from vdrl.environment.environment import Environment
import numpy as np
from typing import Optional, Tuple, List
from joblib import load, dump
from stable_baselines3 import PPO
import torch
import gymnasium as gym


class TreeWrapper:
    """This is a wrapper around the DecisionTreeClassifier as in the VIPER implementation."""

    def __init__(self, tree: DecisionTreeClassifier):
        """Initialises the wrapper with a decision tree.

        Parameters
        ----------
        tree : DecisionTreeClassifier
            The descision tree that will be wrapped.
        """
        self.tree = tree

    def predict(
        self,
        observation: np.ndarray,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """Wrapper around the decision tree predict function so that it doesn't fail
        if called with additional arguments that a stablebaselines model could be called with.

        Parameters
        ----------
        observation : np.ndarray
            The observation from the environment.
        state : Optional[Tuple[np.ndarray, ...]], optional
            This will never be passed.
        episode_start : Optional[np.ndarray], optional
            This will never be passed.
        deterministic : bool, optional
            This will never be passed.

        Returns
        -------
        Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]
            Returns the next action based on the observation.
        """
        return self.tree.predict(observation), None

    @classmethod
    def load(cls, path: str):
        """Loads the decision tree from the given path.

        Parameters
        ----------
        path : str
            Path to load from.

        Returns
        -------
        TreeWrapper
            Returns a loaded wrapper.
        """
        clf = load(path)
        return TreeWrapper(clf)

    def save(self, path: str):
        """Saves the wrapper at the given path.

        Parameters
        ----------
        path : str
            Path to save wrapper to.
        """
        print(f"Saving to\t{path}")
        dump(self.tree, path)

    def print_info(self):
        """Prints the max depth and leaves of the tree."""
        print(f"Max depth:\t{self.tree.get_depth()}")
        print(f"# Leaves:\t{self.tree.get_n_leaves()}")


def get_loss(env: Environment, model: Model, obs):
    """
    This is the ~l loss from the VIPER paper that tries to capture
    how "critical" a state is, i.e. how much of a difference
    it makes to choose the best vs the worst action

    Instead of training the decision tree with this loss directly (which is not possible because it is not convex)
    we use it as a weight for the samples in the dataset which in expectation leads to the same result.

    Parameters
    ----------
    env : Environment
        The environment that the oracle and tree is being trained on.
    model : Model
        The oracle model.
    obs :
        The observation obtained from the environment.

    Returns
    -------
    Returns the loss for the current observation based on the VIPER paper.

    Raises
    ------
    NotImplementedError
        Currently, only PPO is supported for the oracle model. Will raise NotImplementedError
        if the oracle is any other kind of model.
    """
    _model = model._model
    _env = env
    if hasattr(env, "_env"):
        _env = env._env

    if isinstance(_model, PPO):
        # For policy gradient methods we use the max entropy formulation
        # to get Q(s, a) \approx log pi(a|s)
        # See Ziebart et al. 2008
        assert isinstance(
            _env.action_space, gym.spaces.Discrete
        ), "Only discrete action spaces supported for loss function"
        possible_actions = np.arange(_env.action_space.n)

        obs = torch.from_numpy(obs)
        log_probs = []
        for action in possible_actions:
            action = torch.from_numpy(np.array([action])).repeat(obs.shape[0])
            _, log_prob, _ = model.policy.evaluate_actions(obs, action)
            log_probs.append(log_prob.detach().numpy().flatten())

        log_probs = np.array(log_probs).T

        return log_probs.max(axis=1) - log_probs.min(axis=1)

    raise NotImplementedError(f"Model type {type(model)} not supported.")


def sample_trajectory(
    oracle: Model,
    environment: Environment,
    policy: Optional[DecisionTreeClassifier],
    beta: float,
    n_steps: int,
) -> List:
    """Samples a trajectory to build the dataset in each iteration as implemented by VIPER.

    Parameters
    ----------
    oracle : Model
        The oracle model that the decision tree is being learned for.
    environment : Environment
        The environment that the agent is running in.
    policy : Optional[DecisionTreeClassifier]
        The decision tree policy. This will not be passed in the first iteration of training
        the VIPER decision tree.
    beta : float
        The probablity by which the oracle is chosen over the policy for sampling.
    n_steps: int
        The number of steps to sample for the trajectory.
    Returns
    -------
    list
        Returns the sampled trajectory in the environment.
    """
    trajectory = []

    obs = environment.reset()
    while len(trajectory) < n_steps:
        active_policy = [policy, oracle][np.random.binomial(1, beta)]

        action, _ = active_policy.predict(obs, deterministic=True)

        if not isinstance(active_policy, DecisionTreeClassifier):
            oracle_action = action
        else:
            oracle_action = oracle.predict(obs, deterministic=True)[0]

        next_obs, reward, terminated, truncated, info = environment.step(action)

        state_loss = get_loss(environment, oracle, obs)
        trajectory += list(zip(obs, oracle_action, state_loss))

        obs = next_obs

    return trajectory
