import os

import numpy as np
from stable_baselines3 import PPO

from vdrl.environment import Environment

from .model import Model


class PPOModel(Model):
    """
    A PPO model using a deep neural network policy to be trained in an environment.

    Attributes
    ------
    environment : Environment
        The environment in which this model operates.
    """

    def __init__(self, environment: Environment, *args, model=None, **kwargs):
        """
        Create a PPO model based on an MlpPolicy.
        """
        self.environment = environment

        if model is None:
            self._model = PPO("MlpPolicy", environment, *args, **kwargs)
        else:
            self._model = model

    def predict(self, obs, deterministic=True):
        """
        Get the action predicted by this model, given the current environment
        observation (state if fully observable).

        Inputs
        ------
        obs : tuple
            Tuple representing the current environment observation.
        deterministic (optional) : bool
            Flag indicating whether or not to return a deterministic action or
            a distribution over actions.

        Outputs
        ------
        action : List[Any]
            A list (or numpy array) indicating the action to take. Discrete
            actions can be represented as an integer in a singleton list.
        _state : Any
            The resulting state from following the predicted action.?
        """
        action, _state = self._model.predict(obs, deterministic=deterministic)
        return np.atleast_1d(action), _state

    def learn(self, total_timesteps=5e4):
        """
        Train the PPO Model with a fixed computation budget.

        Inputs
        ------
        total_timesteps : int
            Total number of timesteps to train the model for.
        """
        self._model.learn(total_timesteps=total_timesteps)

    def save(self, save_dir):
        """
        Save the Model object.

        Inputs
        ------
        save_dir : string
            Absolute path to the directory that will be used to save this Controller.
        """
        model_file = os.path.join(save_dir, "model")
        self._model.save(model_file)

    def load(load_dir, environment):
        """
        Load a Model object.

        Inputs
        ------
        load_dir : string
            Absolute path to the directory of a previously saved Controller.
        """

        # model_file = os.path.join(load_dir, "model")
        model = PPO.load(load_dir, env=environment._env)

        return PPOModel(environment, model=model)
