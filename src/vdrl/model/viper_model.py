import numpy as np
from vdrl.environment import Environment
from .model import Model
from sklearn.tree import DecisionTreeClassifier
from vdrl.viper.train import train_viper
from typing import Optional
from vdrl.config import ViperConfig


class ViperModel(Model):
    """A verifiable model based on the VIPER framework to be trained in an environment."""

    def __init__(
        self,
        environment: Environment,
        model: Model,
        viper_config: ViperConfig,
        tree: Optional[DecisionTreeClassifier] = None,
        *args,
        **kwargs,
    ):
        """
        Create a VIPER model.

        Inputs
        ------
        environment : Environment
            The environment in which this model operates.
        model: Model
            The underlying model that VIPER is built on. This model is the oracle policy
            that the viper decision tree policy will imitate.
        viper_config: ViperConfig
            The configuration for the viper framework
        tree: Optional[DecisionTreeClassifier]
            A decision tree that imitates the policy. Pass this if Viper is already trained.
        """
        self.viper_config = viper_config
        self.environment = environment
        self._model = model
        if tree:
            self._tree = tree

    def predict(self, obs, deterministic=True):
        """
        Get the action predicted by this model, given the current environment
        observation (state if fully observable). Viper simply uses the underlying model
        for this.
        NOTE: Should this use the model or the tree to predict?

        Inputs
        ------
        obs : tuple
            Tuple representing the current environment observation.
        deterministic (optional) : bool
            Flag indicating whether or not to return a deterministic action or
            a distribution over actions.
        """
        action, _state = self._model.predict(obs, deterministic=deterministic)
        return np.atleast_1d(action), _state

    def learn(self, total_timesteps=5e4):
        """
        Trains the oracle model as well as the viper decision tree

        Inputs
        ------
        total_timesteps : int
            Total number of timesteps to train the model for.
        """
        self._model.learn(total_timesteps=total_timesteps)
        self._tree = train_viper(
            oracle=self._model,
            environment=self.environment,
            num_iterations=self.viper_config.num_iterations,
            max_depth=self.viper_config.max_depth,
            max_leaves=self.viper_config.max_depth,
            verbose=self.viper_config.verbose,
        )

    def save(self, save_dir):
        """
        Save the Model object.

        Inputs
        ------
        save_dir : string
            Absolute path to the directory that will be used to save this Controller.
        """
        raise NotImplementedError

    def load(load_dir, environment):
        """
        Load a Model object.

        Inputs
        ------
        load_dir : string
            Absolute path to the directory of a previously saved Controller.
        """
        raise NotImplementedError
