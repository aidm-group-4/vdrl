from abc import ABC, abstractmethod

from vdrl.environment import Environment


class Model(ABC):
    """
    A Reinforcement Learning model to be trained in an environment.

    Attributes
    ------
    environment : Environment
        The environment in which this model operates.
    """

    @abstractmethod
    def __init__(self, environment: Environment):
        """
        Create a Model object.

        Inputs
        ------
        environment : Environment
            The environment in which this controller operates.
        """
        self.environment = environment

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def learn(self, total_timesteps=5e4):
        """
        Train the Model with a fixed computation budget.

        Inputs
        ------
        total_timesteps : int
            Total number of timesteps to train the model for.
        """
        raise NotImplementedError

    @abstractmethod
    def save(self, save_dir):
        """
        Save the Model object.

        Inputs
        ------
        save_dir : string
            Absolute path to the directory that will be used to save this Controller.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def load(load_dir):
        """
        Load a Model object.

        Inputs
        ------
        load_dir : string
            Absolute path to the directory of a previously saved Controller.
        """
        raise NotImplementedError
