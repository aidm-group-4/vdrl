from abc import ABC, abstractmethod

import numpy as np

from vdrl.environment import Environment
from vdrl.model import Model


class Controller(ABC):
    """
    A Controller object is responsible for providing the verification information,
    telling models when/how much to train, and evaluating their performance.

    Attributes
    ------
    environment : Environment
        The environment in which this controller operates.
    model : Model
        The model which is trained within the environment.
    verifiable_properties : List[str]
        A tuple of the properties which can be guaranteed by this controller.
    guarantees : dict
        The performance guarantees of the (partially trained) model for each
        verifiable property.

    """

    verifiable_properties: tuple[str]

    def __init__(
        self,
        environment: Environment,
        model: Model,
        verifiable_properties: tuple[str] = (),
    ):
        """
        Create a Controller object

        Inputs
        ------
        environment : Environment
            The environment in which this controller operates.
        model : Model
            The model which is trained within the environment.
        verifiable_properties : List[str]
            A tuple of the properties which can be guaranteed by this controller.
        """
        self.environment = environment
        self.model = model
        self.verifiable_properties = verifiable_properties
        self.guarantees = {prop: 0.0 for prop in self.verifiable_properties}

    @abstractmethod
    def get_success_prob(self):
        """Get the current rate of success for this controller to be used by
        the high-level controller.

        Subclasses must override the definition of this function.

        Returns
        -------
        success_prob : float
            The lower-bound on the probability of success for the model.
        """
        pass

    def predict(self, obs, deterministic=True):
        """
        Get the action predicted by this Controller's model, given the current
        environment observation (state if fully observable).

        Inputs
        ------
        obs : tuple
            Tuple representing the current environment observation.
        deterministic (optional) : bool
            Flag indicating whether to return a deterministic action or
            a distribution over actions.

        Outputs
        ------
        action : List[Any]
            A list (or numpy array) indicating the action to take. Discrete
            actions can be represented as an integer in a singleton list.
        _state : Any
            The resulting state from following the predicted action.?
        """
        return self.model.predict(obs, deterministic=deterministic)

    @abstractmethod
    def learn(self):
        """
        Train the Controller with a fixed computation budget.
        """
        raise NotImplementedError

    def _reset(self):
        """
        Subclasses can use this to reset the controller state if needed.
        """
        pass

    @abstractmethod
    def save(self, save_dir):
        """
        Save the Controller object.

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
        Load a Controller object.

        Inputs
        ------
        load_dir : string
            Absolute path to the directory of a previously saved Controller.
        """
        raise NotImplementedError

    def demonstrate_capabilities(
        self, environment, n_episodes=5, n_steps=100, render=False
    ):
        """
        Run the controller in an environment and visualize the results.

        Inputs
        ------
        environment : Environment
            An Environment in which to evaluate capabilities
        n_episodes : int
            Number of episodes to rollout for evaluation.
        n_steps : int
            Maximum length of each episode.
        render (optional) : bool
            Flag indicating whether or not to render the environment.
        """
        histories = []

        for episode_ind in range(n_episodes):
            history = {
                "actions": [],
                "observations": [],
                "rewards": [],
                "terminated": [],
                "infos": [],
            }
            obs, _ = environment.reset()
            self._reset()

            for step in range(n_steps):
                history["observations"].append(obs)

                action, _states = self.predict(obs, deterministic=True)
                history["actions"].append(action)

                obs, reward, terminated, truncated, info = environment.step(action)
                history["rewards"].append(reward)
                history["terminated"].append(terminated)
                history["infos"].append(info)

                if render:
                    environment.render()
                if terminated:
                    break
            histories.append(history)

        return histories

    def eval_performance(self, environment, n_episodes=200, n_steps=1000):
        """
        Perform empirical evaluation of the performance of the controller
        in an environment.

        Inputs
        ------
        environment : Environment
            Environment to perform evaluation in.
        n_episodes (optional) : int
            Number of episodes to rollout for evaluation.
        n_steps (optional) : int
            Length of each episode.

        Outputs
        -------
        success_rate : float
            Empirically measured rate of success of the meta-controller.
        avg_success_steps: float
            The average number of steps taken in each successful run. If none
            were successful this returns np.inf.
        total_steps : int
            The total number of steps during all episodes.
        """
        success_count = 0
        success_steps = []
        total_steps = 0

        for episode_ind in range(n_episodes):
            obs, _ = environment.reset()
            self._reset()

            num_steps = 0
            for step_ind in range(n_steps):
                num_steps += 1
                action, _states = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = environment.step(action)
                if terminated or truncated:
                    if reward > 0:
                        success_count = success_count + 1
                        success_steps.append(num_steps)
                    total_steps += num_steps
                    break

        if success_count > 0:
            avg_success_steps = np.mean(success_steps)
        else:
            avg_success_steps = np.inf

        return success_count, avg_success_steps, total_steps
