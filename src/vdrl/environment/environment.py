from abc import ABC, abstractmethod
from typing import List, Any


class Environment(ABC):
    """Environment abstract class that defines the structure that should be followed for all our environments."""

    def __init__(
        self,
        start_states: List[Any],
        goal_states: List[Any],
        fail_states: List[Any] = None,
        state_space: Any = None,
        action_space: Any = None,
    ):
        """Create an environment object

        Attributes
        ------
        start_states (List[Any]): The list of start states for the environment.
        goal_states (List[Any]): The list of goal states for the environment.
        fail_states (List[Any]): The list of fail states for the environment.
        state_space (Any): The state space for the environment.
        action_space (Any): The action space for the environment.
        """
        # TODO: Does the Environment actually need these, or just the meta-environment?
        self.action_space = action_space
        self.state_space = state_space

        self.start_states = start_states
        self.goal_states = goal_states
        self.fail_states = fail_states

        for state in self.start_states:
            self.validate_state(state)

        for state in self.goal_states:
            self.validate_state(state)

        if fail_states:
            for state in self.fail_states:
                self.validate_state(state)

    @abstractmethod
    def step(self, action: List[Any]):
        """Executes a step in the environement.

        Parameters
        ----------
        action : List[Any]
            The action that is to be executed in the environment.
            This is made a list to handle both continous and discrete action
            spaces. For discrete environments, we will pass a singular list with
            the action to be executed.

        Returns
        -------
        observation : List[Any]
            An element of the environment’s observation_space as the next
            observation due to the agent actions. An example is a numpy array
            containing the positions and velocities of the pole in CartPole.
        reward : SupportsFloat
            The reward as a result of taking the action.
        terminated : bool
            Whether the agent reaches the terminal state (as defined under the
            MDP of the task) which can be positive or negative. An example is
            reaching the goal state or moving into the lava from the Sutton and
            Barton, Gridworld. If true, the user needs to call reset().
        truncated : bool
            Whether the truncation condition outside the scope of the MDP is
            satisfied. Typically, this is a timelimit, but could also be used
            to indicate an agent physically going out of bounds. Can be used to
            end the episode prematurely before a terminal state is reached. If
            true, the user needs to call reset().
        info : dict
            Contains auxiliary diagnostic information (helpful for
            debugging, learning, and logging). This might, for instance,
            contain: metrics that describe the agent’s performance state,
            variables that are hidden from observations, or individual reward
            terms that are combined to produce the total reward. In OpenAI Gym
            <v26, it contains “TimeLimit.truncated” to distinguish truncation
            and termination, however this is deprecated in favour of returning
            terminated and truncated variables.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self, **kwargs):
        """Resets the environment to it's original state.

        Outputs
        -------
        obs : Any
            Observation of the starting state
        info : dict
            Information relevant to this environment
        """
        raise NotImplementedError

    @abstractmethod
    def render(self, **kwargs):
        """Renders the environments to help visualise what the agent sees."""
        raise NotImplementedError

    @abstractmethod
    def close(self, **kwargs):
        """Closes the environment."""
        raise NotImplementedError

    @abstractmethod
    def validate_state(self, state: Any):
        """Validates the format for a state for a particular environment."""
        raise NotImplementedError
