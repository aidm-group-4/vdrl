from enum import IntEnum
from typing import Any, List, Tuple
import numpy as np


from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Goal, Lava
from minigrid.core.mission import MissionSpace
from gymnasium import spaces

from vdrl.environment import Environment
from vdrl.exceptions import IncorrectStateDefinition
from vdrl.config import MinigridConfig


class MiniGridLabyrinth(Environment):
    """This is a wrapper class around the MinigridLabyrinthRefactored class. This is wrapped so that it inherits from our environment base class and
    follows the same structure. Internally, this uses the MinigridLabyrinthRefactored to interact with the environment.
    """

    def __init__(
        self,
        start_states: List[Any],
        goal_states: List[Any],
        problem_config: MinigridConfig,
        fail_states: List[Any] = None,
    ):

        self._env: MinigridLabyrinthRefactored = MinigridLabyrinthRefactored(
            problem_config=problem_config,
            start_states=start_states,
            goal_states=goal_states,
            fail_states=fail_states,
        )

        super().__init__(
            start_states,
            goal_states,
            fail_states,
            self._env.observation_space,
            self._env.action_space,
        )

    def step(self, action: List[Any]):
        """Executes a step in the MiniGridLabyrinth environment.

        Uses the MinigridLabyrinthRefactored to take the step.

        Parameters
        ----------
        action : List[int]
            The action that is to be executed in the environment. This must be
            a singleton list.

        Returns
        -------
        observation : (int, int, int)
            The state the agent moves into with integer coordinates (x, y, orientation).
        reward : SupportsFloat
            The reward as a result of taking the action.
        terminated : bool
            Whether the agent reaches a state where it can no longer move (which
            can be a goal or an avoid state). If true, the user needs to call
            reset() on this environment before using it again.
        truncated : bool
            Whether the environment was stopped, typically due to a time limit.
        info : dict
            Contains auxiliary diagnostic information (helpful for
            debugging, learning, and logging).
        """
        action = action[0]
        return self._env.step(action=action)

    def get_transition(self, state, action):

        return self._env.get_transition(state, action)

    def close(self, **kwargs):
        """Defines the close function for the MiniGridLabyrinth environment. Uses the MinigridLabyrinthRefactored to close the environment."""
        return self._env.close(**kwargs)

    def reset(self, **kwargs):
        """Resets the MiniGridLabyrinth to it's original state.

        Outputs
        -------
        obs : Any
            Observation of the starting state
        info : dict
            Information relevant to this environment
        """
        return self._env.reset(**kwargs)

    def render(self, render_mode=None, highlight=False):
        """Defines the render function for the MiniGridLabyrinth environment. Uses the MinigridLabyrinthRefactored to render the environment."""
        old_render_mode = self._env.render_mode
        old_highlight = self._env.highlight
        self._env.render_mode = render_mode
        self._env.highlight = highlight
        img = self._env.render()
        self._env.render_mode = old_render_mode
        self._env.highlight = old_highlight
        return img

    def validate_state(self, state: Tuple):
        """In the Minigrid Labyrinth class, a state should be a tuple of the format (x_coordinate, y_coordinate, direction). This method ensure that
        the start, goal and fail states provided follow that format.

        Args:
            states (List[Tuple]): List of states to validate.

        Returns:
            Returns a boolean whether the state is valid or not.

        """

        if not isinstance(state, tuple) or len(state) != 3:
            raise IncorrectStateDefinition(
                f"The state {state} should be a tuple of length 3 representing (x_coordinate, y_coordinate, direction.) in the Minigrid Environment."
            )


class MinigridLabyrinthRefactored(MiniGridEnv):
    """This is the refactored version of the original Maze environment defined by Neary. Instead of hardcoding the environment world objects and start/final states,
    it takes them from the user.
    TODO: Add validation for world objects in this class.
    """

    class Actions(IntEnum):
        """Default actions for the minigrid environment limiting to only left, right and forward."""

        left = 0
        right = 1
        forward = 2

    def __init__(
        self,
        problem_config: MinigridConfig,
        start_states: List[Any],
        goal_states: List[Any],
        fail_states: List,
    ):
        """
        Inputs
        ------
        problem_config: MinigridConfig
            The configuration for the MiniGrid problem that defines the positions of the goal objects
        start_states:  List[Any]
            The start states for the MiniGrid environment.
        goal_states: List[Any]:
            The goal states for the MiniGrid environement.
        fail_states: List[Any]:
            The fail states for the MiniGrid environemnt.
        """
        self.problem_config = problem_config

        # self.width = self.problem_config.width
        # self.height = self.problem_config.height

        super().__init__(
            mission_space=MissionSpace(mission_func=self._gen_mission),
            height=problem_config.height,
            width=problem_config.width,
            # TODO: should we change the max number of steps allowed?
            max_steps=4 * problem_config.height * problem_config.width,
            # render_mode="human",
            # highlight=False,
        )

        # Actions are modified to only the actions needed
        self.actions = MinigridLabyrinthRefactored.Actions
        self.action_space = spaces.Discrete(len(self.actions))

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([problem_config.width, problem_config.height, 3]),
            dtype="uint8",
        )

        self.start_states = start_states
        self.goal_states = goal_states
        self.fail_states = fail_states
        self._gen_grid()

    @staticmethod
    def _gen_mission():
        return "Solve the maze without falling in lava."

    def _gen_grid(self, width=None, height=None):
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate the rooms. Each sub-policy is confined within a room since the walls do not allow it to exit the space of the sub-policy.
        for wall in self.problem_config.walls:
            self.grid.wall_rect(*wall)

        # Adding the doors in the environemt
        for door in self.problem_config.doors:
            self.put_obj(Door("grey", is_open=True), door[0], door[1])

        # Adding the goal states in the environment.
        for goal_state in self.goal_states:
            self.put_obj(Goal(), goal_state[0], goal_state[1])

        # Place the avoid states. The avoid states are in the format (x_cordinate, y_cordinate, length).
        for lava in self.fail_states:
            self.grid.horz_wall(lava[0], lava[1], 1, obj_type=Lava)

        # Place the agent
        if self.start_states:
            # Uniformly pick from the possible start states
            agent_start_state = self.start_states[
                np.random.choice(len(self.start_states))
            ]
            self.agent_pos = (agent_start_state[0], agent_start_state[1])
            self.agent_dir = agent_start_state[2]
        else:
            self.place_agent()

        self.mission = f"Get from {self.start_states} to {self.goal_states}"

    def gen_obs(self):
        """
        Generate the observation of the agent, which in this environment, is its state.
        """
        pos = self.agent_pos
        direction = self.agent_dir
        obs_out = np.array([pos[0], pos[1], direction])
        return obs_out

    def get_transition(self, state, model_action):

        slip_p = self.problem_config.slip_p

        state_distribution = {}

        # It is not possible to enumerate the action space from gymnasium
        action_space = [0, 1, 2]

        for action in action_space:
            if action == model_action:
                next_state = self.get_next_state(state, action)
                # This is consistent with the step function definition
                state_distribution[next_state] = 1 - slip_p + slip_p / len(action_space)
            else:
                next_state = self.get_next_state(state, action)
                state_distribution[next_state] = slip_p / len(action_space)

        return state_distribution

    def get_next_state(self, state, taken_action):
        pos = (state[0], state[1])
        dir = state[2]

        if taken_action == self.actions.forward:
            fwd_pos = self.fwd_pos(pos, dir)
            fwd_cell = self.grid.get(*fwd_pos)
            if fwd_cell is None or fwd_cell.can_overlap():
                return (fwd_pos[0], fwd_pos[1], dir)
            else:
                return state

        elif taken_action == self.actions.right:
            dir = (dir + 1) % 4
        elif taken_action == self.actions.left:
            dir = (dir - 1) % 4
        return (state[0], state[1], dir)

    def fwd_pos(self, current_pos, dir):
        if dir == 0:
            return (current_pos[0] + 1, current_pos[1])
        if dir == 1:
            return (current_pos[0], current_pos[1] + 1)
        if dir == 2:
            return (current_pos[0] - 1, current_pos[1])
        if dir == 3:
            return (current_pos[0], current_pos[1] - 1)

    def step(self, action):
        """Executes a step in the MinigridLabyrinthRefactored environment.

        Parameters
        ----------
        action : int
            The action that is to be executed in the environment. This must be
            a singleton list.

        Returns
        -------
        observation : (int, int, int)
            The state the agent moves into with integer coordinates (x, y, orientation).
        reward : SupportsFloat
            The reward as a result of taking the action.
        terminated : bool
            Whether the agent reaches a state where it can no longer move (which
            can be a goal or an avoid state). If true, the user needs to call
            reset() on this environment before using it again.
        truncated : bool
            Whether the environment was stopped, typically due to a time limit.
        info : dict
            Contains auxiliary diagnostic information (helpful for
            debugging, learning, and logging).
        """
        self.step_count += 1
        slip_p = self.problem_config.slip_p

        reward = 0
        terminated = False
        truncated = False

        info = {"lava": False}

        # Slip probability causes agent to randomly take the wrong action
        # TODO: Check this (this also needs to change the transition function if it changes)
        if np.random.rand() <= slip_p:
            action = self.action_space.sample()

        current_pos = self.agent_pos
        current_cell = self.grid.get(*current_pos)
        if current_cell is not None and current_cell.type == "lava":
            # If the agent is in lava, it can no longer do anything
            action = None

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell is not None and fwd_cell.type == "lava":
                terminated = True
                info["lava"] = True

        else:
            assert False, "unknown action"

        next_state = (self.agent_pos[0], self.agent_pos[1], self.agent_dir)
        if next_state in self.goal_states:
            terminated = True
            reward = 1.0

        if self.step_count >= self.max_steps:
            truncated = True

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, info

    def get_num_states(self):
        return self.width * self.height * 4
