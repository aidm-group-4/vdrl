import os
import pickle

import numpy as np
import gurobipy as gb

from vdrl.model import Model
from .meta_environment import MetaEnvironment


class MetaModel(Model):
    """
    Class representing the MDP model of the high-level decision making process.
    """

    def __init__(self, meta_environment: MetaEnvironment, meta_policy=None):
        """
        Inputs
        ------
        meta_environment : MetaEnvironment
            The abstracted higher-level environment over the high-level decomposition.
        meta_policy : numpy array
            Numpy array representing the meta-policy.
        """
        self.environment = meta_environment
        self.successors = self.environment.successor

        self.N_S = len(
            self.environment.state_space
        )  # Number of states in the high-level MDP
        self.N_A = len(
            self.environment.action_space
        )  # Number of actions in the high-level MDP

        self.avail_actions = {}
        self._construct_avail_actions()
        self.avail_states = {}
        self._construct_avail_states()

        # Transition probabilities from (state,action) pairs to state
        self.P = np.zeros((self.N_S, self.N_A, self.N_S), dtype=np.float64)

        # Using the successor map, construct a predecessor map.
        self.predecessors = {}
        self._construct_predecessor_map()

        if meta_policy is None:
            self.meta_policy = np.zeros((self.N_S, self.N_A), dtype=np.float64)
        else:
            assert np.shape(meta_policy) == (self.N_S, self.N_A)
            self.meta_policy = meta_policy

    def _construct_avail_actions(self):
        """
        Construct a map of lists of available actions at each meta-state.
        """
        for s in self.environment.state_space:
            self.avail_actions[s] = []

        for s in self.environment.state_space:
            for a in range(self.N_A):
                if (s, a) in self.successors.keys():
                    self.avail_actions[s].append(a)

    def _construct_avail_states(self):
        """
        Construct a map of lists of available states after each meta-action.
        """
        for a in self.environment.action_space:
            self.avail_states[a] = set()

        for s in self.environment.state_space:
            avail_actions = self.avail_actions[s]
            for action in avail_actions:
                self.avail_states[action].add(s)

    def _construct_predecessor_map(self):
        """
        Using the successor map, construct a predecessor map.
        """
        for s in self.environment.state_space:
            self.predecessors[s] = []
            for sp in self.environment.state_space:
                avail_actions = self.avail_actions[sp]
                for action in avail_actions:
                    if self.successors[(sp, action)] == s:
                        self.predecessors[s].append((sp, action))

    def solve_feasible_policy(self, prob_threshold):
        """
        If a meta-policy exists that reaches the goal state from the target
        state with probability above the specified threshold, return it.

        Inputs
        ------
        prob_threshold : float
            Value between 0 and 1 that represents the desired probability of
            reaching the goal.

        Outputs
        -------
        policy : numpy (N_S, N_A) array
            Array representing the solution policy. If there is no feasible
            solution, an array of -1 is returned.
        feasible_flag : bool
            Flag indicating whether or not a feasible solution was found.
        """
        if prob_threshold > 1 or prob_threshold < 0:
            raise RuntimeError("prob threshold is not a probability")

        # initialize gurobi model
        linear_model = gb.Model("abs_mdp_linear")

        # dictionary for state action occupancy
        state_act_vars = dict()

        avail_actions = self.avail_actions.copy()

        # dummy action for goal state
        avail_actions[self.environment.goal_states[0]] = [0]

        # create occupancy measures, probability variables and reward variables
        for s in self.environment.state_space:
            for a in avail_actions[s]:
                state_act_vars[s, a] = linear_model.addVar(
                    lb=0, name="state_act_" + str(s) + "_" + str(a)
                )

        # gurobi updates model
        linear_model.update()

        # MDP bellman or occupancy constraints for each state
        for s in self.environment.state_space:
            cons = 0
            # add outgoing occupancy for available actions
            for a in avail_actions[s]:
                cons += state_act_vars[s, a]

            # add ingoing occupancy for predecessor state actions
            for s_bar, a_bar in self.predecessors[s]:
                # this if clause ensures that you dont double count reaching goal and failure
                if (
                    s_bar not in self.environment.goal_states
                    and s_bar not in self.environment.fail_states
                ):
                    cons -= state_act_vars[s_bar, a_bar] * self.P[s_bar, a_bar, s]
            # initial state occupancy
            if s in self.environment.start_states:
                cons = cons - 1

            # sets occupancy constraints
            linear_model.addConstr(cons == 0)

        # prob threshold constraint
        for s in self.environment.state_space:
            if s in self.environment.goal_states:
                linear_model.addConstr(state_act_vars[s, 0] >= prob_threshold)

        # set up the objective
        obj = 0

        # set the objective, solve the problem
        linear_model.setObjective(obj, gb.GRB.MINIMIZE)
        linear_model.optimize()

        if linear_model.SolCount == 0:
            feasible_flag = False
        else:
            feasible_flag = True

        if feasible_flag:
            # Construct the policy from the occupancy variables
            policy = np.zeros((self.N_S, self.N_A), dtype=np.float64)
            for s in self.environment.state_space:
                if len(self.avail_actions[s]) == 0:
                    policy[s, :] = (
                        -1
                    )  # If no actions are available, return garbage value
                else:
                    occupancy_state = np.sum(
                        [state_act_vars[s, a].x for a in self.avail_actions[s]]
                    )
                    # If the state has no occupancy measure under the solution, set the policy to
                    # be uniform over available actions
                    if occupancy_state == 0.0:
                        for a in self.avail_actions[s]:
                            policy[s, a] = 1 / len(self.avail_actions[s])
                    if occupancy_state > 0.0:
                        for a in self.avail_actions[s]:
                            policy[s, a] = state_act_vars[s, a].x / occupancy_state
        else:
            policy = -1 * np.ones((self.N_S, self.N_A), dtype=np.float64)

        return policy, feasible_flag

    def solve_max_reach_prob_policy(self):
        """
        Find the meta-policy that maximizes probability of reaching the goal state.

        Outputs
        -------
        policy : numpy (N_S, N_A) array
            Array representing the solution policy. If there is no feasible solution, an array of
            -1 is returned.
        reach_prob : float
            The probability of reaching the goal state under the policy.
        feasible_flag : bool
            Flag indicating whether a feasible solution was found.
        """
        # initialize gurobi model
        linear_model = gb.Model("abs_mdp_linear")

        # dictionary for state action occupancy
        state_act_vars = dict()

        avail_actions = self.avail_actions.copy()

        # dummy action for goal state
        avail_actions[self.environment.goal_states[0]] = [0]

        # create occupancy measures, probability variables and reward variables
        for s in self.environment.state_space:
            for a in avail_actions[s]:
                state_act_vars[s, a] = linear_model.addVar(
                    lb=0, name="state_act_" + str(s) + "_" + str(a)
                )

        # gurobi updates model
        linear_model.update()

        # MDP bellman or occupancy constraints for each state
        for s in self.environment.state_space:
            cons = 0
            # add outgoing occupancy for available actions
            for a in avail_actions[s]:
                cons += state_act_vars[s, a]

            # add ingoing occupancy for predecessor state actions
            for s_bar, a_bar in self.predecessors[s]:
                # this if-clause ensures that you don't double count reaching goal and failure
                if (
                    s_bar not in self.environment.goal_states
                    and s_bar not in self.environment.fail_states
                ):
                    cons -= state_act_vars[s_bar, a_bar] * self.P[s_bar, a_bar, s]
            # initial state occupancy
            if s in self.environment.start_states:
                cons = cons - 1

            # sets occupancy constraints
            linear_model.addConstr(cons == 0)

        # set up the objective
        obj = 0
        obj += state_act_vars[
            self.environment.goal_states[0], 0
        ]  # Probability of reaching goal state

        # set the objective, solve the problem
        linear_model.setObjective(obj, gb.GRB.MAXIMIZE)
        linear_model.optimize()

        if linear_model.SolCount == 0:
            feasible_flag = False
        else:
            feasible_flag = True

        if feasible_flag:
            # Construct the policy from the occupancy variables
            policy = np.zeros((self.N_S, self.N_A), dtype=np.float64)
            for s in self.environment.state_space:
                if len(self.avail_actions[s]) == 0:
                    policy[s, :] = (
                        -1
                    )  # If no actions are available, return garbage value
                else:
                    occupancy_state = np.sum(
                        [state_act_vars[s, a].x for a in self.avail_actions[s]]
                    )
                    # If the state has no occupancy measure under the solution, set the policy to
                    # be uniform over available actions
                    if occupancy_state == 0.0:
                        for a in self.avail_actions[s]:
                            policy[s, a] = 1 / len(self.avail_actions[s])
                    if occupancy_state > 0.0:
                        for a in self.avail_actions[s]:
                            policy[s, a] = state_act_vars[s, a].x / occupancy_state
        else:
            policy = -1 * np.ones((self.N_S, self.N_A), dtype=np.float64)

        reach_prob = state_act_vars[self.environment.goal_states[0], 0].x

        self.meta_policy = policy
        return policy, reach_prob, feasible_flag

    def learn(self):
        """
        Find the meta-policy that maximizes probability of reaching the goal state.

        Outputs
        -------
        policy : numpy (N_S, N_A) array
            Array representing the solution policy. If there is no feasible solution, an array of
            -1 is returned.
        reach_prob : float
            The probability of reaching the goal state under the policy.
        feasible_flag : bool
            Flag indicating whether or not a feasible solution was found.
        """
        self.meta_policy, _, _ = self.solve_max_reach_prob_policy()

    def predict(self, obs):
        """
        Get the next controller, given the current meta-state

        Inputs
        ------
        obs : int
            Integer representing the current meta state.
        """
        transition_probabilities = self.meta_policy[obs, :]
        return np.random.choice(
            self.environment.action_space, p=transition_probabilities
        )

    def save(self, save_dir: str):
        """
        Save the Model object.

        Inputs
        ------
        save_dir : string
            Absolute path to the directory that will be used to save this Controller.
        """
        # create save directory if it doesn't exist
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        meta_model_file = os.path.join(save_dir, "meta_model.p")
        meta_model_data = {
            # TODO: add whatever data we think is important here
            "meta_policy": self.meta_policy,
        }

        with open(meta_model_file, "wb") as pickleFile:
            pickle.dump(meta_model_data, pickleFile)

    def load(load_dir: str, meta_environment: MetaEnvironment):
        """
        Load a Model object.

        Inputs
        ------
        load_dir : string
            Absolute path to the directory of a previously saved Controller.
        meta_environment : MetaEnvironment
            The abstracted higher-level environment over the high-level decomposition.
        """
        meta_model_file = os.path.join(load_dir, "meta_model.p")
        with open(meta_model_file, "rb") as pickleFile:
            meta_model_data = pickle.load(pickleFile)

        return MetaModel(meta_environment, meta_model_data["meta_policy"])
