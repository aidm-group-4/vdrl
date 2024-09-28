import contextlib
from datetime import datetime
import os

import numpy as np
import gurobipy as gb
import torch
import random
import pickle

from vdrl.controller import Controller, EmpericalSubController, ReachAvoidSubController
from vdrl.utils.results_saver import Results

from .meta_model import MetaModel
from .meta_environment import MetaEnvironment


class MetaController(Controller):
    """
    Class representing the controller of the high-level decision making process.
    """

    def __init__(
        self,
        meta_environment: MetaEnvironment,
        controller_list,
        obs_mapping,
        discount=1.0,
        meta_model=None,
    ):
        """
        Inputs
        -----
        meta_environment : MetaEnvironment
            The abstracted higher-level environment over the high-level decomposition.
        controller_list : list
            List of MinigridController objects (the sub-systems being used as
            components of the overall RL system).
        obs_mapping : lambda (obs) -> int or None
            A function mapping observations in the real environment to
            high-level states. If the high-level state isnt found, this
            function returns None.
        discount : float
            The discount factor for the MDP.
        meta_model : MetaModel
            Optionally provide an existing meta model (useful if loading from a previous save)
        """
        self._obs_mapping = obs_mapping
        self.controller_list = controller_list
        self.discount = discount

        assert len(meta_environment.action_space) == len(
            controller_list
        )  # Each meta-action should have a controller
        if meta_model is None:
            meta_model = MetaModel(meta_environment)
        else:
            assert (
                meta_environment == meta_model.environment
            )  # The loaded meta-environment should match a loaded meta-model
        verifiable_properties = (
            "reachablity",  # The probability of reaching a goal state
        )

        super().__init__(
            meta_environment, meta_model, verifiable_properties=verifiable_properties
        )
        self._success_prob = 0

        self.current_controller_ind = None
        self.save_dir = None
        self.results = None
        self.rseed = None

    def update_transition_function(self):
        """
        Re-construct the transition function to reflect any changes in the
        measurements of how likely each controller is to succeed.
        """
        self.model.P = np.zeros((self.model.N_S, self.model.N_A, self.model.N_S))
        self._construct_transition_function()

    def _construct_transition_function(self):
        """
        Construct the transition function to reflect any changes in the
        measurements of how likely each controller is to succeed.
        """
        for s in self.environment.state_space:
            for action in self.model.avail_actions[s]:
                success_prob = self.get_success_prob(action)
                next_s = self.environment.successor[(s, action)]

                self.model.P[s, action, next_s] = success_prob
                self.model.P[s, action, self.environment.fail_states[0]] = (
                    1 - success_prob
                )

    def process_high_level_demonstrations(self, demos: list) -> tuple:
        """
        Process the high-level demonstrations into average expected
        discounted feature counts. The features currently just correspond
        to state-action pairs.

        Inputs
        ------
        demos :
            A list of demonstration trajectories. demos[i] is a trajectory
            and trajectory[t] is a state-action pair represented as a list.
            stateAction[0] is the state at time t and stateAction[1] is the
            action.

        Outputs
        -------
        state_features_counts :
            The discounted average feature counts of the demonstrations.
            feature_counts[i] is the discounted count of state i.
        state_act_feature_counts :
            The discounted average feature counts of the demonstrations.
            feature_counts[i,j] is the count of action j in state i.
        """
        num_trajectories = len(demos)
        state_act_feature_counts = np.zeros((self.model.N_S, self.model.N_A))
        state_feature_counts = np.zeros(self.model.N_S)
        for i in range(num_trajectories):
            traj = demos[i]
            for t in range(len(traj)):
                state, action = traj[t]
                state_feature_counts[state] = (
                    state_feature_counts[state] + self.discount**t
                )
                state_act_feature_counts[state, action] = (
                    state_act_feature_counts[state, action] + self.discount**t
                )
        state_act_feature_counts = state_act_feature_counts / num_trajectories
        state_feature_counts = state_feature_counts / num_trajectories

        return state_feature_counts, state_act_feature_counts

    def solve_low_level_requirements_action(
        self, prob_threshold, max_timesteps_per_component=None
    ):
        """
        Find new transition probabilities guaranteeing that a feasible meta-policy exists.

        Inputs
        ------
        prob_threshold : float
            The required probability of reaching the target set in the HLM.
        max_timesteps_per_component : int
            Number of training steps (for an individual sub-system) beyond which its current
            estimated performance value should be used as an upper bound on the corresponding
            transition probability in the HLM.

        Outputs
        -------
        policy : numpy array
            The meta-policy satisfying the task specification, under the solution
            transition probabilities in the HLM.
            Returns an array of -1 if no feasible solution exists.
        required_success_probs : list
            List of the solution transition probabilities in the HLM.
            Returns a list of -1 if no feasible solution exists.
        reach_prob : float
            The HLM predicted probability of reaching the target set under the solution
            meta-policy and solution transition probabilities in the HLM.
        feasibility_flag : bool
            Flag indicating the feasibility of the bilinear program being solved.
        """
        if prob_threshold > 1 or prob_threshold < 0:
            raise RuntimeError("prob threshold is not a probability")

        # initialize gurobi model
        bilinear_model = gb.Model("abs_mdp_bilinear")

        # activate gurobi nonconvex
        bilinear_model.params.NonConvex = 2

        # dictionary for state action occupancy
        state_act_vars = dict()

        # dictionary for MDP prob variables
        MDP_prob_vars = dict()

        # dictionary for slack variables
        slack_prob_vars = dict()

        # dictionary for epigraph variables used to define objective
        MDP_prob_diff_maximizers = dict()

        # dummy action for goal state
        self.model.avail_actions[self.environment.goal_states[0]] = [0]

        # create occupancy measures, probability variables and reward variables
        for s in self.environment.state_space:
            for a in self.model.avail_actions[s]:
                state_act_vars[s, a] = bilinear_model.addVar(
                    lb=0, name="state_act_" + str(s) + "_" + str(a)
                )

        for a in self.environment.action_space:
            MDP_prob_vars[a] = bilinear_model.addVar(
                lb=0, ub=1, name="mdp_prob_" + str(a)
            )
            slack_prob_vars[a] = bilinear_model.addVar(
                lb=0, ub=1, name="slack_" + str(a)
            )

            MDP_prob_diff_maximizers[a] = bilinear_model.addVar(
                lb=0, name="mdp_prob_difference_maximizer_" + str(a)
            )

        # #epigraph variable for max probability constraint
        # prob_maximizer = bilinear_model.addVar(lb=0, name="prob_maximizer")

        # gurobi updates model
        bilinear_model.update()

        # MDP bellman or occupancy constraints for each state
        for s in self.environment.state_space:
            cons = 0
            # add outgoing occupancy for available actions

            for a in self.model.avail_actions[s]:
                cons += state_act_vars[s, a]

            # add ingoing occupancy for predecessor state actions
            for s_bar, a_bar in self.model.predecessors[s]:
                # this if-clause ensures that you don't double count reaching goal and failure
                if (
                    not s_bar == self.environment.goal_states[0]
                    and not s_bar == self.environment.fail_states[0]
                ):
                    cons -= state_act_vars[s_bar, a_bar] * MDP_prob_vars[a_bar]
            # initial state occupancy
            if s == self.environment.start_states[0]:
                cons = cons - 1

            # sets occupancy constraints
            bilinear_model.addConstr(cons == 0)

        # TODO: This can be merged with the section below like max of prob threshold and achieved
        # prob threshold constraint
        for s in self.environment.goal_states:
            bilinear_model.addConstr(state_act_vars[s, 0] >= prob_threshold)
        print("opt")

        # For each low-level component, add constraints corresponding to
        # the existing performance.
        # for s in self.S:
        for a in self.environment.action_space:
            existing_success_prob = self.get_success_prob(a)
            assert 0 <= existing_success_prob <= 1
            bilinear_model.addConstr(MDP_prob_vars[a] >= existing_success_prob)

        # If one of the components exceeds the maximum allowable training steps, upper bound its success probability.
        if max_timesteps_per_component:
            for a in self.environment.action_space:
                if (
                    self.controller_list[a].data["total_training_steps"]
                    >= max_timesteps_per_component
                ):
                    existing_success_prob = self.get_success_prob(a)
                    assert 0 <= existing_success_prob <= 1
                    print(
                        "Controller {}, max success prob: {}".format(
                            a, existing_success_prob
                        )
                    )
                    bilinear_model.addConstr(
                        MDP_prob_vars[a] <= existing_success_prob + slack_prob_vars[a]
                    )

        # set up the objective
        obj = 0

        slack_cons = 1e3
        # # Minimize the sum of success probability lower bounds

        for a in self.environment.action_space:
            obj += MDP_prob_diff_maximizers[a]
            obj += slack_cons * slack_prob_vars[a]

        # Minimize the sum of differences between probability objective and empirical achieved probabilities
        for a in self.environment.action_space:
            bilinear_model.addConstr(
                MDP_prob_diff_maximizers[a]
                >= MDP_prob_vars[a] - self.get_success_prob(a)
            )

        # set the objective, solve the problem
        bilinear_model.setObjective(obj, gb.GRB.MINIMIZE)
        bilinear_model.optimize()

        if bilinear_model.SolCount == 0:
            feasible_flag = False
        else:
            feasible_flag = True

        for a in self.environment.action_space:
            if slack_prob_vars[a].x > 1e-6:
                print(
                    "required slack value {} at action: {} ".format(
                        slack_prob_vars[a].x, a
                    )
                )

        if feasible_flag:
            # TODO: Isn't this all overwritten directly below
            # Update the requirements for the individual components
            required_success_probs = {}
            for a in self.environment.action_space:
                if a not in required_success_probs.keys():
                    required_success_probs[a] = []
                    required_success_probs[a].append(np.copy(MDP_prob_vars[a].x))
            for a in self.environment.action_space:
                self.controller_list[a].data["required_success_prob"] = np.max(
                    required_success_probs[a]
                )

            # Create a list of the required success probabilities of each of the components
            required_success_probs = [
                MDP_prob_vars[a].x for a in self.environment.action_space
            ]

            # Save the probability of reaching the goal state under the solution
            reach_prob = state_act_vars[self.environment.goal_states[0], 0].x

            # Construct the policy from the occupancy variables
            policy = np.zeros((self.model.N_S, self.model.N_A), dtype=np.float64)
            for s in self.environment.state_space:
                if len(self.model.avail_actions[s]) == 0:
                    policy[s, :] = (
                        -1
                    )  # If no actions are available, return garbage value
                else:
                    occupancy_state = np.sum(
                        [state_act_vars[s, a].x for a in self.model.avail_actions[s]]
                    )
                    # If the state has no occupancy measure under the solution, set the policy to
                    # be uniform over available actions
                    if occupancy_state == 0.0:
                        for a in self.model.avail_actions[s]:
                            policy[s, a] = 1 / len(self.model.avail_actions[s])
                    if occupancy_state > 0.0:
                        for a in self.model.avail_actions[s]:
                            policy[s, a] = state_act_vars[s, a].x / occupancy_state
        else:
            policy = -1 * np.ones((self.model.N_S, self.model.N_A), dtype=np.float64)
            required_success_probs = [
                [-1 for a in self.model.avail_actions[s]]
                for s in self.environment.state_space
            ]
            reach_prob = -1

        # Remove dummy action from goal state
        self.model.avail_actions[self.environment.goal_states[0]].remove(0)

        return policy, required_success_probs, reach_prob, feasible_flag

    # TODO: remove me?
    def reset(self):
        """
        Deselect the current controller.
        """
        self._reset()

    # TODO: remove me?
    def _reset(self):
        """
        Deselect the current controller.
        """
        self.current_controller_ind = None
        self.environment.reset()

    def save(self, save_dir):
        """
        Save the MetaController object.

        Inputs
        ------
        save_dir : string
            Absolute path to the directory that will be used to save this MetaController.
        """
        # create save directory if it doesn't exist
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        # Save the meta-model
        meta_model_file = os.path.join(save_dir, "meta_model")
        self.model.save(meta_model_file)

        # Save the meta-environment
        meta_environment_file = os.path.join(save_dir, "meta_environment")
        self.environment.save(meta_environment_file)

        # Save data from each of the sub-controllers
        for controller_ind, controller in enumerate(self.controller_list):
            controller_dir = os.path.join(save_dir, f"controller_{controller_ind}")
            controller.save(controller_dir)

        # Save data from the meta-controller
        meta_controller_file = os.path.join(save_dir, "meta_controller_data.p")
        meta_controller_data = {
            # TODO: add whatever data we think is important here
            # 'data' : self.data,
            "discount": self.discount,
            "n_controllers": len(self.controller_list),
        }
        with open(meta_controller_file, "wb") as pickleFile:
            pickle.dump(meta_controller_data, pickleFile)

    @staticmethod
    def load(load_dir: str, controller_loader, obs_mapping, environment):
        """
        Load a MetaController object

        Inputs
        ------
        load_dir : str
            Absolute path to the directory of a previously saved MetaController.
        controller_cls : lambda (load_dir) -> Controller
            The function to used to load each controller.
        obs_mapping : lambda (obs) -> int or None
            A function mapping observations in the real environment to
            high-level states. If the high-level state isn't found, this
            function returns None.
        """
        # load old results
        results = Results(load_dir=load_dir)
        results.load(load_dir)

        # load old meta-controller data
        meta_controller_file = os.path.join(load_dir, "meta_controller_data.p")
        with open(meta_controller_file, "rb") as pickleFile:
            meta_controller_data = pickle.load(pickleFile)

        # load old sub-controllers
        n_controllers = meta_controller_data["n_controllers"]
        controller_list = [
            controller_loader(os.path.join(load_dir, f"controller_{i}"), environment)
            for i in range(n_controllers)
        ]

        # load old meta-environment
        meta_environment = MetaEnvironment.load(
            os.path.join(load_dir, "meta_environment")
        )

        # load old meta-model
        meta_model = MetaModel.load(
            os.path.join(load_dir, "meta_model"), meta_environment
        )

        # create meta-controller and prep for further learning
        meta_controller = MetaController(
            meta_environment,
            controller_list,
            obs_mapping,
            discount=meta_controller_data["discount"],
            meta_model=meta_model,
        )
        meta_controller.results = results
        meta_controller.save_dir = load_dir
        meta_controller.rseed = results.data["random_seed"]

        return meta_controller

    def _init_folders(
        self,
        env_settings=None,
        experiment_name="",
        num_rollouts=300,
        save_learned_controllers=True,
        training_iters=5e4,
        prob_threshold=0.95,
    ):
        """
        A helper function to initialize folders when learning.

        Inputs
        ------
        env_settings : dict
            A dictionary of settings to use to create the environment.
        experiment_name : str
            Optionally provide a name for the experiment.
        num_rollouts : int
            The number of random rollouts to use to evaluate incremental
            subcontroller controller performance.
        save_learned_controllers : bool
            Optionally do not save the learned controllers by setting this to False.
        training_iters : int
            The number of timesteps to train each subsystem controller for.
        prob_threshold : float
            A minimum probability threshold between 0 and 1 at which to stop training.
        """
        # TODO: modify this to fit our experiment format
        base_path = os.path.abspath(os.path.curdir)
        # string_ind = base_path.find("src")
        # assert string_ind >= 0
        # base_path = base_path[0 : string_ind + 4]
        base_path = os.path.join(base_path, "data", "saved_controllers")

        # create a new directory with the current time
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d_%H-%M-%S")
        rseed = int(now.time().strftime("%H%M%S"))
        save_dir = os.path.join(base_path, dt_string + "_" + experiment_name)
        if save_learned_controllers and not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            print(f"Directory '{save_dir}' created")

        # Create object to store the results
        results = Results(
            self.controller_list,
            env_settings,
            prob_threshold,
            training_iters,
            num_rollouts,
            random_seed=rseed,
        )

        self.save_dir = save_dir
        self.results = results
        self.rseed = rseed

    def get_success_prob(self, action=None):
        """Get the current rate of success for this MetaController.

        Parameters
        ----------
        action : int, optional
            Optionally provide a meta-action to get the success probability of
            that subcontroller instead.

        Returns
        -------
        success_prob : float
            The lower-bound on the probability of success.
        """
        if action is not None:
            return self.controller_list[action].get_success_prob()

        return self._success_prob

    def learn(
        self,
        env,
        env_settings=None,
        experiment_name="",
        num_rollouts=300,
        n_steps_per_rollout=100,
        meta_controller_n_steps_per_rollout=200,
        save_learned_controllers=True,
        evaluate_controllers_progress=True,
        training_iters=5e4,
        prob_threshold=0.95,
        max_timesteps_per_component=5e5,
    ):
        """
        Train the meta-controller with a fixed computation budget.
        Inputs
        ------
        env : Environment
            The real environment to be trained in.
        env_settings : dict
            A dictionary of setting to use to create the environment.
        experiment_name : str
            Optionally provide a name for the experiment.
        num_rollouts : int
            The number of random rollouts to use to evaluate incremental
            subcontroller controller performance.
        n_steps_per_rollout:
            The number of steps per random rollout to use to evaluate
            incremental subcontroller performances.
        meta_controller_n_steps_per_rollout : int
            The number of steps per random rollout to use to evaluate
            incremental meta-controller performance.
        save_learned_controllers : bool
            Optionally do not save the learned controllers by setting this to False.
        evaluate_controllers_progress : bool
            Optionally do not evaluate controller average performance during training
            by setting this to False.
        training_iters : int
            The number of timesteps to train each subsystem controller for.
        prob_threshold : float
            A minimum probability threshold between 0 and 1 at which to stop training.
        max_timesteps_per_component : int
            The maximum number of training steps allowed for each individual
            subsystem controller.
        """

        # Create a new directory in which to save results
        if self.save_dir is None:
            self._init_folders(
                env_settings=env_settings,
                experiment_name=experiment_name,
                num_rollouts=num_rollouts,
                save_learned_controllers=save_learned_controllers,
                training_iters=training_iters,
                prob_threshold=prob_threshold,
            )
        else:
            save_learned_controllers = True  # ?

        # Seed random-number generators
        torch.manual_seed(self.rseed)
        random.seed(self.rseed)
        np.random.seed(self.rseed)
        print("Random seed: {}".format(self.results.data["random_seed"]))

        # Evaluate initial performance of controllers (they haven't learned anything yet so they will likely have no chance of success.)
        if evaluate_controllers_progress:
            print("Initial performances:")
            for controller_ind, controller in enumerate(self.controller_list):

                results = controller.eval_performance(
                    n_episodes=num_rollouts, n_steps=n_steps_per_rollout
                )
                print(
                    f' - Controller {controller_ind} achieved prob succes: {results["success_rate"]:0.3f}'
                )
            self.results.update_training_steps(0)
            self.results.update_controllers(
                self.controller_list, emperical_evaluation=evaluate_controllers_progress
            )
            self.results.save(self.save_dir)

        # Save learned controller
        if save_learned_controllers:
            for controller_ind, controller in enumerate(self.controller_list):
                controller_save_dir = os.path.join(
                    self.save_dir, f"controller_{controller_ind}"
                )
                controller.save(controller_save_dir)

        # self.results.update_training_steps(0)
        # self.results.update_controllers(self.controller_list, emperical_evaluation=evaluate_controllers_progress)
        # self.results.save(self.save_dir)

        # Calculate the initial max reach probability, policy, and empirical performance
        self.update_transition_function()
        policy, reach_prob, feasible_flag = self.model.solve_max_reach_prob_policy()
        meta_success_rate = self.eval_performance(
            env, n_episodes=num_rollouts, n_steps=meta_controller_n_steps_per_rollout
        )
        policy_lower_bound = self.verify_performance(policy=policy)
        self.results.update_composition_data(
            meta_success_rate, num_rollouts, policy, reach_prob, policy_lower_bound
        )
        self.results.save(self.save_dir)

        # Main loop of iterative compositional reinforcement learning
        total_timesteps = training_iters

        while reach_prob < prob_threshold:

            # Solve the HLM biliniear program to automatically obtain sub-task specifications.
            (
                optimistic_policy,
                required_reach_probs,
                optimistic_reach_prob,
                feasible_flag,
            ) = self.solve_low_level_requirements_action(
                prob_threshold, max_timesteps_per_component=max_timesteps_per_component
            )

            if not feasible_flag:
                print(required_reach_probs)

            # Print the empirical sub-system estimates and the sub-system specifications to terminal
            for controller_ind, controller in enumerate(self.controller_list):
                controller.data["required_success_prob"] = required_reach_probs[
                    controller_ind
                ]
                print(
                    f"Init state: {controller.environment.start_states}",
                    f"Action: {controller_ind}",
                    f"End state: {controller.environment.goal_states}",
                    f"Achieved success prob: {controller.get_success_prob()}",
                    f"Required success prob: {controller.data['required_success_prob']}",
                )

            # Decide which sub-system to train next.
            performance_gaps = []
            for controller_ind, controller in enumerate(self.controller_list):
                performance_gaps.append(
                    required_reach_probs[controller_ind] - controller.get_success_prob()
                )

            largest_gap_ind = np.argmax(performance_gaps)
            controller_to_train = self.controller_list[largest_gap_ind]

            # Train the sub-system and empirically evaluate its performance
            print(
                f"Training controller {largest_gap_ind}: performance gap = {performance_gaps[largest_gap_ind]}",
                f"({controller_to_train.data['total_training_steps']}/{max_timesteps_per_component})",
                f"for {total_timesteps} timesteps.",
            )
            controller_to_train.learn(total_timesteps=total_timesteps)
            print(f"Completed training controller {largest_gap_ind}")
            if evaluate_controllers_progress:
                controller_to_train.eval_performance(
                    n_episodes=num_rollouts, n_steps=n_steps_per_rollout
                )
            print(
                f"Achieved success prob: {controller_to_train.get_success_prob():0.3f}"
            )

            # Save learned controller
            if save_learned_controllers:
                controller_save_dir = os.path.join(
                    self.save_dir, f"controller_{largest_gap_ind}"
                )
                controller_to_train.save(controller_save_dir)

            # Calculate the initial max reach probability, policy, and empirical performance
            self.update_transition_function()
            policy, reach_prob, feasible_flag = self.model.solve_max_reach_prob_policy()
            self._success_prob = reach_prob

            meta_success_rate = self.eval_performance(
                env,
                n_episodes=num_rollouts,
                n_steps=meta_controller_n_steps_per_rollout,
            )
            policy_lower_bound = self.verify_performance(policy=policy)

            # Save results
            self.results.update_training_steps(total_timesteps)
            self.results.update_controllers(
                self.controller_list, emperical_evaluation=evaluate_controllers_progress
            )
            self.results.update_composition_data(
                meta_success_rate, num_rollouts, policy, reach_prob, policy_lower_bound
            )
            self.results.save(self.save_dir)

        self.save(self.save_dir)

    def obs_mapping(self, obs):
        """
        Map from an environment observation (state) to the corresponding
        high-level state.

        Inputs
        ------
        obs : tuple
            Tuple representing the current environment observation (state).

        Returns
        -------
        high_level_state : int, None
            The high level state corresponding to the observation or None if
            the observation does not match any high level state.
        """
        return self._obs_mapping(obs)

    # TODO: there is a lot of repetition happening here that may be unnecessary
    def predict(self, obs, deterministic=True):
        """
        Get the system's action, given the current environment observation (state)

        Inputs
        ------
        obs : np.array
            The current environment observation (state).
        deterministic (optional) : bool
            Flag indicating whether to return a deterministic action or a distribution
            over actions.
        """
        meta_state = self.obs_mapping(obs)

        # If no controller is selected, choose which controller to execute
        if self.current_controller_ind is None:
            self.current_controller_ind = self.model.predict(meta_state)

        elif meta_state in self.environment.goal_states:
            print("We reached the goal...now what?")

        # If the controller's task has been completed, grab the next one
        elif (
            meta_state
            == self.environment.successor[
                (self.environment.current_state, self.current_controller_ind)
            ]
        ):
            self.environment.step([self.current_controller_ind])
            self.current_controller_ind = self.model.predict(meta_state)

        # Use the currently selected controller
        controller = self.controller_list[self.current_controller_ind]
        action, _states = controller.predict(obs, deterministic=deterministic)

        return action, _states

    def get_performance_results(self):
        """
        Retrieves the stored performance results of the empirical rollouts and the
        formal source (if trained with a formal verifier).

        Returns
        -------
        performance_results : List[(float, float, float)]
            A list of tuples containing the timestep and the formal and empirical performance results.
        """
        formal_results_map = self.results.data["composition_policy_lower_bound"]
        empirical_data_map = self.results.data["composition_rollout_mean"]
        empirical_rollouts_map = self.results.data["composition_num_rollouts"]
        performance_results = []

        for steps in formal_results_map.keys():
            formal_result = formal_results_map[steps]
            empirical_result = (
                empirical_data_map[steps][0] / empirical_rollouts_map[steps]
            )
            performance_results.append((steps, formal_result, empirical_result))
            print(
                f"Performance at steps {steps}: verified: {formal_result}, empirical: {empirical_result}"
            )

        return performance_results

    def verify_performance(self, policy=None):
        """
        Retrieves the latest policy to follow and verify the subcontrollers for.
        These subcontrollers success rates are multiplied to get the meta policy performance.

        Parameters
        ----------
        policy : np.array, optional
            Optionally provide a policy to verify, otherwise uses the current policy.

        Returns
        -------
        performance : float
            The verified performance of the latest policy.
        """
        # Get the latest policy or use the provided one
        if policy is None:
            latest_policy = self.results.data["composition_policy"][
                self.results.elapsed_training_steps
            ]
        else:
            latest_policy = policy

        # Start in the start state and keep track of the states we visited to exit if there are loops
        # The meta environment only has 1 start and goal state therefore we grab index 0
        visited_states = set()
        current_state = self.environment.start_states[0]
        goal_state = self.environment.goal_states[0]
        performance = 1

        # Continue until we reach the goal state, or we loop (exception)
        while current_state != goal_state:
            # Check for policy loops
            if current_state in visited_states:
                print(
                    "There are loops in the latest policy, performance calculated as 0!"
                )
                return 0
            else:
                visited_states.add(current_state)

            # Get the chosen action (subcontroller index) in the policy (max value not -1) and
            action = int(np.argmax(latest_policy[current_state]))
            assert action >= 0

            # Verify the subcontroller (without print outputs)
            # Get a subcontroller success rate lower bound and multiply the performance with its success rate
            success_rate = self.get_success_prob(action)
            performance *= success_rate
            print(
                f"Success rate of subcontroller {action}: {success_rate}, overall performance now: {performance}"
            )

            # Update to the next state
            current_state = self.environment.successor[(current_state, action)]

        return performance
