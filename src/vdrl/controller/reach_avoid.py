import os
import pickle
import copy

from vdrl.environment import Environment
from vdrl.model import PPOModel, Model
from decimal import Decimal
from .controller import Controller


class ReachAvoidSubController(Controller):
    """
    A Controller object which can make formal guarantees about its model's
    probability of reaching a goal state while avoiding avoid-states.
    """

    def __init__(
        self,
        environment: Environment,
        model: Model,
        init_states,
        final_states,
        avoid_states,
        p_prune=0.0000000001,
    ):
        """
        Create a Controller object

        Inputs
        ------
        environment : Environment
            The environment in which this controller operates.
        model : Model
            The model which is trained within the environment.
        init_states : states
            A list of possible initial states for this controller.
        final_states : states
            A list of valid final states for this controller.
        avoid_states : states
            A list of avoid states (results in failure) for this controller.


        verifiable_properties : List[str]
            A tuple of the properties which can be guaranteed by this controller.
        """
        verifiable_properties = (
            "reachability",  # The probability of reaching a goal state
        )
        super().__init__(
            environment, model, verifiable_properties=verifiable_properties
        )

        self.init_states = init_states
        self.final_states = final_states
        self.avoid_states = avoid_states
        # TODO: pass this as an argument.
        self.time_horizon = 100
        self.p_prune = p_prune

        self.data = {
            "total_training_steps": 0,
            "performance_estimates": {},
            "required_success_prob": 0,
        }

        self.verification_data = {}
        for property in verifiable_properties:
            self.verification_data[property] = {
                "transition_function_calls": 0,
                "max_states": -1,
            }

    def learn(self, total_timesteps=5e4):
        """
        Train the subcontroller with a fixed computation budget.

        Inputs
        ------
        total_timesteps : int
            Total number of timesteps to train the subcontroller for.
        """
        self.model.learn(total_timesteps=total_timesteps)
        self.data["total_training_steps"] += total_timesteps
        self.guarantees["reachability"] = self.verify("reachability")

    def verify(self, property, verbose=False):
        """
        Evaluate the specified property of the model.

        Inputs
        ------
        property : str
            The name of the property to verify.
        verbose : bool, optional
            Inidcate if results should be printed for each time step.
        """
        if property in self.verifiable_properties:
            if property == "reachability":
                return self._verify_reachability(verbose=verbose)
            else:
                raise ValueError(
                    "That property is not implemented yet for this subcontroller."
                )
        else:
            raise ValueError("That property is not verifiable by this subcontroller.")

    def _verify_reachability(self, verbose=False):
        """Evaluate the minimum probability of the model reaching a goal state
        while avoiding avoid-states.

        Parameters
        ----------
        verbose : bool, optional
            Inidcate if results should be printed for each time step.
        """
        min_prob = 1

        for start_state in self.init_states:
            min_prob = min(
                self._verify_reachability_from(start_state, verbose=verbose), min_prob
            )

        return min_prob

    def _verify_reachability_from(self, start_state, verbose=False):
        """A helper function to evaluate the minimum reach-avoid probability of
        the model over thhe time horizon.

        Parameters
        ----------
        start_state : list[Any]
            A state in the environment to start from.
        verbose : bool, optional
            Inidcate if results should be printed for each time step.
        """
        success = Decimal(0.0)
        failure = Decimal(0.0)
        state_distribution = {start_state: Decimal(1.0)}
        transition_function_calls = 0
        max_state_space = -1

        for t in range(self.time_horizon):
            eligible_states_for_current_time_step = 0
            next_state_distribution = {}

            for state in state_distribution.keys():

                if state_distribution[state] < self.p_prune:
                    failure += state_distribution[state]
                    continue
                eligible_states_for_current_time_step += 1
                action = self.model.predict(state)[0]
                next_state_probs = self.environment.get_transition(state, action)
                transition_function_calls += 1
                for next_state, next_state_prob in next_state_probs.items():
                    next_state_distribution[next_state] = (
                        next_state_distribution.setdefault(next_state, Decimal(0.0))
                        + Decimal(next_state_prob) * state_distribution[state]
                    )

            state_distribution = next_state_distribution

            max_state_space = max(
                max_state_space, eligible_states_for_current_time_step
            )

            for goal in self.final_states:
                success += state_distribution.setdefault(goal, Decimal(0.0))
                del state_distribution[goal]
            for avoid in self.avoid_states:
                failure += state_distribution.setdefault(avoid, Decimal(0.0))
                del state_distribution[avoid]
            if verbose:
                print(
                    f"Timestep: {t}, Success rate so far: {success}, Failure rate so far: {failure}"
                )
        self.verification_data["reachability"][
            "transition_function_calls"
        ] += transition_function_calls
        self.verification_data["reachability"]["max_states"] = max(
            self.verification_data["reachability"]["max_states"], max_state_space
        )
        return success

    def save(self, save_dir):
        """
        Save the subcontroller object.

        Inputs
        ------
        save_dir : string
            Absolute path to the directory that will be used to save this subcontroller.
        """
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

        self.model.save(save_dir)
        controller_file = os.path.join(save_dir, "controller_data.p")

        controller_data = {
            "init_states": self.init_states,
            "final_states": self.final_states,
            "avoid_states": self.avoid_states,
            "p_prune": self.p_prune,
            "data": self.data,
            "guarantees": self.guarantees,
            "verification_data": self.verification_data,
        }

        with open(controller_file, "wb") as pickleFile:
            pickle.dump(controller_data, pickleFile)

    @staticmethod
    def load(load_dir: str, environment: Environment, p_prune: float = 1e-10):
        """
        Load a ReachAvoidSubController object.

        Inputs
        ------
        load_dir : str
            Absolute path to the directory of a previously saved ReachAvoidSubController.
        environment : Environment
            The Environment in which the subcontrollers operate.
        """
        controller_file = os.path.join(load_dir, "controller_data.p")
        with open(controller_file, "rb") as pickleFile:
            controller_data = pickle.load(pickleFile)

        # Load the old model
        model_file = os.path.join(load_dir, "model")
        model = PPOModel.load(model_file, environment=environment)

        # Initialize the model with the correct data
        init_states = controller_data["init_states"]
        final_states = controller_data["final_states"]
        avoid_states = controller_data.get("avoid_states", [])
        p_prune = controller_data.get("p_prune", p_prune)
        controller = ReachAvoidSubController(
            environment, model, init_states, final_states, avoid_states, p_prune=p_prune
        )

        # Load the remaining non-initialization data
        controller.data = controller_data["data"]
        if "guarantees" in controller_data:
            controller.guarantees = controller_data["guarantees"]

        # Not loading the verification data here since it will be collected upon verification.

        return controller

    def get_success_prob(self):
        """Get the current rate of success for this controller to be used by
        the high-level controller.

        Runs the verifier if it hasn't been run yet.

        Returns
        -------
        success_prob : float
            The lower-bound on the probability of success for the model.
        """
        if self.guarantees["reachability"] == 0:
            self.guarantees["reachability"] = self.verify("reachability")
        return float(self.guarantees["reachability"])

    def eval_performance(self, n_episodes=400, n_steps=100):
        """
        Perform empirical evaluation of the performance of the learned controller.

        Inputs
        ------
        n_episodes : int
            Number of episodes to rollout for evaluation.
        n_steps : int
            Length of each episode.
        """

        # Create a smaller environment just for this task
        environment = copy.copy(self.environment)
        environment.initial_states = self.init_states
        environment.goal_states = self.final_states

        success_count, avg_success_steps, total_steps = super().eval_performance(
            environment, n_episodes, n_steps
        )

        results = {
            "success_count": success_count,
            "success_rate": success_count / n_episodes,
            "num_trials": n_episodes,
            "avg_num_steps": avg_success_steps,
        }

        self.data["performance_estimates"][self.data["total_training_steps"]] = results
        return results

    def demonstrate_capabilities(self, n_episodes=5, n_steps=100, render=True):
        """
        Demonstrate the capabilities of the learned controller in the training
        environment.

        Inputs
        ------
        n_episodes : int
            Number of episodes to rollout for evaluation.
        n_steps : int
            Length of each episode.
        render (optional) : bool
            Whether or not to render the environment at every timestep.
        """
        # Create a smaller environment just for this task
        environment = copy.copy(self.environment)
        environment.initial_states = self.initial_states
        environment.goal_states = self.final_states

        return super().demonstrate_capabilities(
            environment, n_episodes, n_steps, render
        )
