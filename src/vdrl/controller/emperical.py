# implment an emperical controller...
import os
import pickle
import copy

from vdrl.environment import Environment
from vdrl.model import Model, PPOModel

from .controller import Controller


class EmpericalSubController(Controller):
    """
    A Controller object which can only make statistical guarantees about its model's
    probability of reaching a goal state while avoiding avoid states.
    """

    def __init__(
        self,
        environment: Environment,
        model: Model,
        init_states=None,
        final_states=None,
        avoid_states=None,
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
        verifiable_properties = ()
        super().__init__(
            environment, model, verifiable_properties=verifiable_properties
        )

        self.init_states = init_states
        self.final_states = final_states
        self.avoid_states = avoid_states

        self.data = {
            "total_training_steps": 0,
            "performance_estimates": {},
            "required_success_prob": 0,
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
        self.performance = self.eval_performance(n_episodes=400, n_steps=100)

    def verify(self, property):
        """
        Evaluate the specified property of the model.

        Inputs
        ------
        property : str
            The name of the property to verify.
        """
        if property not in self.verifiable_properties:
            raise ValueError("That property is not verifiable by this subcontroller.")

        return 0.0

    def save(self, save_dir):
        """
        Save the subcontroller object.

        Inputs
        ------
        save_dir : string
            Absolute path to the directory that will be used to save this subcontroller.
        """
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        self.model.save(save_dir)
        controller_file = os.path.join(save_dir, "controller_data.p")

        controller_data = {
            "init_states": self.init_states,
            "final_states": self.final_states,
            "data": self.data,
        }

        with open(controller_file, "wb") as pickleFile:
            pickle.dump(controller_data, pickleFile)

    @staticmethod
    def load(load_dir: str, environment: Environment):
        """
        Load an EmpericalSubController object.

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

        # load the old model
        model_file = os.path.join(load_dir, "model")
        model = PPOModel.load(model_file, environment=environment)

        init_states = controller_data["init_states"]
        final_states = controller_data["final_states"]
        controller = EmpericalSubController(
            environment, model, init_states, final_states
        )
        controller.data = controller_data["data"]

        return controller

    def get_success_prob(self):
        """Get the current rate of success for this controller to be used by
        the high-level controller.

        Evaluates the model performance emperically if it has not been evaluated
        since the most recent training iteration.

        Returns
        -------
        success_prob : float
            The average rate of success predicted by emperical testing.
        """
        total_training_steps = self.data["total_training_steps"]
        if total_training_steps == 0:
            return 0

        if total_training_steps not in self.data["performance_estimates"]:
            self.eval_performance()

        return self.data["performance_estimates"][total_training_steps]["success_rate"]

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
        # environment = copy.copy(self.environment)
        # environment.initial_states = self.init_states
        # environment.goal_states = self.final_states

        success_count, avg_success_steps, total_steps = super().eval_performance(
            self.environment, n_episodes, n_steps
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
        # environment = copy.copy(self.environment)
        # environment.initial_states = self.init_states
        # environment.goal_states = self.final_states

        return super().demonstrate_capabilities(
            self.environment, n_episodes, n_steps, render
        )
