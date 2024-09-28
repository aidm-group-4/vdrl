from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier
from vdrl.model.model import Model
from vdrl.environment.environment import Environment
import numpy as np
from .utils import sample_trajectory, TreeWrapper
from .evaluate import evaluate_policy


def train_viper(
    oracle: Model,
    environment: Environment,
    num_iterations: int,
    max_depth: int,
    max_leaves: int,
    sampling_n_steps: int,
    verbose: bool = False,
):
    """_summary_

    Parameters
    ----------
    oracle : Model
        The oracle model that the decision tree policy is being trained to imitate.
    environment : Environment
        The environment that the agent runs in.
    num_iterations : int
        The number of iterations to run the VIPER training for. The algorithm will sample trajectories
        and re-train the decision tree for these many iterations.
    max_depth : int
        The maximum depth of the decision tree.
    max_leaves : int
        The maximum leaves of the decision tree.
    sampling_n_steps : int
        The number of steps in the trajectory in each iteration.
    verbose : bool
        If set to True, it will print the VIPER training logs.

    Returns
    -------
    TreeWrapper
        Returns the trained decision tree in a TreeWrapper
    """
    dataset = []
    policy = None
    policies = []
    rewards = []

    for i in tqdm(range(num_iterations), disable=verbose):
        beta = 1 if i == 0 else 0
        dataset += sample_trajectory(
            oracle=oracle,
            environment=environment,
            decision_tree=policy,
            beta=beta,
            n_steps=sampling_n_steps,
        )

        clf = DecisionTreeClassifier(
            ccp_alpha=0.0001,
            criterion="entropy",
            max_depth=max_depth,
            max_leaf_nodes=max_leaves,
        )
        x = np.array([traj[0] for traj in dataset])
        y = np.array([traj[1] for traj in dataset])
        weight = np.array([traj[2] for traj in dataset])

        clf.fit(x, y, sample_weight=weight)

        policies.append(clf)
        policy = clf

        mean_reward, std_reward = evaluate_policy(
            TreeWrapper(policy), environment, n_eval_episodes=100
        )

        if verbose:
            print(f"Policy score: {mean_reward:0.4f} +/- {std_reward:0.4f}")

        rewards.append(mean_reward)

    print(f"Viper iteration complete. Dataset size: {len(dataset)}")
    best_policy = policies[np.argmax(rewards)]
    wrapper = TreeWrapper(best_policy)
    wrapper.print_info()

    return wrapper
