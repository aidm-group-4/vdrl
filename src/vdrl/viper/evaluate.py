from .utils import TreeWrapper
from vdrl.environment.environment import Environment
import numpy as np
from typing import Optional, Union, List, Tuple
import warnings


def evaluate_policy(
    model: TreeWrapper,
    environment: Environment,
    n_eval_episodes: int = 10,
    render: bool = False,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Copied from stable_baselines3.common.evaluation as done by VIPER but without the
    additional logic of vector environments. For now, we are assuming our environments
    are just singular environments. Can modify this later if needed.

    Parameters
    ----------
    model: TreeWrapper
        The RL agent you want to evaluate. We only run this for the decision tree policy.
    env: Environment
        The environment.
    n_eval_episodes: int
        Number of episode to evaluate the agent
    render: bool
        Whether to render the environment or not
    reward_threshold: Optional[float]
        Minimum expected reward per episode,
        this will raise an error if the performance is not met
    return_episode_rewards: bool
        If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    warn: bool
        If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.

    Returns
    -------
    Mean reward per episode, std of reward per episode.
    Returns ([float], [int]) when ``return_episode_rewards`` is True, first
    list containing per-episode rewards and second containing per-episode lengths
    (in number of steps).
    """
    env = environment
    if hasattr(environment, "_env"):
        env = environment._env

    is_monitor_wrapped = False
    from stable_baselines3.common.monitor import Monitor

    is_monitor_wrapped = env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    episode_rewards = []
    episode_lengths = []

    episode_count = 0
    current_reward = 0
    current_length = 0
    observation = env.reset()

    while episode_count < n_eval_episodes:
        actions, _ = model.predict(observation)
        observation, reward, terminated, truncated, info = env.step(actions)
        current_reward += reward
        current_length += 1

        if terminated:
            if is_monitor_wrapped:
                # Atari wrapper can send a "done" signal when
                # the agent loses a life, but it does not correspond
                # to the true end of episode
                if "episode" in info.keys():
                    # Do not trust "done" with episode endings.
                    # Monitor wrapper includes "episode" key in info if environment
                    # has been wrapped with it. Use those rewards instead.
                    episode_rewards.append(info["episode"]["r"])
                    episode_lengths.append(info["episode"]["l"])
                    # Only increment at the real end of an episode
                    episode_count += 1
            else:
                episode_rewards.append(current_reward)
                episode_lengths.append(current_length)
                episode_count += 1

            current_reward = 0
            current_length = 0

        if render:
            env.render()

    if len(episode_rewards) > 0:
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
    else:
        mean_reward = 0
        std_reward = 0

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, (
            "Mean reward below threshold: "
            f"{mean_reward:.2f} < {reward_threshold:.2f}"
        )
    if return_episode_rewards:
        return episode_rewards, episode_lengths

    return mean_reward, std_reward
