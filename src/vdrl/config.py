from pydantic import BaseModel, Field
from typing import List, Tuple, Optional, Union, Literal, Any
from enum import Enum
from pydantic import field_validator


class VerificationMethod(str, Enum):
    """Defines all the verification methods supported for the learned sub-policies"""

    EMPERICAL = "emperical"
    VIPER = "viper"
    EXHAUSTIVE = "exhaustive"


class Problems(str, Enum):
    """Defines all the problems that are implemented. For now, a problem is analogous to an environment"""

    MINIGRID = "minigrid_labyrinth"


class LowerLevelPolicies(str, Enum):
    """Defines all the lower level policies that are implemented."""

    PPO = "PPO"


class DecompositionConfig(BaseModel):
    """Defines the decomposition of the environment. A Lower Level Policy will be trained on each decomposition.

    initial_states:
      The list of initial states for this decomposition. The integer represents
      the index of the state in the high-level states.
    final_states:
      The list of final states for this decomposition. The integer represents
      the index of the state in the high-level states.
    """

    initial_states: List[int]
    final_state: int


class EnvironmentConfig(BaseModel):
    """Defines the problem-independent configuration of the environment.

    start_states:
      The list of start states of the environment. The integers represents the
      indices of the state in the high-level states.
    avoid_states:
      The list of states that need to be avoided. If the agent reaches the
      avoid states, the task will fail and terminate. The integer represents
      the index of the state in the high-level states.
    reach_states:
      The list of states that the agent should reach but are not the goal (?).
      The integer represents the index of the state in the high-level states.
    goal_states:
      The list of goal states of the environment. The integers represent the
      indices of the states in the high-level states.
    decompositions:
      The list of decompositions of the environment. The decompositions can be
      thought of as the edges in the decomposition graph. high_level_states: A
      list of all the high level states of the environment. The
      high_level_states can be though of as the nodes in the decomposition
      graph.
    """

    start_states: List[int]
    avoid_states: List[int]
    reach_states: List[int]
    goal_states: List[int]
    decompositions: List[DecompositionConfig]


class MinigridConfig(BaseModel):
    """Defines the problem specific configuration for the MiniGrid Labyrinth environment.

    env_name:
      Uniquely identifies the MiniGrid config. If env_name is set to
      "minigrid_labyrinth" in the config file, Pydantic will load the
      env_config into a MinigridConfig object.
    width:
      The width of the grid.
    height:
      The height of the grid.
    walls:
      List of all the walls in the environment.
    doors:
      List of all the doors in the environment.
    """

    problem_name: Literal[Problems.MINIGRID]
    slip_p: float
    width: int
    height: int
    walls: List[Tuple]
    doors: List[Tuple]


class DiscreteMappingConfig(BaseModel):
    """
    Defines the problem specific mapping from high level states to discrete states.

    high_level_states:
        A list of lists of minigrid states which correspond to indexed high level states.
        A high level state could include multiple minigrid states
    """

    problem_name: Literal[Problems.MINIGRID]
    high_level_states: List[List[Any]]


class LLPBaseConfig(BaseModel):
    """Defines the base configuration for a lower level policy. These are fields that every LLP needs to define.

    n_epochs:
      Number of epochs to run train the sub-ppolicy for.
    batch_size:
      Batch size for the sub-policy.
    learning_rate:
      Learning rate for the sub-policy.
    is_verified:
      Whether it is formally verified or not. If not, emperical rollouts are used.
    """

    n_epochs: int
    batch_size: int
    learning_rate: float


class PPOConfig(LLPBaseConfig):
    """Defines the LLP Configuration for a PPO based sub-policy. Inherits from LLPBaseConfig and adds some additional
    configuration fields for PPO. Refer here for documentation - https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

    llp_name:
      Uniquely identifies the PPOConfig. If llp_name is set to "PPO" in the config file,
      Pydantic will load the llp_config into a PPOConfig object.
    n_steps:
      The number of steps to run for each environment per update.
    gae_lambda:
      Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
    gamma:
      Discount factor.
    ent_coef:
      Entropy coefficient for the loss calculation.
    clip_range:
      Clipping parameter for the value function.
    """

    llp_name: Literal[LowerLevelPolicies.PPO]
    n_steps: int
    gae_lambda: float
    gamma: float
    ent_coef: float
    clip_range: float


class ViperConfig(BaseModel):
    """The configuration for VIPER verification.
    verification_method:
      Uniquely identifies the ViperConfig. If verification_method is set to "viper" in the config file,
      Pydantic will load the verification_config into a ViperConfig object.
    num_iterations:
      The number of iterations to run the VIPER training for. The algorithm will sample trajectories
      and re-train the decision tree for these many iterations.
    max_depth:
      The maximum depth of the decision tree.
    max_leaves:
      The maximum leaves of the decision tree.
    sampling_n_steps:
      The number of steps in the trajectory in each iteration.
    verbose:
      If set to True, it will print the VIPER training logs.

    """

    verification_method: Literal[VerificationMethod.VIPER]
    num_iterations: int
    max_depth: int
    max_leaves: int
    sampling_n_steps: int
    verbose: bool = False


class ExhaustiveConfig(BaseModel):

    verification_method: Literal[VerificationMethod.EXHAUSTIVE]


class EmpericalConfig(BaseModel):
    """The configuration for emperical verification.

    verification_method:
        Uniquely identifies the EmpericalConfig. If verification_method is set to "emperical" in the config file,
        Pydantic will load the verification_config into a EmpericalConfig object.
    """

    verification_method: Literal[VerificationMethod.EMPERICAL]


class ExperimentConfig(BaseModel):
    """Defines an overall experiment configuration.

     load_dir:
        If specified, the sub-policies will be loaded from this directory.
     save_dir:
        The directory where trained sub-policies will be saved.
     env_config:
        This describes the start, goal and reach states of the environment and how the environment is decomposed.
     problem_config:
        Each problem will have its own configuration which provides information about how to build the environment like
       world objects, split probability etc. The corresponding configuration object is loaded based on the discriminator field (env_name).
    llp_config:
      Each lower-level-policy we define will have its own configuration. The corresponding configuration object is loaded based on the discriminator
      field (llp_name).
    verification_config:
      The configuration for the verification method that is going to be used to verify the policy.

    """

    load_dir: Optional[str] = None
    save_dir: str
    env_config: EnvironmentConfig
    mapping_config: Optional[Union[DiscreteMappingConfig]] = Field(
        ..., discriminator="problem_name"
    )
    problem_config: Union[MinigridConfig] = Field(..., discriminator="problem_name")
    llp_config: Union[PPOConfig] = Field(..., discriminator="llp_name")
    verification_config: Union[ViperConfig, EmpericalConfig, ExhaustiveConfig] = Field(
        ..., discriminator="verification_method"
    )
