from dataclasses import asdict, dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

from pyrallis import field


@dataclass
class BCTrainConfig:
    # wandb params
    project: str = "FAWAC_BulletSafety"
    group: str = None
    name: Optional[str] = None
    prefix: Optional[str] = "IQL-Cost"
    suffix: Optional[str] = ""
    logdir: Optional[str] = "DSRL"
    verbose: bool = True
    # Dataset params (from DSRL)
    outliers_percent: float = None
    noise_scale: float = None
    inpaint_ranges: Tuple[Tuple[float, float, float, float], ...] = None
    epsilon: float = None
    density: float = 1.0
    max_action: float = 1.0
    num_workers: int = 8
    # Training params
    task: str = "OfflineAntCircle-v0"
    seed: int = 0
    cost_limit: int = 20
    reward_scale: float = 1.0
    cost_scale: float = 1.0
    episode_len: int = 300
    device: str = "cuda"
    eval_episodes: int = 10 # Number of episodes to eval
    eval_every: int = 20_000  # Eval every n steps
    max_timesteps: int = int(5e5)  # Max timesteps to update
    # IQL params
    batch_size: int = 1024  # Batch size for all networks
    discount: float = 0.99  # Discount factor
    tau: float = 0.005  # Target network update rate
    beta: float = 2.0  # Inverse temperature. Small inv temp -> BC, big beta -> optimizing objective
    iql_tau: float = 0.7  # Coefficient for asymmetric loss
    vf_lr: float = 3e-4  # V function learning rate
    qf_lr: float = 3e-4  # Critic learning rate
    actor_lr: float = 3e-4  # Actor learning rate
    exp_adv_max_cost: float = 200.0  # Max advantage for exp weight (cost)
    exp_adv_max_reward: float = 200.0  # Max advantage for exp weight (reward)
    # FAWAC
    lag_max: float = 20.0
    cost_planning_steps: float = 100.0
    obj_method: str = "penalty" # penalty or statewise_lagrangian



@dataclass
class BCCarCircleConfig(BCTrainConfig):
    # training params
    task: str = "OfflineCarCircle-v0"
    episode_len: int = 300
    pass


@dataclass
class BCDroneRunConfig(BCTrainConfig):
    # training params
    task: str = "OfflineDroneRun-v0"
    episode_len: int = 200


@dataclass
class BCDroneCircleConfig(BCTrainConfig):
    # training params
    task: str = "OfflineDroneCircle-v0"
    episode_len: int = 300


@dataclass
class BCCarRunConfig(BCTrainConfig):
    # training params
    task: str = "OfflineCarRun-v0"
    episode_len: int = 200


@dataclass
class BCBallRunConfig(BCTrainConfig):
    # training params
    task: str = "OfflineBallRun-v0"
    episode_len: int = 100


@dataclass
class BCBallCircleConfig(BCTrainConfig):
    # training params
    task: str = "OfflineBallCircle-v0"
    episode_len: int = 200




BC_DEFAULT_CONFIG = {
    # bullet_safety_gym
    "OfflineCarCircle-v0": BCCarCircleConfig,
    "OfflineDroneRun-v0": BCDroneRunConfig,
    "OfflineDroneCircle-v0": BCDroneCircleConfig,
    "OfflineCarRun-v0": BCCarRunConfig,
    "OfflineBallCircle-v0": BCBallCircleConfig,
    "OfflineBallRun-v0": BCBallRunConfig,
}