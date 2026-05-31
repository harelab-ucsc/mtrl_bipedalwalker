"""
ppo_bc_sb3
==========

PPO clipped surrogate + BC auxiliary loss with DAgger expert relabeling,
layered on top of stable_baselines3. Public surface:

    from ppo_bc_sb3 import PPO_BC, PpoBcPolicy, DaggerDataset, load_expert
"""

from ppo_bc_sb3.ppo.ppo_bc import PPO_BC, load_expert
from ppo_bc_sb3.common.policies import PpoBcPolicy, PpoBcNetwork
from ppo_bc_sb3.common.buffers import RolloutBuffer, DictRolloutBuffer
from ppo_bc_sb3.common.dagger_dataset import DaggerDataset
from ppo_bc_sb3.common.on_policy_algorithm import ExpertFn, OnPolicyAlgorithm

__all__ = [
    "PPO_BC",
    "PpoBcPolicy",
    "PpoBcNetwork",
    "RolloutBuffer",
    "DictRolloutBuffer",
    "DaggerDataset",
    "OnPolicyAlgorithm",
    "ExpertFn",
    "load_expert",
]
