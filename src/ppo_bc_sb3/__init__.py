"""
ppo_bc_sb3
==========

local copy of the slice of stable-baselines3 we need to extend with a
behavior cloning (bc) loss term and dagger-style expert relabeling.

the folder structure mirrors stable_baselines3:

    ppo_bc_sb3/
      ppo/
        ppo_bc.py            -> mirrors stable_baselines3/ppo/ppo.py
      common/
        on_policy_algorithm.py -> mirrors stable_baselines3/common/on_policy_algorithm.py
        policies.py            -> our actor/critic policy on top of sb3's ActorCriticPolicy
        buffers.py             -> verbatim copy of stable_baselines3/common/buffers.py
        dagger_buffer.py       -> new, stores (obs, expert_action) pairs for dagger

import sites kept stable through this top-level package so callers can do:

    from ppo_bc_sb3 import PPO_BC, PpoBcPolicy, RolloutBuffer, DaggerBuffer
"""

from ppo_bc_sb3.ppo.ppo_bc import PPO_BC, load_expert
from ppo_bc_sb3.common.policies import PpoBcPolicy, PpoBcNetwork
from ppo_bc_sb3.common.buffers import RolloutBuffer, DictRolloutBuffer
from ppo_bc_sb3.common.dagger_buffer import DaggerBuffer
from ppo_bc_sb3.common.on_policy_algorithm import OnPolicyAlgorithm

__all__ = [
    "PPO_BC",
    "PpoBcPolicy",
    "PpoBcNetwork",
    "RolloutBuffer",
    "DictRolloutBuffer",
    "DaggerBuffer",
    "OnPolicyAlgorithm",
    "load_expert",
]
