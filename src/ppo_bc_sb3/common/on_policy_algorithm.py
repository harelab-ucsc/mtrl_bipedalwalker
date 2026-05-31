"""
ppo_bc_sb3.common.on_policy_algorithm
=====================================

OnPolicyAlgorithm extended with a DAgger expert-relabeling hook. After the
student policy picks an action the matching expert is queried on the same obs,
and the (obs, expert_action) pair is appended to ``self.demo_dataset`` for the
BC loss in PPO_BC.train().

``experts`` is keyed by task tuple (the last ``task_bits`` of obs identify the
task) and each value is a callable ``(obs[N, obs_dim]) -> action[N, act_dim]``
so the algorithm stays agnostic to any single env's observation layout.
"""

from __future__ import annotations

import sys
import time
import warnings
from typing import Any, Callable, TypeVar, cast

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

from ppo_bc_sb3.common.buffers import DictRolloutBuffer, RolloutBuffer
from ppo_bc_sb3.common.dagger_dataset import DaggerDataset
from mdp.bipedal_walker.tasks import SINGLE_TASKS, TaskSpec, resolve_task
from wrappers.ppo_bc.ppo_bc_env import RlFTEnv

# An expert is any callable mapping raw obs (already in the env's full layout)
# to actions. Wrapping in a callable lets each expert slice/translate the obs
# to its own training-time layout without leaking that detail into this class.
ExpertFn = Callable[[np.ndarray], np.ndarray]

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm", bound="OnPolicyAlgorithm")

# ---- adversarial task selection ------------------------------------------------
# Run the current policy through each isolated task to get a per-task time-alive
# score, then convert scores → sampling PMF (harder = more frequent). Mirrors
# scripts/distillation/train.py: getTaskPMF / evaluate. The eval budget per task
# and the adversarial/uniform mix are per-run config (see OnPolicyAlgorithm args).
# The directional single tasks: walk_forward, walk_backward, flamingo, tilt. Walk
# is split by direction so the PMF can up-weight the weaker direction on its own.
_ADVERSARIAL_TASKS: tuple[TaskSpec, ...] = SINGLE_TASKS


def _task_pmf_from_scores(scores: list[float], k: float) -> list[float]:
    """Port of distill's getTaskPMF: weight by (max - score) so worst task gets
    the most mass, then mix with uniform by ``k``."""
    w = [max(scores) - s for s in scores]
    sum_w = sum(w)
    U = [1.0 / len(scores)] * len(scores)
    P = [U[i] if sum_w == 0 else w[i] / sum_w for i in range(len(scores))]
    return [k * p + (1.0 - k) * u for p, u in zip(P, U)]


class OnPolicyAlgorithm(BaseAlgorithm):
    """SB3-compatible on-policy base with a DAgger expert-relabeling hook.

    Parameter semantics mirror ``stable_baselines3.common.on_policy_algorithm``;
    see the SB3 docs for the inherited args. DAgger-specific args:

    * ``experts``: ``{task_name: callable(obs[N, D]) -> act[N, A]}`` where task_name
      is a directional task (walk_forward / walk_backward / flamingo / tilt).
    * ``task_bits``: number of trailing obs dims that identify the task tuple.
    * ``act_var_floor``: additive variance bias / floor on the *student* policy's
      action distribution (passed through to PpoBcPolicy via policy_kwargs); the
      expert demos themselves are stored noise-free.
    * ``dagger_max_size``: cap on aggregated demos (None disables eviction).
    """

    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(
        self,
        policy: str | type[ActorCriticPolicy],
        env: GymEnv | str,
        experts: dict[str, ExpertFn],
        task_bits: int,
        act_var_floor: float,
        learning_rate: float | Schedule,
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        bc_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        collect_data: bool = True,
        adversarial_ag: bool = False,
        adversarial_eval_env: RlFTEnv | None = None,
        adversarial_eval_steps_per_task: int = 10000,
        adversarial_k: float = 0.85,
        dagger_max_size: int | None = None,
        rollout_buffer_class: type[RolloutBuffer] | None = None,
        rollout_buffer_kwargs: dict[str, Any] | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        monitor_wrapper: bool = True,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: tuple[type[spaces.Space], ...] | None = None,
    ):
        # forward to sb3's BaseAlgorithm which handles env wrapping, seeds,
        # logger, callback plumbing, etc.
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.bc_coef = bc_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}

        # DAgger plumbing. `experts` maps task_id_tuple -> callable on raw obs.
        # demo_dataset is created here (before _setup_model) so that subclasses
        # constructed without an _init_setup_model call can still inspect it.
        self.experts = experts
        # additive variance bias / floor on the student action distribution.
        # It lives on the policy, so thread it into policy_kwargs before
        # _setup_model builds the policy (PpoBcPolicy consumes act_var_floor).
        self.act_var_floor = act_var_floor
        self.policy_kwargs["act_var_floor"] = act_var_floor
        self.task_bits = task_bits
        # when False, skip expert relabeling + demo buffer growth entirely
        # (used by critic-pretrain runs where the BC loss is unused).
        self.collect_data = collect_data
        self.demo_dataset = DaggerDataset(device=self.device, max_size=dagger_max_size)
        # per-rollout expert query counters, dumped to TB in dump_logs().
        self._expert_queries_rollout: dict[str, int] = {k: 0 for k in self.experts.keys()}
        # where to save the current dagger dataset
        self.dataset_save_path: str | None = None
        # whether to do adversaral aggregation
        self.adversarial_ag: bool = adversarial_ag
        # single (non-vectorized) RlFTEnv used by eval_expert_task_performance
        # to score each isolated task. Constructed by the caller so the
        # algorithm stays agnostic to RlFTEnv constants.
        self.adversarial_eval_env: RlFTEnv | None = adversarial_eval_env
        # per-rescore eval budget per task, and the adversarial/uniform mix.
        self.adversarial_eval_steps_per_task: int = adversarial_eval_steps_per_task
        self.adversarial_k: float = adversarial_k

        if _init_setup_model:
            self._setup_model()

    # ------------------------------------------------------------------
    # dagger setup
    # ------------------------------------------------------------------
    
    def set_dataset_save_path(self, path: str):
        self.dataset_save_path = path
    
    def save_dataset(self):
        if self.dataset_save_path is None:
            warnings.warn("Dataset save path not specified, dataset not saved!")
            return 
        
        self.demo_dataset.dump(self.dataset_save_path)
    
    # ------------------------------------------------------------------
    # model setup
    # ------------------------------------------------------------------

    def _setup_model(self) -> None:
        # called once at the end of __init__ (and again when loading from disk).
        # builds the lr schedule, the rollout buffer, and the policy network.
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # pick the right rollout buffer flavor for the obs space if the caller
        # did not specify one explicitly.
        if self.rollout_buffer_class is None:
            if isinstance(self.observation_space, spaces.Dict):
                self.rollout_buffer_class = DictRolloutBuffer
            else:
                self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )
        # construct the policy. policy_class comes from BaseAlgorithm which
        # resolved the string alias (e.g. "MlpPolicy") to the actual class.
        self.policy = self.policy_class(  # type: ignore[assignment]
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)
        self._maybe_recommend_cpu()

    def _maybe_recommend_cpu(self, mlp_class_name: str = "ActorCriticPolicy") -> None:
        policy_class_name = self.policy_class.__name__
        if self.device != th.device("cpu") and policy_class_name == mlp_class_name:
            warnings.warn(
                f"You are trying to run {self.__class__.__name__} on the GPU, "
                "but it is primarily intended to run on the CPU when not using a CNN policy "
                f"(you are using {policy_class_name} which should be a MlpPolicy). "
                "See https://github.com/DLR-RM/stable-baselines3/issues/1245 "
                "for more info. "
                "You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU."
                "Note: The model will train, but the GPU utilization will be poor and "
                "the training might take longer than on CPU.",
                UserWarning,
            )

    # ---- rollout collection ------------------------------------------------

    def _predict_actions(
        self, obs: np.ndarray | dict[str, np.ndarray]
    ) -> tuple[np.ndarray, th.Tensor, th.Tensor]:
        """Forward the policy under no-grad for one env-step decision."""
        with th.no_grad():
            obs_tensor = obs_as_tensor(obs, self.device)  # type: ignore[arg-type]
            actions, values, log_probs = self.policy(obs_tensor)
        return actions.cpu().numpy(), values, log_probs

    def _collect_expert_actions(
        self, obs: np.ndarray | dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Poll the task-matched expert for each env and return
        (obs, expert_act, has_expert). Envs that share a task tuple are
        batched into a single expert call. Expert actions are stored as-is
        (noise-free); exploration variance is injected on the student side via
        the policy's act_var_floor, not here. `has_expert` is a [N] bool mask
        marking which envs actually had a matching expert — used by the
        caller to filter DAgger aggregation so combined / unmatched tasks
        don't pollute the buffer with zero-action pairs.
        """
        assert not isinstance(obs, dict), "DAgger collection does not work with dict observations"

        obs = cast(np.ndarray, obs)
        task_ids = obs[:, -self.task_bits:]
        # effective cmd_vel (its sign selects walk_forward vs walk_backward).
        # Layout: [..., cmd_vel, cmd_tilt, *task_bits], so cmd_vel = -task_bits - 2.
        cmd_vels = obs[:, -self.task_bits - 2]
        n_envs = obs.shape[0]

        act_shape = self.action_space.shape or ()
        actions = np.zeros((n_envs, *act_shape), dtype=np.float32)
        has_expert = np.zeros(n_envs, dtype=bool)

        # Resolve each env to its directional task name; experts are keyed by name
        # (walk_forward / walk_backward / flamingo / tilt). Combined / unrecognized
        # task vectors resolve to "" → no matching expert → filtered out below.
        names = np.empty(n_envs, dtype=object)
        for i in range(n_envs):
            spec = resolve_task(task_ids[i], float(cmd_vels[i]))
            names[i] = spec.name if spec is not None else ""

        for task_name, expert_fn in self.experts.items():
            mask = names == task_name
            n_matched = int(mask.sum())
            if n_matched == 0:
                continue
            actions[mask] = expert_fn(obs[mask])
            has_expert |= mask
            self._expert_queries_rollout[task_name] = (
                self._expert_queries_rollout.get(task_name, 0) + n_matched
            )

        return obs, actions, has_expert
    
    def _clip_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        rescale (if squash_output) or clip (otherwise) the policy's raw output
        so it lies inside the env action space. returns the numpy array that is
        actually fed to env.step. note the rollout buffer stores the
        unclipped, raw policy output (that's what gradients flow through).
        """
        clipped = actions
        if isinstance(self.action_space, spaces.Box):
            if self.policy.squash_output:
                clipped = self.policy.unscale_action(clipped)
            else:
                # unbounded gaussian samples can land outside [low, high]; clip.
                clipped = np.clip(actions, self.action_space.low, self.action_space.high)
        return clipped

    def _step_envs(
        self, env: VecEnv, clipped_actions: np.ndarray
    ) -> tuple[np.ndarray | dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """
        single env.step call. isolated so a custom training loop could insert
        timing, logging, or a different vec env api without touching the rest.
        """
        new_obs, rewards, dones, infos = env.step(clipped_actions)
        return new_obs, rewards, dones, infos # type: ignore

    def _handle_timeout_bootstrap(
        self, rewards: np.ndarray, dones: np.ndarray, infos: list[dict[str, Any]]
    ) -> np.ndarray:
        """
        for truncations (done due to timelimit, not actual terminal state) we
        add gamma * V(s_terminal) to the reward so the bootstrap is not lost.
        sb3 issue #633. mutates `rewards` in place but also returns it.
        """
        for idx, done in enumerate(dones):
            if (
                done
                and infos[idx].get("terminal_observation") is not None
                and infos[idx].get("TimeLimit.truncated", False)
            ):
                terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                with th.no_grad():
                    terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                rewards[idx] += self.gamma * terminal_value
        return rewards

    def _store_transition(
        self,
        rollout_buffer: RolloutBuffer,
        obs: np.ndarray | dict[str, np.ndarray],
        actions: np.ndarray,
        rewards: np.ndarray,
        episode_starts: np.ndarray,
        values: th.Tensor,
        log_probs: th.Tensor,
    ) -> None:
        """
        push one transition into the rollout buffer. thin shim around
        RolloutBuffer.add so subclasses can stash additional fields (e.g.
        expert actions stored timestep-aligned with rollouts).
        """
        rollout_buffer.add(obs, actions, rewards, episode_starts, values, log_probs) # type: ignore

    def _compute_last_values_and_returns(
        self,
        rollout_buffer: RolloutBuffer,
        new_obs: np.ndarray | dict[str, np.ndarray],
        dones: np.ndarray,
    ) -> None:
        """
        post-rollout step: bootstrap V(s_final) with the current critic and
        call compute_returns_and_advantage to fill in returns and gae
        advantages on the rollout buffer.
        """
        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> tuple[bool, list[tuple[np.ndarray, np.ndarray]]]:
        """Collect one rollout and the matching DAgger expert demos.

        Returns ``(continue_training, Di)`` where ``Di`` is the list of
        ``(obs, expert_action)`` pairs collected this rollout (one per step
        per env). ``continue_training`` is False if a callback asked to stop.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # eval mode for batch norm / dropout. the optimization step in train()
        # flips it back to True.
        self.policy.set_training_mode(False)

        n_steps = 0
        Di: list[tuple[np.ndarray, np.ndarray]] = []  # current rollout dataset
        # reset per-rollout query counters — dumped at the next dump_logs call.
        self._expert_queries_rollout = {k: 0 for k in self.experts.keys()}

        rollout_buffer.reset()
        # gSDE: resample exploration weights at the start of every rollout.
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            # gSDE periodic resampling (different cadence from rollout reset).
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            # 1) ask the policy for an action on the current obs.
            actions, values, log_probs = self._predict_actions(self._last_obs)

            # DAgger: query the task-matched expert on the same obs.
            # Envs without a matching expert (e.g. combined-task rollouts) are
            # filtered out so they don't poison the buffer with zero actions.
            # Skipped when collect_data=False (e.g. critic-only pretraining).
            if self.collect_data:
                obs_e, act_e, has_exp = self._collect_expert_actions(self._last_obs)
                Di += [
                    (obs_e[i].copy(), act_e[i].copy())
                    for i in range(obs_e.shape[0])
                    if has_exp[i]
                ]

            # 2) prepare the action for env.step (clip / unscale).
            clipped_actions = self._clip_actions(actions)

            # 3) step the vec env.
            new_obs, rewards, dones, infos = self._step_envs(env, clipped_actions)

            self.num_timesteps += env.num_envs

            # 4) callback hooks. _on_step can stop training early by returning False.
            callback.update_locals(locals())
            if not callback.on_step():
                return False, Di

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # match storage shape expected by rollout buffer for discrete actions.
                actions = actions.reshape(-1, 1)

            # 5) timelimit-truncation bootstrap. infos may carry
            #    terminal_observation + TimeLimit.truncated flags; if so, we
            #    bootstrap the reward with gamma * V(s_terminal).
            rewards = self._handle_timeout_bootstrap(rewards, dones, infos)

            # 6) store this transition into the rollout buffer.
            self._store_transition(
                rollout_buffer,
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        # rollout ended. fold the bootstrap value and run gae / return computation.
        self._compute_last_values_and_returns(rollout_buffer, new_obs, dones)

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True, Di

    # ------------------------------------------------------------------
    # training step
    # ------------------------------------------------------------------

    def train(self) -> None:
        """
        Consume the current rollout buffer and update policy parameters.
        Implemented by individual algorithms (PPO_BC overrides this).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # logging + outer learn loop
    # ------------------------------------------------------------------

    def dump_logs(self, iteration: int = 0) -> None:
        # standard sb3 logging dump: fps, time elapsed, rollout reward stats.
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        if iteration > 0:
            self.logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success_rate", safe_mean(self.ep_success_buffer))

        # DAgger telemetry: dataset size + per-task composition + per-rollout queries.
        ds_len = len(self.demo_dataset)
        self.logger.record("dagger/dataset_size", ds_len)
        if ds_len > 0:
            counts = self.demo_dataset.task_counts(self.task_bits)
            for tag, c in counts.items():
                self.logger.record(f"dagger/task_pct_{tag}", c / ds_len)
        for tag, n_q in self._expert_queries_rollout.items():
            self.logger.record(f"dagger/expert_queries_{tag}", n_q)

        self.logger.dump(step=self.num_timesteps)

    def eval_expert_task_performance(self) -> list[float]:
        """Score the current policy on each isolated task and push the
        resulting adversarial PMF to every training env.

        Returns the per-task mean time-alive in step units (same order as
        ``_ADVERSARIAL_TASKS`` = walk_forward, walk_backward, flamingo, tilt).
        Returns ``[]`` and leaves env probs untouched if adversarial selection is
        disabled, the eval env is missing, or the env's allowed_task_mixing is not
        exactly the directional single tasks.
        """
        if not self.adversarial_ag or self.adversarial_eval_env is None:
            return []
        env = self.adversarial_eval_env
        if list(env._allowed_tasks) != list(_ADVERSARIAL_TASKS):
            warnings.warn(
                "adversarial task selection requires allowed_task_mixing == "
                "SINGLE_TASKS (walk_forward, walk_backward, flamingo, tilt); "
                "skipping eval and PMF update."
            )
            return []

        self.policy.set_training_mode(False)
        scores: list[float] = []
        for task in _ADVERSARIAL_TASKS:
            env.set_forced_task(task)
            obs, _ = env.reset()
            times: list[int] = []
            alive = 0
            for _ in range(self.adversarial_eval_steps_per_task):
                with th.no_grad():
                    act_t, _, _ = self.policy(obs_as_tensor(obs[None], self.device))
                act = self._clip_actions(act_t.cpu().numpy())[0]
                obs, _, term, trunc, _ = env.step(act)
                if term or trunc:
                    times.append(alive)
                    alive = 0
                    obs, _ = env.reset()
                else:
                    alive += 1
            times.append(alive)
            scores.append(float(np.mean(times)))
        env.set_forced_task(None)

        probs = _task_pmf_from_scores(scores, self.adversarial_k)
        # broadcast the new PMF to every SubprocVecEnv worker. set_attr would
        # only write to the outer Monitor wrapper; env_method routes through
        # gym.Wrapper.__getattr__ to the inner RlFTEnv.set_task_sample_probs.
        assert self.env is not None
        self.env.env_method("set_task_sample_probs", tuple(probs))

        for task, s, p in zip(_ADVERSARIAL_TASKS, scores, probs):
            self.logger.record(f"adversarial/time_alive_{task.name}", s)
            self.logger.record(f"adversarial/prob_{task.name}", p)

        return scores

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        # outer training lifecycle. collect_rollouts -> train -> repeat until
        # num_timesteps >= total_timesteps or a callback asks to stop.
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training, Di = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps
            )

            if not continue_training:
                break

            # aggregate this rollout's expert demos into the persistent D.
            # When collect_data=False, Di is always empty and we also skip the
            # save-on-disk call to avoid dumping an empty npz every rollout.
            if self.collect_data:
                self.demo_dataset.extend(Di)
                if self.dataset_save_path is not None:
                    self.save_dataset()

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # periodic log dump (rollout stats from the just-collected batch).
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self.dump_logs(iteration)

            # train on the just-collected rollout buffer.
            self.train()

            # adversarial task selection: rescore tasks with the just-updated
            # policy and broadcast a new sampling PMF to the training envs.
            # No-op when adversarial_ag is False or guards in the eval routine
            # don't pass.
            if self.adversarial_ag:
                self.eval_expert_task_performance()

        callback.on_training_end()

        return self

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.optimizer"]
        return state_dicts, []

    def _excluded_save_params(self) -> list[str]:
        # sb3 JSON-serializes everything in __dict__ that isn't in this list.
        # `experts` has tuple keys and callable values (both unserializable);
        # `demo_dataset` carries numpy arrays + a torch device; the per-rollout
        # query counter also has tuple keys. None of these need to survive a
        # save/load cycle — experts are re-supplied at construction time on load.
        return super()._excluded_save_params() + [
            "experts",
            "demo_dataset",
            "_expert_queries_rollout",
        ]
