from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """
    Sets `_difficulty` (0.0 → 1.0) on every training env each step,
    scaled linearly against the run's total timestep budget.
    Envs read this attribute in reset() to interpolate randomization ranges.
    """

    def __init__(self, total_timesteps: int):
        super().__init__()
        self._total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        difficulty = min(1.0, self.num_timesteps / self._total_timesteps)
        self.training_env.set_attr("_difficulty", difficulty)
        return True
