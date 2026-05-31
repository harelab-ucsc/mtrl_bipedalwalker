from pathlib import Path

ROOT = Path(__file__).parent.parent.parent

MODELS_DIR = ROOT / "models"
DATASET_DIR = ROOT / "datasets"
LOGS_DIR = ROOT / "logs"
SAVE_DIR = ROOT / "save"
SRC_DIR = ROOT / "src"


def rudin_distill_experiment(adversarial: bool, version: str) -> str:
    """Experiment name for a rudin distillation run.

    Adversarial task selection picks the base dir (rudin_adv vs rudin); the run
    is identified by a 3-number semver. Noise / mix mode live in the config, not
    the path, so distinguish those variants by bumping the version.
    """
    return f"{'rudin_adv' if adversarial else 'rudin'}/distill/{version}"


def rudin_distill_ckpt(adversarial: bool, version: str, ckpt: str = "best.pt") -> Path:
    """Resolve a checkpoint path for a rudin distillation run."""
    return MODELS_DIR / rudin_distill_experiment(adversarial, version) / ckpt


def rudin_pretrained_critic_experiment(adversarial: bool, version: str) -> str:
    """Experiment name for a rudin critic-pretraining run.

    Mirrors rudin_distill_experiment: ``adversarial`` selects the base dir
    (``rudin_adv`` vs ``rudin``) — here it tracks which distilled-student source
    the critic was pretrained on — and ``version`` is a 3-number semver.
    """
    return f"{'rudin_adv' if adversarial else 'rudin'}/pretrained_critic/{version}"


def rudin_finetuned_experiment(adversarial: bool, version: str) -> str:
    """Experiment name for a rudin RL-finetuning run (post critic-pretrain)."""
    return f"{'rudin_adv' if adversarial else 'rudin'}/finetuned/{version}"
