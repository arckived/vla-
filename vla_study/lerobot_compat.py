"""All LeRobot imports live here, so when LeRobot renames something you fix it in ONE place.

LeRobot moves fast (scripts were renamed from `lerobot/scripts/train.py` to the
`lerobot-train` command, and policies gained separate pre/post-processors).
If something here breaks, check the installed source with:
    python -c "import lerobot, os; print(os.path.dirname(lerobot.__file__))"
and grep for the function name.
"""
from __future__ import annotations

import importlib


def import_eval_module():
    """Return the module that implements `lerobot-eval`."""
    errors = []
    for name in ("lerobot.scripts.lerobot_eval", "lerobot.scripts.eval"):
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as e:
            errors.append(f"{name}: {e}")
    raise ImportError("Could not find LeRobot's eval script. Tried:\n  " + "\n  ".join(errors))


def load_policy(path: str, device: str = "cuda"):
    """Load any pretrained LeRobot policy (SmolVLA, ACT, ...) from the Hub or a local folder."""
    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class

        cfg = PreTrainedConfig.from_pretrained(path)
        policy_cls = get_policy_class(cfg.type)
    except Exception as e:  # fall back to SmolVLA directly
        print(f"[compat] generic loader failed ({e!r}); falling back to SmolVLAPolicy")
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as policy_cls

    policy = policy_cls.from_pretrained(path)
    policy.to(device)
    policy.eval()
    return policy


def load_processors(policy, path: str):
    """Return (preprocessor, postprocessor) saved alongside the policy checkpoint."""
    from lerobot.policies.factory import make_pre_post_processors

    try:
        return make_pre_post_processors(policy.config, pretrained_path=path)
    except TypeError:
        return make_pre_post_processors(policy_cfg=policy.config, pretrained_path=path)
