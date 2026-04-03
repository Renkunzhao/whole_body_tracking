from __future__ import annotations


def env_cfg_requires_motion(env_cfg) -> bool:
    if env_cfg is None:
        return False
    commands = getattr(env_cfg, "commands", None)
    return commands is not None and hasattr(commands, "motion")


def apply_play_overrides(env_cfg):
    if env_cfg is None:
        return env_cfg

    override_fn = getattr(env_cfg, "apply_play_overrides", None)
    if not callable(override_fn):
        return env_cfg

    updated_cfg = override_fn()
    return env_cfg if updated_cfg is None else updated_cfg
