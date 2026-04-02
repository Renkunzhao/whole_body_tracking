from __future__ import annotations


def env_cfg_requires_motion(env_cfg) -> bool:
    if env_cfg is None:
        return False
    commands = getattr(env_cfg, "commands", None)
    return commands is not None and hasattr(commands, "motion")
