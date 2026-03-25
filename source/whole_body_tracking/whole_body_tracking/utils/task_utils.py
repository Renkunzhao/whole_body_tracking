from __future__ import annotations


def env_cfg_requires_motion(env_cfg) -> bool:
    if env_cfg is None:
        return False
    commands = getattr(env_cfg, "commands", None)
    return commands is not None and hasattr(commands, "motion")


def env_requires_motion(env) -> bool:
    if env is None:
        return False
    if hasattr(env, "command_manager"):
        active_terms = getattr(env.command_manager, "active_terms", ())
        return "motion" in active_terms
    return env_cfg_requires_motion(getattr(env, "cfg", None))
