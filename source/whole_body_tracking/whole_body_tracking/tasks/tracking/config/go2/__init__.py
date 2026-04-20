import gymnasium as gym

from . import agents, flat_env_cfg, trampoline_env_cfg

##
# Register Gym environments.
##


gym.register(
    id="Tracking-Flat-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Go2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-Go2-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Go2FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Go2-Jump-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Go2FlatJumpWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Trampoline-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": trampoline_env_cfg.Go2TrampolineEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2TrampolinePPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Trampoline-Go2-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": trampoline_env_cfg.Go2TrampolineWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2TrampolinePPORunnerCfg",
    },
)
