import gymnasium as gym

from . import go2_hopping_env_cfg
from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Go2-Hopping-Flat",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_hopping_env_cfg.Go2HoppingFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2HoppingPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Hopping-Trampoline",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": go2_hopping_env_cfg.Go2HoppingTrampolineEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2HoppingPPORunnerCfg",
    },
)
