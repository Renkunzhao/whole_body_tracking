from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class Go2HoppingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 100
    experiment_name = "go2_hopping"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go2HoppingRecurrentPPORunnerCfg(Go2HoppingPPORunnerCfg):
    num_steps_per_env = 64
    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )


@configclass
class Go2HoppingDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 30000
    save_interval = 100
    experiment_name = "go2_hopping"
    empirical_normalization = True
    obs_groups = {"policy": ["policy"], "teacher": ["teacher"]}
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        student_obs_normalization=True,
        teacher_obs_normalization=True,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=5,
        learning_rate=1.0e-3,
        gradient_length=16,
        max_grad_norm=1.0,
        loss_type="mse",
        optimizer="adam",
    )


@configclass
class Go2HoppingRecurrentDistillationRunnerCfg(Go2HoppingDistillationRunnerCfg):
    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        init_noise_std=0.1,
        student_obs_normalization=True,
        teacher_obs_normalization=True,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        teacher_recurrent=False,
    )
