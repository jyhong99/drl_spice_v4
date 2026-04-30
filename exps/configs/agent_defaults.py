"""Default agent hyperparameter presets for experiment launchers."""

def build_agent_launch_defaults_10k():
    """Return algorithm launch defaults tuned for 10k-step experiments.

    Returns
    -------
    dict
        Nested mapping of per-agent keyword-argument presets.
    """

    return {
        "ppo_kwargs": {
            "actor_size": (128, 128),
            "critic_size": (128, 128),
            "buffer_size": 1024,
            "update_after": 512,
            "train_iters": 10,
            "batch_size": 64,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "gamma": 0.99,
            "lmda": 0.95,
            "clip_range": 0.2,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "adv_norm": True,
            "max_grad_norm": 0.5,
        },
        "ddpg_kwargs": {
            "actor_size": (256, 256),
            "critic_size": (256, 256),
            "buffer_size": 10000,
            "batch_size": 256,
            "update_after": 1000,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "action_noise_std": 0.1,
            "noise_type": "normal",
            "max_grad_norm": 5.0,
        },
        "td3_kwargs": {
            "actor_size": (256, 256),
            "critic_size": (256, 256),
            "buffer_size": 10000,
            "batch_size": 256,
            "update_after": 1000,
            "update_freq": 2,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "action_noise_std": 0.1,
            "target_noise_std": 0.2,
            "noise_clip": 0.5,
            "noise_type": "normal",
            "max_grad_norm": 5.0,
        },
        "sac_kwargs": {
            "actor_size": (256, 256),
            "critic_size": (256, 256),
            "buffer_size": 10000,
            "batch_size": 256,
            "update_after": 1000,
            "update_freq": 1,
            "actor_lr": 3e-4,
            "critic_lr": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "adaptive_alpha_mode": True,
            "ent_lr": 3e-4,
            "max_grad_norm": 5.0,
        },
    }
