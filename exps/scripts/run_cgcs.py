"""Command-line entry point for the preset CGCS experiment sweep."""

from exps.configs.cgcs import build_cgcs_experiment_config
from exps.env_factory import build_lna_env
from exps.launcher import run_experiment


def main():
    """Run the preset CGCS experiment for every configured seed."""

    config = build_cgcs_experiment_config()
    circuit_type = config["circuit_type"]
    env_name = config.get("env_name", "modular")
    env_kwargs = dict(config["env_kwargs"])
    launch_kwargs = dict(config["launch_kwargs"])

    for seed in config["seeds"]:
        env = build_lna_env(
            circuit_type=circuit_type,
            env_kwargs=env_kwargs,
            env_name=env_name,
        )
        run_experiment(
            env,
            circuit_type=circuit_type,
            sweep_tag=f"seed{seed}",
            seed=seed,
            **launch_kwargs,
        )


if __name__ == "__main__":
    main()
