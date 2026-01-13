from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download


ENVS = {
    "atari_qbert": ["ppo", "dqn"],
    "atari_double_dunk": ["ppo", "dqn"],
    "atari_phoenix": ["ppo", "dqn"],
    "atari_this_game": ["ppo", "dqn"],
    "atari_battle_zone": ["ppo", "dqn"],
    "box2d_lunar_lander": ["ppo", "dqn"],
    "box2d_continuous_lunar_lander": ["ppo", "sac"],
    "box2d_bipedal_walker": ["ppo", "sac"],
    "cc_acrobot": ["ppo", "dqn"],
    "cc_cartpole": ["ppo", "dqn"],
    "cc_mountain_car": ["ppo", "dqn"],
    "cc_continuous_mountain_car": ["ppo", "sac"],
    "cc_pendulum": ["ppo", "sac"],
    "minigrid_door_key": ["ppo", "dqn"],
    "minigrid_empty_random": ["ppo", "dqn"],
    "minigrid_four_rooms": ["ppo", "dqn"],
    "minigrid_unlock": ["ppo", "dqn"],
    "brax_ant": ["ppo", "sac"],
    "brax_halfcheetah": ["ppo", "sac"],
    "brax_hopper": ["ppo", "sac"],
    "brax_humanoid": ["ppo", "sac"],
}

atari_games = ["battle_zone", "double_dunk", "phoenix", "qbert", "this_game"]


def bootstrap_95ci(series, n_iterations=1000):
    data = series.values
    sample_size = len(data)
    bootstrap_indices = np.random.randint(
        0, sample_size, size=(n_iterations, sample_size)
    )
    bootstrap_samples = data[bootstrap_indices]
    means = bootstrap_samples.mean(axis=1)
    ci_lower, ci_upper = np.percentile(means, [2.5, 97.5])
    return ci_lower, ci_upper


def get_arlbench_last_performance(df, hp_keys):
    last_df = df
    opt_dfs = []
    for optimizer in df["optimizer"].unique():
        opt_df = last_df[last_df["optimizer"] == optimizer]
        last_opt_df = opt_df[opt_df["training_steps"] == opt_df["training_steps"].max()]
        opt_dfs.append(last_opt_df)
    last_df = pd.concat(opt_dfs)

    def summarize_group(g):
        perf = g["performance"]
        mean_perf = perf.mean()
        ci_lower, ci_upper = bootstrap_95ci(perf)
        return pd.Series(
            {
                "last_performance": mean_perf,
                "last_performance_cil": ci_lower,
                "last_performance_ciu": ci_upper,
                "config_id": g["config_id"].iloc[0],
            }
        )

    result = last_df.groupby(hp_keys + ["optimizer"]).apply(summarize_group).reset_index()
    return result


def get_arlbench_auc_performance(df, hp_keys):
    auc_df = df.groupby(hp_keys + ["seed", "optimizer"], as_index=False).agg(
        {
            "performance": "mean",
            "config_id": "first",
        }
    )

    def summarize_group(g):
        perf = g["performance"]
        mean_perf = perf.mean()
        ci_lower, ci_upper = bootstrap_95ci(perf)
        return pd.Series(
            {
                "max_performance": mean_perf,
                "max_performance_cil": ci_lower,
                "max_performance_ciu": ci_upper,
                "config_id": g["config_id"].iloc[0],
            }
        )

    result = auc_df.groupby(hp_keys+ ["optimizer"]).apply(summarize_group).reset_index()
    return result


def get_arlbench_max_performance(df, hp_keys):
    max_df = df.groupby(hp_keys + ["seed", "optimizer"], as_index=False).agg(
        {
            "performance": "max",
            "config_id": "first",
        }
    )

    def summarize_group(g):
        perf = g["performance"]
        mean_perf = perf.mean()
        ci_lower, ci_upper = bootstrap_95ci(perf)
        return pd.Series(
            {
                "auc_performance": mean_perf,
                "auc_performance_cil": ci_lower,
                "auc_performance_ciu": ci_upper,
                "config_id": g["config_id"].iloc[0],
            }
        )

    result = max_df.groupby(hp_keys+ ["optimizer"]).apply(summarize_group).reset_index()
    return result


def arlbench_preprocessing(df, env_name, num_seeds):
    hp_keys = [key for key in df.columns if "hp_config" in key]

    sel_seeds = list(range(num_seeds))
    df = df[df["seed"].isin(sel_seeds)]

    df.loc[:, hp_keys] = df[hp_keys].fillna("NaN")
    df.loc[:, "performance"] = df["performance"].fillna(min(df["performance"]))
    # clip lower performance due to numerical instabilities
    if "brax" in env_name:
        df["performance"] = df["performance"].clip(lower=-2000)
    elif "box2d" in env_name:
        df["performance"] = df["performance"].clip(lower=-200)

    hp_df = df
    last_df = get_arlbench_last_performance(df, hp_keys).drop(columns=hp_keys)
    max_df = get_arlbench_max_performance(df, hp_keys).drop(columns=hp_keys)
    auc_df = get_arlbench_auc_performance(df, hp_keys).drop(columns=hp_keys)

    df = hp_df.merge(last_df, on=["config_id", "optimizer"], how="left")
    df = df.merge(max_df, on=["config_id", "optimizer"], how="left")
    df = df.merge(auc_df, on=["config_id", "optimizer"], how="left")
    df = df[["config_id"] + [col for col in df.columns if col != "config_id"]]
    df = df.drop(columns=["training_steps", "optimization_step"])
    df.drop_duplicates(subset=hp_keys + ["config_id", "optimizer", "seed"], keep="first", inplace=True)

    for perf_key in ["last_performance", "max_performance", "auc_performance"]:
        df[f"{perf_key}_unnormalized"] = df[perf_key].copy()
        max_perf = max(df[perf_key])
        min_perf = min(df[perf_key])
        df[perf_key] = df[perf_key].map(
            lambda x: (x - min_perf) / (max_perf - min_perf)
        )
        for metric in ["ciu", "cil"]:
            key = f"{perf_key}_{metric}"
            df[key] = df[key].map(lambda x: (x - min_perf) / (max_perf - min_perf))

    return df


def download_arlbench(cfg):
    if "filename" in cfg.keys():
        assert (
            cfg.env_name in ENVS
        ), f"Environment {cfg.env_name} not available. Available environments are: {ENVS.keys()}"
        assert (
            cfg.algorithm in ENVS[cfg.env_name]
        ), f"cfg.algorithm {cfg.algorithm} not available for environment {cfg.env_name}. Available cfg.algorithms for {cfg.env_name} are: {ENVS[cfg.env_name]}"

    if "filename" in cfg.keys() and Path(cfg.filename).exists():
        print(f"Reading local dataset: {cfg.env_name}, {cfg.algorithm}.")
        filename = cfg.filename
    else:
        filename = Path(f"{cfg.dir}/{cfg.data_dir}/{cfg.num_configs}_{cfg.num_seeds}")
        filename = filename / f"{cfg.env_name}_{cfg.algorithm}.csv"
        if not filename.exists():
            print(f"Downloading data from huggingface: {cfg.env_name}, {cfg.algorithm}.")
            hugging_face_landscape_file = f"landscapes/{cfg.env_name}_{cfg.algorithm}.csv"
            data = []
            if not Path(
                f"{cfg.dir}/{cfg.data_dir}/{cfg.env_name}_{cfg.algorithm}.csv"
            ).exists():
                data.append(
                    pd.read_csv(
                        hf_hub_download(
                            repo_id=cfg.repo_id,
                            filename=hugging_face_landscape_file,
                            repo_type="dataset",
                        )
                    )
                )
                data[0].to_csv(
                    Path(f"{cfg.dir}/{cfg.data_dir}/{cfg.env_name}_{cfg.algorithm}.csv"),
                    index=False,
                )
            else:
                data.append(
                    pd.read_csv(
                        Path(f"{cfg.dir}/{cfg.data_dir}/{cfg.env_name}_{cfg.algorithm}.csv")
                    )
                )

            data = pd.concat(data).reset_index()

            data = arlbench_preprocessing(data, cfg.env_name, cfg.num_seeds)
            num_configs = data["config_id"].nunique()
            rng = np.random.default_rng(42)
            config_permutation = rng.permutation(num_configs)
            sel_configs = config_permutation[: cfg.num_configs]
            data = data[data["config_id"].isin(sel_configs)]
            filename.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(filename, index=False)

    if not cfg.landscape_only:
            print("Downloading optimizer data from huggingface.")
            hugging_face_smac_file = f"smac/{cfg.env_name}_{cfg.algorithm}.csv"
            hugging_face_smac_mf_file = f"smac_mf/{cfg.env_name}_{cfg.algorithm}.csv"
            hugging_face_rs_file = f"rs/{cfg.env_name}_{cfg.algorithm}.csv"
            smac_data = pd.read_csv(
                    hf_hub_download(
                        repo_id=cfg.repo_id,
                        filename=hugging_face_smac_file,
                        repo_type="dataset",
                    )
                )
            
            smac_mf_data = pd.read_csv(
                    hf_hub_download(
                        repo_id=cfg.repo_id,
                        filename=hugging_face_smac_mf_file,
                        repo_type="dataset",
                    )
                )
            smac_mf_data["optimizer"] = "smac_mf"
            smac_data["optimizer"] = "smac"
            rs_opt_data = pd.read_csv(
                    hf_hub_download(
                        repo_id=cfg.repo_id,
                        filename=hugging_face_rs_file,
                        repo_type="dataset",
                    )
                )
            rs_opt_data["optimizer"] = "random_search"

            data = pd.concat([smac_data, smac_mf_data, rs_opt_data]).reset_index()
            data = arlbench_preprocessing(data, cfg.env_name, cfg.num_seeds)
            filename = Path(f"{cfg.dir}/{cfg.data_dir}/optimizer_data/{cfg.env_name}_{cfg.algorithm}.csv")
            num_configs = data["config_id"].nunique()
            filename.parent.mkdir(parents=True, exist_ok=True)
            data.to_csv(filename, index=False)
        

    return filename, f"{cfg.dir}/{cfg.data_dir}/{cfg.env_name}_{cfg.algorithm}.csv"

@hydra.main(config_path=".", config_name="download_config", version_base="1.1")
def download_data(cfg):
    if cfg.benchmark == "arlbench":
        return download_arlbench(cfg)
    elif cfg.benchmark == "lunar_lander":
        return download_arlbench(cfg)
    else:
        raise ValueError("Unknown benchmark.")


if __name__ == "__main__":
    download_data()