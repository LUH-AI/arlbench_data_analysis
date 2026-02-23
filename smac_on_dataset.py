import pandas as pd
import smac
from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade as HPOFacade
from smac import Scenario
from ConfigSpace import ConfigurationSpace, Float, Categorical
from arlbench.core.algorithms import DQN, PPO, SAC
import numpy as np
from pathlib import Path

algorithms = {"dqn": DQN, "ppo": PPO, "sac": SAC}
environments = ["atari_qbert", 'atari_double_dunk', 'atari_phoenix', 'atari_this_game', 'atari_battle_zone', 
        'box2d_lunar_lander', 'box2d_bipedal_walker', 'cc_acrobot', 'cc_cartpole', 'cc_mountain_car', 'cc_pendulum', 'cc_continuous_mountain_car',
        'minigrid_door_key', 'minigrid_empty_random', 'minigrid_four_rooms', 'minigrid_unlock', 'brax_ant', 'brax_halfcheetah', 'brax_hopper', 'brax_humanoid']

def get_configspace(algo, dataset, seed):
    default_configspace = algorithms[algo].get_hpo_search_space().get_hyperparameter_names()
    configspace = ConfigurationSpace(seed=seed)
    for c in default_configspace:
        if c in ["buffer_prio_sampling", "use_target_network", "alpha_auto", "normalize_advantage"]:
            hp = Categorical(c, [True, False])
            configspace.add_hyperparameter(hp)
        else:
            key = [a for a in dataset.keys() if c in a]
            if len(key) > 0:
                key = key[0]
                hp_min = dataset[key].min()
                hp_max = dataset[key].max()
                bounds = (min(0, hp_min-0.1*hp_min), hp_max+0.1*hp_max)
                hp = Float(c, bounds=bounds)
                configspace.add_hyperparameter(hp)
    return configspace

algos = []
envs = []
best_seen = []
mean_seen = []
budgets = []
seeds = []
methods = []   

for seed in [0,1,2,3,4]:
    for budget in [16, 32, 64, 128]:
        for algo in ["ppo", "dqn", "sac"]:
            for env in environments:
                print(f"{algo}: {env}-{seed} {budget}")
                if not Path(f"arlbench_data/256_10/{env}_{algo}.csv").exists():
                    continue
                dataset = pd.read_csv(f"arlbench_data/256_10/{env}_{algo}.csv")
                configspace = get_configspace(algo, dataset, seed)
                # normalize all hyperparameters
                # but save min and max for each hyperparameter to be able to normalize the hyperparameters in the config
                hyperparameter_columns = [col for col in dataset.columns if col.startswith("hp_config.")]
                hp_limits = {}
                for col in hyperparameter_columns:
                    hp_limits[col] = (dataset[col].min(), dataset[col].max())
                    if isinstance(dataset[col].min(), np.bool_) or isinstance(dataset[col], bool):
                        continue 
                    dataset[col] = (dataset[col] - dataset[col].min()) / (dataset[col].max() - dataset[col].min())

                def get_closest_point(config, seed):
                    # normalize the hyperparameters in the config
                    normalized_config = {}
                    for col in hyperparameter_columns:
                        min_val, max_val = hp_limits[col]
                        if isinstance(dataset[col].min(), np.bool_) or isinstance(dataset[col], bool):
                            continue 
                        normalized_config[col] = (config[col.split(".")[1]] - min_val) / (max_val - min_val)
                    # calculate the distance to all points in the dataset
                    distances = []
                    for _, row in dataset.iterrows():
                        distance = 0
                        for col in hyperparameter_columns:
                            if col in normalized_config.keys() and not np.isnan(normalized_config[col]) and not np.isnan(row[col]):
                                distance += (normalized_config[col] - row[col])**2
                        distances.append(distance)
                    # return the closest point
                    closest_index = distances.index(min(distances))
                    return -dataset.iloc[closest_index]["last_performance"]


                # Scenario object specifying the optimization "environment"
                scenario = Scenario(configspace, deterministic=True, n_trials=budget)

                # Now we use SMAC to find the best hyperparameters
                smac = HPOFacade(
                        scenario,
                        get_closest_point,  # We pass the target function here
                        overwrite=True,  # Overrides any previous results that are found that are inconsistent with the meta-data
                    )

                incumbent = smac.optimize()

                # Let's calculate the cost of the incumbent
                incumbent_cost = smac.validate(incumbent)

                total_cost = 0
                for key in smac._runhistory._data.keys():
                    total_cost += smac._runhistory._data[key].cost
                total_cost = total_cost / len(list(smac._runhistory._data.keys()))

                best_seen.append(incumbent_cost)
                mean_seen.append(total_cost)
                algos.append(algo)
                envs.append(env)
                budgets.append(budget)
                seeds.append(seed)
                methods.append("smac")

                best_seen.append(dataset.sample(n=budget, random_state=seed)["last_performance"].max())
                mean_seen.append(dataset.sample(n=budget, random_state=seed)["last_performance"].mean())
                algos.append(algo)
                envs.append(env)
                budgets.append(budget)
                seeds.append(seed)
                methods.append("rs")

df = pd.DataFrame({"algorithm": algos, "env_name": envs, "budget": budgets, "incumbent": best_seen, "mean_performance": mean_seen, "method": methods})
df.to_csv("smac_on_dataset.csv")