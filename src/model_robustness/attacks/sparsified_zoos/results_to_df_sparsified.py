import os
from pathlib import Path
import json
import pandas as pd 
from ray import tune, air
import ray

ROOT = Path("")


def return_names_for_path(dataset, setup):
    if dataset=="CIFAR10":
        size = "large"
    else:
        size = "small"
    
    if setup == "seed":
        zoo_p = f"cnn_{size}_{dataset.lower()}_ard"
    elif setup == "hyp-10-f":
        zoo_p = f"cnn_{size}_{dataset.lower()}_fixed_ard"
    else: 
        zoo_p = f"cnn_{size}_{dataset.lower()}_rand_ard"

    return zoo_p


def add_results_to_df(tune_config):

    results_root = os.path.join(ROOT, "/netscratch2/jlautz/model_robustness/src/model_robustness/data/sparsified")

    # Get all model names
    all_models_df = pd.read_csv(os.path.join(results_root, "clean_zoos", f"{tune_config['dataset']}_{tune_config['setup']}_clean_zoo.csv"))
    model_paths = all_models_df["path"].tolist()

    results_root = Path(
            os.path.join(results_root, tune_config["dataset"], tune_config["attack"], tune_config["setup"], f"eps_{tune_config['eps']}", "results")
        )

    # path to model zoo 
    zoo_p = return_names_for_path(tune_config["dataset"], tune_config["setup"])
    parameter_path = os.path.join(ROOT, f"/ds2/model_zoos/zoos_sparsified/distillation/zoos/{tune_config['dataset']}/ARD/{zoo_p}")

    # Instantiate empty dataframe
    df = pd.DataFrame(columns=["name", "dataset", "attack", "eps", "dropout", 
                                "init_type", "nlin", "lr", "momentum", "optimizer", "wd", "seed",
                                "old_loss", "old_acc", "new_loss", "new_acc"])

    # Iterate over all models and save results in dataframe
    for i, path in enumerate(model_paths):

        # Read in config containing the parameters for the i-th model
        model_config_path = os.path.join(parameter_path, path, "params.json")
        config_model = json.load(open(model_config_path, ))

        df.loc[i, "name"] = path
        df.loc[i, "dataset"] = tune_config["dataset"]
        df.loc[i, "attack"] = tune_config["attack"]
        df.loc[i, "setup"] = tune_config["setup"]
        df.loc[i, "eps"] = tune_config["eps"]
        df.loc[i, "dropout"] = config_model["model::dropout"]
        df.loc[i, "init_type"] = config_model["model::init_type"]
        df.loc[i, "nlin"] = config_model["model::nlin"]
        df.loc[i, "lr"] = config_model["optim::lr"]
        df.loc[i, "optimizer"] = config_model["optim::optimizer"]
        df.loc[i, "wd"] = config_model["optim::wd"]
        df.loc[i, "seed"] = config_model["seed"]
        try:
            df.loc[i, "momentum"] = config_model["optim::momentum"]
        except KeyError:
            pass

        # # Old results
        # result_model_path = os.path.join(parameter_path, path, "result.json")
        # c = 0
        # for line in open(result_model_path, "r"):
        #     c += 1
        #     if c == 50:
        #         index = line.index("test_acc")
        #         df.loc[i, "old_acc"] = eval(line[index+11:index+16])
        #         index = line.index("test_loss")
        #         try:
        #             df.loc[i, "old_loss"] = eval(line[index+12:index+17])
        #         except:
        #             pass

        result_model_path = os.path.join(parameter_path, path, "result.json")
        
        for iter, line in enumerate(open(result_model_path, "r")):
            if iter == 25:
                aux_dict = json.loads(line)
        
        df.loc[i, "old_acc"] = aux_dict["test_acc"]
        try:
            df.loc[i, "old_loss"] = aux_dict["test_acc"]
        except:
            pass

        # New results
        results_path = os.path.join(results_root, path, "result.json")
        new_results = json.load(open(results_path, ))

        df.loc[i, "new_loss"] = new_results["test_loss"]
        df.loc[i, "new_acc"] = new_results["test_acc"]

    df.to_csv(os.path.join(results_root, "results_df.csv"), index=False)


def main():
    cpus = 6
    gpus = 1

    cpus_per_trial = 3
    gpu_fraction = ((gpus*100) // (cpus/cpus_per_trial)) / 100
    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpu_fraction}

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus
    )

    assert ray.is_initialized() == True

    # Define search space (all experiment configurations)
    search_space = {
        "dataset": tune.grid_search(["CIFAR10", "MNIST", "SVHN"]),
        "attack": tune.grid_search(["PGD"]),
        "setup": tune.grid_search(["hyp-10-f", "hyp-10-r", "seed"]),
        "eps": tune.grid_search([0.1, 0.2, 0.3, 0.4, 0.5]),
    }

    add_results_w_resources = tune.with_resources(add_results_to_df, resources_per_trial)

    tuner = tune.Tuner(
        add_results_w_resources,
        tune_config=tune.TuneConfig(num_samples=1),
        param_space=search_space
    )

    results = tuner.fit()

    ray.shutdown()


if __name__ == "__main__":
    main()



