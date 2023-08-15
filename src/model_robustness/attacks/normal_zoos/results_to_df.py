import os
from pathlib import Path
import json
import pandas as pd 
from ray import tune, air
import ray

ROOT = Path("")


def add_results_to_df(tune_config):

    results_root = os.path.join(ROOT, "/netscratch2/jlautz/model_robustness/src/model_robustness/data")
    parameter_path = os.path.join(ROOT, "/netscratch2/dtaskiran/zoos") 

    if tune_config["dataset"] == "MNIST":
        parameter_path = os.path.join(parameter_path, "MNIST")
        if tune_config["setup"] == "hyp-10-r":
            parameter_path = os.path.join(parameter_path, "tune_zoo_mnist_hyperparameter_10_random_seeds")
        elif tune_config["setup"] == "hyp-10-f":
            parameter_path = os.path.join(parameter_path, "tune_zoo_mnist_hyperparameter_10_fixed_seeds")
        elif tune_config["setup"] == "seed":
            parameter_path = os.path.join(parameter_path, "tune_zoo_mnist_uniform")

    elif tune_config["dataset"] == "CIFAR10":
        parameter_path = os.path.join(parameter_path, "CIFAR10", "large")
        if tune_config["setup"] == "hyp-10-r":
            parameter_path = os.path.join(parameter_path, "tune_zoo_cifar10_large_hyperparameter_10_random_seeds")
        elif tune_config["setup"] == "hyp-10-f":
            parameter_path = os.path.join(parameter_path, "tune_zoo_cifar10_large_hyperparameter_10_fixed_seeds")
        elif tune_config["setup"] == "seed":
            parameter_path = os.path.join(parameter_path, "tune_zoo_cifar10_uniform_large")

    elif tune_config["dataset"] == "SVHN":
        parameter_path = os.path.join(parameter_path, "SVHN")
        if tune_config["setup"] == "hyp-10-r":
            parameter_path = os.path.join(parameter_path, "tune_zoo_svhn_hyperparameter_10_random_seeds")
        elif tune_config["setup"] == "hyp-10-f":
            parameter_path = os.path.join(parameter_path, "tune_zoo_svhn_hyperparameter_10_fixed_seeds")
        elif tune_config["setup"] == "seed":
            parameter_path = os.path.join(parameter_path, "tune_zoo_svhn_uniform")
    
    if tune_config["dataset"] == "CIFAR10":
        results_root = os.path.join(results_root, tune_config["dataset"], "large", tune_config["attack"], tune_config["setup"],
            f"eps_{tune_config['eps']}", "results")
    else:
        results_root = os.path.join(results_root, tune_config["dataset"], tune_config["attack"], tune_config["setup"],
            f"eps_{tune_config['eps']}", "results")

    # Get all model names
    model_paths = []
    for path in os.listdir(results_root):
        if not os.path.isfile(os.path.join(results_root, path)):
            model_paths.append(path)

    # Instantiate empty dataframe
    df = pd.DataFrame(columns=["name", "dataset", "attack", "setup", "eps", "dropout", 
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

        # Old results
        result_model_path = os.path.join(parameter_path, path, "result.json")
        c = 0
        for line in open(result_model_path, "r"):
            c += 1
            if c == 50:
                index = line.index("test_acc")
                df.loc[i, "old_acc"] = eval(line[index+11:index+16])
                index = line.index("test_loss")
                try:
                    df.loc[i, "old_loss"] = eval(line[index+12:index+17])
                except:
                    pass

        # New results
        results_path = os.path.join(results_root, path, "result.json")
        new_results = json.load(open(results_path, ))

        df.loc[i, "new_loss"] = new_results["test_loss"]
        df.loc[i, "new_acc"] = new_results["test_acc"]

    df.to_csv(os.path.join(results_root, "results_df.csv"), index=False)


def main():
    ray.init()

    assert ray.is_initialized() == True

    search_space = {
        "dataset": tune.grid_search(["SVHN"]),
        "attack": tune.grid_search(["PGD", "FGSM"]),
        "setup": tune.grid_search(["hyp-10-f", "hyp-10-r", "seed"]),
        "eps": tune.grid_search([0.1, 0.2, 0.3, 0.4, 0.5]),
    }

    tuner = tune.Tuner(
        add_results_to_df,
        tune_config=tune.TuneConfig(num_samples=1),
        param_space=search_space
    )

    results = tuner.fit()

    ray.shutdown()


if __name__ == "__main__":
    main()



