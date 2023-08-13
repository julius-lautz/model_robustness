import os
from pathlib import Path
import json
import pandas as pd 
from ray import tune, air
import ray

ROOT = Path("")


def add_results_to_df(tune_config):

    results_root = os.path.join(ROOT, "/netscratch2/jlautz/model_robustness/src/model_robustness/data/resnet")
    results_root = Path(
            os.path.join(results_root, tune_config["dataset"], tune_config["attack"], f"eps_{tune_config['eps']}", "results")
        )

    # path to model zoo parameters
    if tune_config["dataset"] == "CIFAR10":
        parameter_path = os.path.join(
            ROOT, f"/ds2/model_zoos/zoos_resnet/zoos/CIFAR10/resnet19/kaiming_uniform/tune_zoo_cifar10_resnet18_kaiming_uniform")  
    else:
        parameter_path = os.path.join(
            ROOT, f"/ds2/model_zoos/zoos_resnet/zoos/{tune_config['dataset']}/resnet18/kaiming_uniform/tune_zoo_{tune_config['dataset'].lower()}_resnet18_kaiming_uniform")  

    # Get all model names
    model_paths = []
    for path in os.listdir(results_root):
        if not os.path.isfile(os.path.join(results_root, path)):
            model_paths.append(path)

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
        "dataset": tune.grid_search(["CIFAR10", "CIFAR100", "TinyImageNet"]),
        "attack": tune.grid_search(["PGD", "FGSM"]),
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



