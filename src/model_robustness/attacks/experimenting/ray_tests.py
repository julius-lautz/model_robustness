# Imports
import argparse
import logging
import sys
import os
import random
import json
from pathlib import Path
import numpy as np

from ray import tune, air
import ray
from ray.air.integrations.wandb import WandbLoggerCallback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.dataset import random_split

from model_robustness.attacks.networks import ConvNetSmall, ConvNetLarge

ROOT = Path("")

_logger = logging.getLogger(__name__)


def setup_logging(logLevel):
    logformat= "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=logLevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args(args):
    parser = argparse.ArgumentParser(description="testing")
    parser.add_argument("--zoo", type=str, default="SVHN")
    parser.add_argument("--attack", type=str, default="FGSM")
    parser.add_argument("--setup", type=str, default="fixed")

    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )

    return parser.parse_args(args)


def load_zoo(zoo_name, setup):
    
    ROOT = Path("")

    if setup == "fixed":
        s = "fix"
    elif setup == "random":
        s = "rand"

    checkpoint_path = ROOT.joinpath(f'/netscratch2/dtaskiran/zoos/{zoo_name}/tune_zoo_{zoo_name.lower()}_hyperparameter_10_{setup}_seeds')
    data_root = ROOT.joinpath(f"/netscratch2/jlautz/model_robustness/src/model_robustness/data/{zoo_name}")
    data_path = checkpoint_path.joinpath("dataset.pt")
    zoo_path = ROOT.joinpath(f"/netscratch2/dtaskiran/zoos/{zoo_name}/analysis_data_hyp_{s}.pt")

    dataset = torch.load(data_path)["testset"]
    zoo = torch.load(zoo_path)

    return dataset, zoo, checkpoint_path


def get_best_model_path(zoo):

    # Get only the results from the 50th epoch for each model
    a = 0
    index_list = [50]
    path_list = []
    for i in range(len(zoo["paths"])):
        if i == 0:
            aux = zoo["paths"][i]
            path_list.append(zoo["paths"][i])

        if zoo["paths"][i] == aux:
            pass
        else:
            a += 1
            index_list.append(i+50)
            aux = zoo["paths"][i]
            path_list.append(aux)
    
    for i in range(len(path_list)):
        path_list[i] = path_list[i].__str__().split("/")[-1]

    # Get all accuracies
    acc_list = []
    for index in index_list:
        acc_list.append(zoo["acc"][index])

    # Get the index of max element
    max_index = acc_list.index(max(acc_list))

    # Get the corresponding model name
    best_model_path = path_list[max_index]

    return path_list, max_index, best_model_path


def transferability_experiments(tune_config):
    print(tune_config["best_model_path"])
    print(tune_config["checkpoint_path"])
    print(tune_config["experiment"])


def main(args):

    args=parse_args(args)
    setup_logging(args.loglevel)

    config = {}
    config["dataset"] = args.zoo
    config["attack"] = args.attack
    config["setup"] = args.setup
    config["eps"] = 0.1
    config["nb_iter"] = 10
    config["eps_iter"] = config["eps"]/10
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Get dataset + zoo
    _logger.info("Getting zoo")
    dataset, zoo, checkpoint_path = load_zoo(config["dataset"], config["setup"])

    # Get the best performing model from zoo
    _logger.info("Getting best model from zoo")
    path_list, max_index, best_model_path = get_best_model_path(zoo)

    cpus = 8
    gpus = 0

    cpus_per_trial = 8
    gpu_fraction = ((gpus*100) // (cpus/cpus_per_trial)) / 100
    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpu_fraction}

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus
    )

    assert ray.is_initialized() == True

    search_space = {
        "best_model_path": best_model_path,
        "checkpoint_path": checkpoint_path,
        "experiment": tune.grid_search(["normal", "white_box", "black_box"])
    }

    transferability_experiments_w_resources = tune.with_resources(transferability_experiments, resources_per_trial)

    tuner = tune.Tuner(
        transferability_experiments_w_resources,
        tune_config=tune.TuneConfig(num_samples=1),
        param_space=search_space
    )
    results = tuner.fit()

    ray.shutdown()


def run():
    main(sys.argv[1:])


if __name__=="__main__":
    run()