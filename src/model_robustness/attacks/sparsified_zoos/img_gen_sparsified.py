import argparse
import logging
import sys
import os
import random
import json
from pathlib import Path
import numpy as np
import pandas as pd

from ray import tune, air
import ray
from ray.air.integrations.wandb import WandbLoggerCallback

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.dataset import random_split

from advertorch.attacks import LinfPGDAttack, GradientSignAttack

from networks import CNN_ARD, CNN3_ARD



ROOT = Path("")


def parse_args(args):
    parser = argparse.ArgumentParser(description="Generates images for PGD attack")

    return parser.parse_args(args)

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


# Define each experiment
def generate_images(tune_config):
    """
    args should be all the arguments set by the CLI command

    tune_config are the parameters that will be iterated over by tune
    in this case, this should be:

    dataset
    setup
    eps
    """

    config = {}
    config["dataset"] = tune_config["dataset"]
    config["attack"] = tune_config["attack"]
    config["setup"] = tune_config["setup"]
    config["n_models"] = 50
    config["eps"] = tune_config["eps"]
    config["nb_iter"] = 10
    config["eps_iter"] = tune_config["eps"]/10
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # path to model zoo checkpoints
    zoo_p = return_names_for_path(config["dataset"], config["setup"])
    checkpoint_path = os.path.join(ROOT, f"/ds2/model_zoos/zoos_sparsified/distillation/zoos/{config['dataset']}/ARD/{zoo_p}")

    # path to "dataset"
    data_path = os.path.join(checkpoint_path, "dataset.pt")  
    data_root = os.path.join(ROOT, "/netscratch2/jlautz/model_robustness/src/model_robustness/data/sparsified")

    # Defining path on where to store the 50 models used to generate perturbed dataset
    model_list_path = Path(
        os.path.join(data_root, config["dataset"], config["attack"], config["setup"]))

    try:
        model_list_path.mkdir(parents="True", exist_ok=False)
    except FileExistsError:
        pass

    # See if the list already exists, otherwise create it
    try:
        with open(os.path.join(model_list_path, 'model_list.txt'), "r") as items:
            model_paths = items.readlines()

            for i, l in enumerate(model_paths):
                model_paths[i] = l.replace("\n", "")
        print("Model_list already existed")

    except FileNotFoundError:

        all_models_df = pd.read_csv(os.path.join(data_root, "clean_zoos", f"{config['dataset']}_{config['setup']}_clean_zoo.csv"))
        zoo_model_list = all_models_df["path"].tolist()

        # Only take the top 50 models
        random.shuffle(zoo_model_list)
        model_paths = zoo_model_list[:50]

        file = open(os.path.join(model_list_path, 'model_list.txt'), 'w')
        for item in model_paths:
            file.write(item + "\n")
        file.close()

    # Load in the data
    dataset = torch.load(data_path)["testset"]

    # Define subsets of testset used for each of the n_models models
    generator = torch.Generator().manual_seed(0)
    imgs_per_model = len(dataset) / config["n_models"]
    split = [int(imgs_per_model) for i in range(config["n_models"])]
    remainder = len(dataset) - sum(split)
    split[-1] += remainder

    subsets = random_split(dataset, split, generator=generator)

    # Create empty tensors that will contain the perturbed images and labels after iterating over the
    # n_models models
    images = torch.tensor((), device=config["device"])
    labels = torch.tensor((), device=config["device"])

    # Iterate over the n_models models and generate imgs_per_model images for each one
    for i, path in enumerate(model_paths):

        # Read in config containing the paramters for the i-th model
        model_config_path = os.path.join(checkpoint_path, path, "params.json")
        config_model = json.load(open(model_config_path, ))


        # For CIFAR10 use large ConvNet, small ConvNet for everything else
        if config["dataset"] == "CIFAR10":
            # Define model and load in state
            model = CNN3_ARD(
                channels_in=config_model["model::channels_in"],
                nlin=config_model["model::nlin"],
                dropout=config_model["model::dropout"],
                init_type=config_model["model::init_type"]
            )
            try:
                model.load_state_dict(
                    torch.load(os.path.join(checkpoint_path, path, "checkpoint_000025", "checkpoints"))
                )
            except RuntimeError:
                model = CNN3_ARD(
                    channels_in=config_model["model::channels_in"],
                    nlin=config_model["model::nlin"],
                    dropout=0,
                    init_type=config_model["model::init_type"]
                )
                try:
                    model.load_state_dict(
                        torch.load(os.path.join(checkpoint_path, path, "checkpoint_000025", "checkpoints"))
                    )
                except RuntimeError:
                    model = CNN3_ARD(
                        channels_in=config_model["model::channels_in"],
                        nlin=config_model["model::nlin"],
                        dropout=0.5,
                        init_type=config_model["model::init_type"]
                    )
                    model.load_state_dict(
                        torch.load(os.path.join(checkpoint_path, path, "checkpoint_000025", "checkpoints"))
                    )
        
        else:
            # Define model and load in state
            model = CNN_ARD(
                channels_in=config_model["model::channels_in"],
                nlin=config_model["model::nlin"],
                dropout=config_model["model::dropout"],
                init_type=config_model["model::init_type"]
            )
            try:
                model.load_state_dict(
                    torch.load(os.path.join(checkpoint_path, path, "checkpoint_000025", "checkpoints"))
                )
            except RuntimeError:
                model = CNN_ARD(
                    channels_in=config_model["model::channels_in"],
                    nlin=config_model["model::nlin"],
                    dropout=0,
                    init_type=config_model["model::init_type"]
                )
                try:
                    model.load_state_dict(
                        torch.load(os.path.join(checkpoint_path, path, "checkpoint_000025", "checkpoints"))
                    )
                except RuntimeError:
                    model = CNN_ARD(
                        channels_in=config_model["model::channels_in"],
                        nlin=config_model["model::nlin"],
                        dropout=0.5,
                        init_type=config_model["model::init_type"]
                    )
                    model.load_state_dict(
                        torch.load(os.path.join(checkpoint_path, path, "checkpoint_000025", "checkpoints"))
                    )
        model.to(config["device"])

        # Attack
        subset = subsets[i]
        aux_loader = DataLoader(
            dataset=subset,
            batch_size=len(subset),
            shuffle=False
        )
        for cln_data, true_labels in aux_loader:
            break
        cln_data, true_labels = cln_data.to(config["device"]), true_labels.to(config["device"])

        if config["attack"] == "PGD":
            adversary = LinfPGDAttack(
                model,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=config["eps"],
                nb_iter=config["nb_iter"],
                eps_iter=config["eps_iter"],
                rand_init=False,
                clip_min=0.0,
                clip_max=1.0,
                targeted=False
            )
        elif config["attack"] == "FGSM":
            adversary = GradientSignAttack(
                model,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=tune_config["eps"],
                targeted=False
            )
        else:
            raise NotImplementedError("error: attack type unknown")

        # Perturb images
        adv_images = adversary.perturb(cln_data, true_labels)

        # Add perturbed images and labels to dataset
        images = torch.cat((images, adv_images))
        labels = torch.cat((labels, true_labels))

        print(f"Model {i+1} done")

    # Save entire dataset
    perturbed_dataset = TensorDataset(images, labels)

    # Define where the perturbed dataset should be saved
    perturbed_path = Path(
        os.path.join(model_list_path, f"eps_{config['eps']}")
    )
    try:
        perturbed_path.mkdir(parents="True", exist_ok=False)
    except FileExistsError:
        pass
    
    assert len(perturbed_dataset) == len(dataset)

    # Save the perturbed dataset
    torch.save(perturbed_dataset, os.path.join(perturbed_path, "perturbed_dataset.pt"))

    # Save config with parameters
    with open((perturbed_path.joinpath("config.json")), "w") as f:
        json.dump(config, f, default=str, indent=4)


def main():
    # ray init to limit memory and storage
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
        "dataset": tune.grid_search(["MNIST", "SVHN", "CIFAR10"]),
        "attack": tune.grid_search(["PGD", "FGSM"]),
        "setup": tune.grid_search(["hyp-10-r", "hyp-10-f", "seed"]),
        "eps": tune.grid_search([0.1, 0.2, 0.3, 0.4, 0.5]),
    }

    generate_images_w_resources = tune.with_resources(generate_images, resources_per_trial)

    # Tune Experiment
    tuner = tune.Tuner(
        generate_images_w_resources,
        # run_config=air.RunConfig(
        #     callbacks=[
        #         WandbLoggerCallback(project="master_thesis", api_key="7fe80de0b53b0ab265297295a37223f3e9cb1215")
        #     ]),
        tune_config=tune.TuneConfig(num_samples=1),
        param_space=search_space
    )
    results = tuner.fit()

    ray.shutdown()


if __name__ == "__main__":
    main()
