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
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.dataset import random_split

from advertorch.attacks import GradientSignAttack, LinfPGDAttack

from model_robustness.attacks.networks import ConvNetSmall


ROOT = Path("")


def calculate_nb_iter(eps_iter):
    return int(np.ceil(min(4+eps_iter, 1.25*eps_iter)))


def parse_args(args):
    parser = argparse.ArgumentParser(description="Generates images for PGD attack")

    return parser.parse_args(args)


# Define each experiment
def generate_images(tune_config):
    """
    args should be all the arguments set by the CLI command

    tune_config are the parameters that will be iterated over by tune
    in this case, this should be:

    dataset
    setup
    nb_iter
    eps_iter
    """

    config = {}
    config["dataset"] = tune_config["dataset"]
    config["attack"] = "PGD"
    config["setup"] = tune_config["setup"]
    config["n_models"] = 50
    config["eps"] = 1
    config["nb_iter"] = calculate_nb_iter(tune_config["eps_iter"])
    config["eps_iter"] = tune_config["eps_iter"]
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = os.path.join(ROOT, "/netscratch2/dtaskiran/zoos")  # path to model zoo checkpoints
    data_root = os.path.join(ROOT, "/netscratch2/jlautz/model_robustness/src/model_robustness/data")  # path to "dataset"

    # Defining data_path and checkpoint_path depending on configuration
    if config["dataset"] == "MNIST":
        checkpoint_path = os.path.join(checkpoint_path, "MNIST")
        data_root = os.path.join(data_root, "MNIST")
        if config["setup"] == "hyp-10-r":
            checkpoint_path = os.path.join(checkpoint_path, "tune_zoo_mnist_hyperparameter_10_random_seeds")
            data_path = os.path.join(data_root, "dataset.pt")
        elif config["setup"] == "hyp-10-f":
            checkpoint_path = os.path.join(checkpoint_path, "tune_zoo_mnist_hyperparameter_10_fixed_seeds")
            data_path = os.path.join(data_root, "dataset.pt")

    elif config["dataset"] == "CIFAR10":
        checkpoint_path = os.path.join(checkpoint_path, "CIFAR10", "small")
        data_root = os.path.join(data_root, "CIFAR10")
        if config["setup"] == "hyp-10-r":
            checkpoint_path = os.path.join(checkpoint_path, "tune_zoo_cifar10_small_hyperparameter_10_random_seeds")
            data_path = os.path.join(data_root, "dataset.pt")
        elif config["setup"] == "hyp-10-f":
            checkpoint_path = os.path.join(checkpoint_path, "tune_zoo_cifar10_small_hyperparameter_10_fixed_seeds")
            data_path = os.path.join(data_root, "dataset.pt")

    # Defining path on where to store the 50 models used to generate perturbed dataset
    model_list_path = Path(
        os.path.join(data_root, config["attack"], config["setup"]))

    try:
        model_list_path.mkdir(parents="True", exist_ok=False)
    except FileExistsError:
        pass
    print(f"Checkpoint Path: {checkpoint_path}")
    print(f"Data Path: {data_path}")
    print(f"Model List Path: {model_list_path}")

    # See if the list already exists, otherwise create it
    try:
        with open(os.path.join(model_list_path, 'model_list.txt'), "r") as items:
            model_paths = items.readlines()

            for i, l in enumerate(model_paths):
                model_paths[i] = l.replace("\n", "")
        print("Model list already exists.")

    except FileNotFoundError:
        print("Model list does not exist yet, defining it now.")
        # list to store files
        model_paths = []

        # Iterate directory
        for path in os.listdir(checkpoint_path):
            # only append directories
            if not os.path.isfile(os.path.join(checkpoint_path, path)):
                model_paths.append(path)

        # Only take the top 50 models
        random.shuffle(model_paths)
        model_paths = model_paths[:50]

        file = open(os.path.join(model_list_path, 'model_list.txt'), 'w')
        for item in model_paths:
            file.write(item + "\n")
        file.close()

    # Load in the data
    dataset = torch.load(data_path)["testset"]
    assert len(dataset) == 10000
    print("Dataset successfully loaded.")

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
    print(f"Starting iteration over the {config['n_models']} models.")
    for i, path in enumerate(model_paths):

        # Read in config containing the paramters for the i-th model
        model_config_path = os.path.join(checkpoint_path, path, "params.json")
        config_model = json.load(open(model_config_path, ))

        # Define model and load in state
        model = ConvNetSmall(
            channels_in=config_model["model::channels_in"],
            nlin=config_model["model::nlin"],
            dropout=config_model["model::dropout"],
            init_type=config_model["model::init_type"]
        )
        model.load_state_dict(
            torch.load(os.path.join(checkpoint_path, path, "checkpoint_000050", "checkpoints"))
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
                eps=config["eps"] / 255,
                nb_iter=config["nb_iter"],
                eps_iter=config["eps_iter"] / 255,
                rand_init=True,
                clip_min=0.0,
                clip_max=1.0,
                targeted=False
            )

        else:
            raise NotImplementedError("error: attack type unknown")

        # Perturb images
        adv_images = adversary.perturb(cln_data, true_labels)

        # Add perturbed images and labels to dataset
        images = torch.cat((images, adv_images))
        labels = torch.cat((labels, true_labels))

        assert len(images) == (i + 1) * imgs_per_model
        assert len(labels) == (i + 1) * imgs_per_model
        print(f"Model {i+1} done.")
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

    # Save the perturbed dataset
    torch.save(perturbed_dataset, os.path.join(perturbed_path, "perturbed_dataset.pt"))
    print("Perturbed dataset successfully saved.")

    # Save config with parameters
    with open((perturbed_path.joinpath("config.json")), "w") as f:
        json.dump(config, f, default=str, indent=4)


def main():
    tune_config = {
        "dataset": "MNIST",
        "setup": "hyp-10-r",
        "eps_iter": 2
    }

    generate_images(tune_config=tune_config)


if __name__ == "__main__":
    main()
