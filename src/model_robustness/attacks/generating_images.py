import argparse
import logging
import sys
import random
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.dataset import random_split

from advertorch.attacks import GradientSignAttack, LinfPGDAttack

from model_robustness.attacks.networks import ConvNetSmall

__author__ = "julius.lautz"

_logger = logging.getLogger(__name__)

ROOT = Path("")


# ---- CLI ----

def setup_logging(logLevel):
    """Setup basic logging

    Args:
        logLevel (int): minimum loglevel for emitting messages
    """
    logformat= "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=logLevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args(args):
    parser = argparse.ArgumentParser(description="Generating perturbed images for model zoos")
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "CIFAR-10"], help="which dataset to use")
    parser.add_argument("--attack", type=str, default="PGD", choices=["PGD", "FGSM"], help="which attack to use")
    parser.add_argument("--setup", type=str, default="hyp-fix", choices=["hyp_fix", "hyp-rand", "seed"], help="zoo setup")
    parser.add_argument("--n_models", type=int, default=50, help="How many models to use for image perturbation")
    parser.add_argument("--eps", type=float, default=2.0, help="eps in PGD/FGSM attack")
    parser.add_argument("--nb_iter_pgd", type=int, default=3, help="how many iterations to perform for PGD")
    parser.add_argument("--eps_iter_pgd", type=float, default=1.0, help="alpha in PGD attack")
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


def main(args):

    args = parse_args(args)
    setup_logging(args.logLevel)
    _logger.info("Starting to generate perturbed images")

    config = {}
    config["dataset"] = args.dataset
    config["attack"] = args.attack
    config["setup"] = args.setup
    config["n_models"] = args.n_models
    config["eps"] = args.eps_pgd
    config["nb_iter_pgd"] = args.nb_iter_pgd
    config["eps_iter_pgd"] = args.eps_iter_pgd
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = os.path.join(ROOT, "")
    data_path = os.path.join(ROOT, "")

    model_paths = [x[0] for x in os.walk(checkpoint_path)]
    model_paths = model_paths[1:]
    model_paths = model_paths[::52][1:]  # only get the 50th checkpoint for each model

    # Load in the data
    dataset = torch.load(os.path.join(data_path, "dataset.pt"))["testset"]

    # Define subsets of testset used for each of the n_models models
    generator = torch.Generator().manual_seed(0)
    imgs_per_model = len(dataset)/config["n_models"]
    split = [int(imgs_per_model) for i in range(config["n_models"])]

    subsets = random_split(dataset, split, generator=generator)

    images = torch.tensor((), device=config["device"])
    labels = torch.tensor((), device=config["device"])

    # Iterate over the n_models models and generate imgs_per_model images for each one
    for i, path in enumerate(model_paths):


        model_config_path = path.split("\\")
        model_config_path = "\\".join(model_config_path[:-1])
        model_config_path = os.path.join(model_config_path, "params.json")
        config_model = json.load(open(model_config_path,))

        # Define model and load in state
        model = ConvNetSmall(
            channels_in=config_model["model::channels_in"],
            nlin=config_model["model::nlin"],
            dropout=config_model["model::dropout"],
            init_type=config_model["model::init_type"]
        )

        model.load_state_dict(
            torch.load(os.path.join(path, "checkpoints"))
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
                eps=config["eps"]/255,
                nb_iter=config["nb_iter_pgd"],
                eps_iter=config["eps_iter_pgd"]/255,
                rand_init=True,
                clip_min=0.0,
                clip_max=1.0,
                targeted=False
            )

        elif config["attack"] == "FGSM":
            adversary = GradientSignAttack(
                model,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                eps=config["eps_fgsm"],
                targeted=False
            )

        else:
            raise NotImplementedError("error: attack type unknown")

        # Perturb images
        adv_images = adversary.perturb(cln_data, true_labels)

        # Add perturbed images and labels to dataset
        images = torch.cat((images, adv_images))
        labels = torch.cat((labels, true_labels))

        assert len(images) == (i+1) * imgs_per_model
        assert len(labels) == (i+1) * imgs_per_model

        _logger.info(f"Model {i+1}/{config['n_models']} done")

    # Save entire dataset
    perturbed_dataset = TensorDataset(images, labels)

    if args.attack == "PGD":
        perturbed_path = Path(os.path.join(data_path, config["attack"], config["setup"], f"eps_{config['eps']}"))
    elif args.attack == "FGSM":
        perturbed_path = Path(os.path.join(data_path, config["attack"], f"eps_{config['eps']}"))
    try:
        perturbed_path.mkdir(parents="True", exist_ok=False)
    except FileExistsError:
        pass

    torch.save(dataset, os.path.join(data_path, "perturbed_dataset.pt"))

    with open((perturbed_path.joinpath("config.json")), "w") as f:
        json.dump(config, f, default=str, indent=4)
