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


def evaluate_images(tune_config):

    device = "cpu"  # dont think we need more than that

    # Take every model from model zoo
    data_root = os.path.join(ROOT, "/netscratch2/jlautz/model_robustness/src/model_robustness/data")
    old_result_path = os.path.join(ROOT, "/netscratch2/dtaskiran/zoos")


    if tune_config["dataset"] == "MNIST":
        old_result_path = os.path.join(old_result_path, "MNIST")
        if tune_config["setup"] == "hyp-10-r":
            old_result_path = os.path.join(old_result_path, "tune_zoo_mnist_hyperparameter_10_random_seeds")
        elif tune_config["setup"] == "hyp-10-f":
            old_result_path = os.path.join(old_result_path, "tune_zoo_mnist_hyperparameter_10_fixed_seeds")
        elif tune_config["setup"] == "seed":
            old_result_path = os.path.join(old_result_path, "tune_zoo_mnist_uniform")

    elif tune_config["dataset"] == "CIFAR10":
        old_result_path = os.path.join(old_result_path, "CIFAR10", "large")
        if tune_config["setup"] == "hyp-10-r":
            old_result_path = os.path.join(old_result_path, "tune_zoo_cifar10_large_hyperparameter_10_random_seeds")
        elif tune_config["setup"] == "hyp-10-f":
            old_result_path = os.path.join(old_result_path, "tune_zoo_cifar10_large_hyperparameter_10_fixed_seeds")
        elif tune_config["setup"] == "seed":
            old_result_path = os.path.join(old_result_path, "tune_zoo_cifar10_uniform_large")

    elif tune_config["dataset"] == "SVHN":
        old_result_path = os.path.join(old_result_path, "SVHN")
        if tune_config["setup"] == "hyp-10-r":
            old_result_path = os.path.join(old_result_path, "tune_zoo_svhn_hyperparameter_10_random_seeds")
        elif tune_config["setup"] == "hyp-10-f":
            old_result_path = os.path.join(old_result_path, "tune_zoo_svhn_hyperparameter_10_fixed_seeds")
        elif tune_config["setup"] == "seed":
            old_result_path = os.path.join(old_result_path, "tune_zoo_svhn_uniform")

    if tune_config["dataset"] == "CIFAR10":
        data_path = os.path.join(data_root, tune_config["dataset"], "large", tune_config["attack"],
            tune_config["setup"], f"eps_{tune_config['eps']}", "perturbed_dataset.pt")
    else:
        data_path = os.path.join(data_root, tune_config["dataset"], tune_config["attack"],
            tune_config["setup"], f"eps_{tune_config['eps']}", "perturbed_dataset.pt")
    

    # Get all models to iterate over
    model_paths = []

    for path in os.listdir(old_result_path):
        # only append directories
        if not os.path.isfile(os.path.join(old_result_path, path)):
            model_paths.append(path)
            
    dataset = torch.load(data_path)
    
    # Iterate over all models in the zoo and evaluate them on the perturbed dataset
    for i, path in enumerate(model_paths):

        # Read in config containing the paramters for the i-th model
        model_config_path = os.path.join(old_result_path, path, "params.json")
        config_model = json.load(open(model_config_path, ))

        # For CIFAR10 use large ConvNet, small ConvNet for everything else
        if tune_config["dataset"] == "CIFAR10":
            # Define model and load in state
            model = ConvNetLarge(
                channels_in=config_model["model::channels_in"],
                nlin=config_model["model::nlin"],
                dropout=config_model["model::dropout"],
                init_type=config_model["model::init_type"]
            )
            try:
                model.load_state_dict(
                    torch.load(os.path.join(old_result_path, path, "checkpoint_000050", "checkpoints"))
                )
            except RuntimeError:
                model = ConvNetLarge(
                    channels_in=config_model["model::channels_in"],
                    nlin=config_model["model::nlin"],
                    dropout=0,
                    init_type=config_model["model::init_type"]
                )
                try:
                    model.load_state_dict(
                        torch.load(os.path.join(old_result_path, path, "checkpoint_000050", "checkpoints"))
                    )
                except RuntimeError:
                    model = ConvNetLarge(
                        channels_in=config_model["model::channels_in"],
                        nlin=config_model["model::nlin"],
                        dropout=0.5,
                        init_type=config_model["model::init_type"]
                    )
                    model.load_state_dict(
                        torch.load(os.path.join(old_result_path, path, "checkpoint_000050", "checkpoints"))
                    )
        
        else:
            # Define model and load in state
            model = ConvNetSmall(
                channels_in=config_model["model::channels_in"],
                nlin=config_model["model::nlin"],
                dropout=config_model["model::dropout"],
                init_type=config_model["model::init_type"]
            )
            try:
                model.load_state_dict(
                    torch.load(os.path.join(old_result_path, path, "checkpoint_000050", "checkpoints"))
                )
            except RuntimeError:
                model = ConvNetSmall(
                    channels_in=config_model["model::channels_in"],
                    nlin=config_model["model::nlin"],
                    dropout=0,
                    init_type=config_model["model::init_type"]
                )
                model.load_state_dict(
                    torch.load(os.path.join(old_result_path, path, "checkpoint_000050", "checkpoints"))
                )
        model.to(device)

        # Define dataloader
        loader = DataLoader(
            dataset=dataset,
            batch_size=config_model["training::batchsize"],
            shuffle=False
        )

        # Define criterion and optimizer
        if config_model["optim::optimizer"] == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=config_model["optim::lr"], 
                momentum=config_model["optim::momentum"], weight_decay=config_model["optim::wd"])

        elif config_model["optim::optimizer"] == "adam":
            optimizer = optim.Adam(model.parameters(), lr=config_model["optim::lr"], 
                weight_decay=config_model["optim::wd"])

        criterion = nn.CrossEntropyLoss()

        # Evaluate        
        loss_avg, acc_avg, num_exp = 0, 0, 0

        for j, data in enumerate(loader):
            
            model.eval()

            imgs, labels = data
            labels = labels.type(torch.LongTensor)
            imgs, labels = imgs.to(device), labels.to(device)
            n_b = labels.shape[0]

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            acc = np.sum(np.equal(np.argmax(outputs.cpu().data.numpy(), axis=-1),
                labels.cpu().data.numpy()))

            loss_avg += loss.item()
            acc_avg += acc
            num_exp += n_b

        loss_avg /= num_exp
        acc_avg /= num_exp

        # Save results
        if tune_config["dataset"] == "CIFAR10":
            save_results_path = Path(
                os.path.join(data_root, tune_config["dataset"], "large", tune_config["attack"], 
                tune_config["setup"], f"eps_{tune_config['eps']}", "results", path)
            )
        else:
            save_results_path = Path(
                os.path.join(data_root, tune_config["dataset"], tune_config["attack"], 
                tune_config["setup"], f"eps_{tune_config['eps']}", "results", path)
            )
        
        try:
            save_results_path.mkdir(parents="True", exist_ok=False)
        except FileExistsError:
            pass

        results = {}
        results["test_loss"] = loss_avg
        results["test_acc"] = acc_avg

        with open((os.path.join(save_results_path, "result.json")), 'w') as f:
            json.dump(results, f, default=str, indent=4)
        
        print(f"Model {i}/{len(model_paths)} done.")


def main():
    cpus = 10
    gpus = 0

    cpus_per_trial = 5
    gpu_fraction = ((gpus*100) // (cpus/cpus_per_trial)) / 100
    resources_per_trial = {"cpu": cpus_per_trial, "gpu": gpu_fraction}

    ray.init(
        num_cpus=cpus,
        num_gpus=gpus
    )

    assert ray.is_initialized() == True

    # Define search space (all experiment configurations)
    search_space = {
        "dataset": tune.grid_search(["MNIST", "CIFAR10", "SVHN"]),
        "attack": tune.grid_search(["PGD", "FGSM"]),
        "setup": tune.grid_search(["hyp-10-r", "hyp-10-f", "seed"]),
        "eps": tune.grid_search([0.1, 0.2, 0.3, 0.4, 0.5]),
    }

    evaluate_images_w_resources = tune.with_resources(evaluate_images, resources_per_trial)

    # Tune Experiment
    tuner = tune.Tuner(
        evaluate_images_w_resources,
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
