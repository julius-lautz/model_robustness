# Imports
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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.dataset import random_split

from networks import CNN_ARD, CNN3_ARD

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


def evaluate_images(tune_config):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # path to model zoo checkpoints
    zoo_p = return_names_for_path(tune_config["dataset"], tune_config["setup"])
    old_result_path = os.path.join(ROOT, f"/ds2/model_zoos/zoos_sparsified/distillation/zoos/{tune_config['dataset']}/ARD/{zoo_p}")

    # path to "dataset" 
    data_root = os.path.join(ROOT, "/netscratch2/jlautz/model_robustness/src/model_robustness/data/sparsified")
    data_path = os.path.join(
        data_root, f"{tune_config['dataset']}/{tune_config['attack']}/{tune_config['setup']}/eps_{tune_config['eps']}", "perturbed_dataset.pt"
    )


    # Get all models to iterate over
    all_models_df = pd.read_csv(os.path.join(data_root, "clean_zoos", f"{tune_config['dataset']}_{tune_config['setup']}_clean_zoo.csv"))
    model_paths = all_models_df["path"].tolist()
            
    dataset = torch.load(data_path)
    
    # Iterate over all models in the zoo and evaluate them on the perturbed dataset
    for i, path in enumerate(model_paths):

        # Read in config containing the paramters for the i-th model
        model_config_path = os.path.join(old_result_path, path, "params.json")
        config_model = json.load(open(model_config_path, ))

        checkpoints = []
        for p in os.listdir(os.path.join(old_result_path, path)):
            if p.startswith("checkpoint"):
                checkpoints.append(p)
            checkpoints.sort()
        
        best_acc = 0
        index = None
        for l, line in enumerate(open(os.path.join(old_result_path, path, "result.json"), "r")):
            aux_dict = json.loads(line)
            if aux_dict["test_acc"] > best_acc:
                best_acc = aux_dict["test_acc"]
                index = l
        checkpoint = checkpoints[index]

    # For CIFAR10 use large ConvNet, small ConvNet for everything else
        if tune_config["dataset"] == "CIFAR10":
            # Define model and load in state
            model = CNN3_ARD(
                channels_in=config_model["model::channels_in"],
                nlin=config_model["model::nlin"],
                dropout=config_model["model::dropout"],
                init_type=config_model["model::init_type"]
            )
            try:
                model.load_state_dict(
                    torch.load(os.path.join(old_result_path, path, checkpoint, "checkpoints"))
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
                        torch.load(os.path.join(old_result_path, path, checkpoint, "checkpoints"))
                    )
                except RuntimeError:
                    model = CNN3_ARD(
                        channels_in=config_model["model::channels_in"],
                        nlin=config_model["model::nlin"],
                        dropout=0.5,
                        init_type=config_model["model::init_type"]
                    )
                    model.load_state_dict(
                        torch.load(os.path.join(old_result_path, path, checkpoint, "checkpoints"))
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
                    torch.load(os.path.join(old_result_path, path, checkpoint, "checkpoints"))
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
                        torch.load(os.path.join(old_result_path, path, checkpoint, "checkpoints"))
                    )
                except RuntimeError:
                    model = CNN_ARD(
                        channels_in=config_model["model::channels_in"],
                        nlin=config_model["model::nlin"],
                        dropout=0.5,
                        init_type=config_model["model::init_type"]
                    )
                    model.load_state_dict(
                        torch.load(os.path.join(old_result_path, path, checkpoint, "checkpoints"))
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
        save_results_path = Path(
            os.path.join(data_root, tune_config["dataset"], tune_config["attack"], tune_config["setup"], f"eps_{tune_config['eps']}", "results", path)
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
        
        print(f"Model {i+1}/{len(model_paths)} done.")


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
