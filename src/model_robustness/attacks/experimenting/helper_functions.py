import argparse
import logging
import sys
import os
import random
import json
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from typing import Dict, List

from ray import tune, air
import ray
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune.logger import LoggerCallback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.dataset import random_split

from advertorch.attacks import LinfPGDAttack, GradientSignAttack

from model_robustness.attacks.networks import ConvNetLarge, ConvNetSmall


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


def load_model(best_model_path, checkpoint_path, config):

    model_config_path = os.path.join(checkpoint_path, best_model_path, "params.json")
    config_model = json.load(open(model_config_path, ))

    model = ConvNetSmall(
        channels_in=config_model["model::channels_in"],
        nlin=config_model["model::nlin"],
        dropout=config_model["model::dropout"],
        init_type=config_model["model::init_type"]
    )

    try:
        model.load_state_dict(
            torch.load(os.path.join(checkpoint_path, best_model_path, "checkpoint_000050", "checkpoints"))
        )
    except RuntimeError:
        model = ConvNetSmall(
            channels_in=config_model["model::channels_in"],
            nlin=config_model["model::nlin"],
            dropout=0,
            init_type=config_model["model::init_type"]
        )
        model.load_state_dict(
            torch.load(os.path.join(checkpoint_path, best_model_path, "checkpoint_000050", "checkpoints"))
        )
        
    model.to(config["device"])

    return model, config_model


def attack(dataset, model, config):

    aux_loader = DataLoader(
        dataset=dataset,
        batch_size=len(dataset),
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
            eps=config["eps"],
            targeted=False
        )
    else:
        raise NotImplementedError("error: attack type unknown")

    # Perturb images
    adv_images = adversary.perturb(cln_data, true_labels)

    adv_data = torch.utils.data.TensorDataset(adv_images, true_labels)

    return adv_data


def evaluate(dataset, model, config_model, config):

    # Define dataloader for evaluation
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
        imgs, labels = imgs.to(config["device"]), labels.to(config["device"])
        n_b = labels.shape[0]

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        acc = np.sum(np.equal(np.argmax(outputs.cpu().data.numpy(), axis=-1),
            labels.cpu().data.numpy()))

        loss_avg += loss.item()
        acc_avg += acc
        num_exp += n_b
        if j % 250 == 0:
            _logger.debug(f"Batch {j} of {len(dataset)/config_model['training::batchsize']} done.")

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg


def black_scenarios_path_lists(source_config, path_list, checkpoint_path):

    scen_1, scen_2, scen_3, scen_4, scen_5 = [], [], [], [], []

    for i, path in enumerate(path_list):

        model_config_path = os.path.join(checkpoint_path, path, "params.json")
        config_model = json.load(open(model_config_path, ))

        # Scenario 1: different dropout
        if config_model["model::nlin"] == source_config["model::nlin"] \
        and config_model["model::init_type"] == source_config["model::init_type"] \
        and config_model["optim::lr"] == source_config["optim::lr"] \
        and config_model["optim::wd"] == source_config["optim::wd"] \
        and config_model["model::dropout"] != source_config["model::dropout"]:
            scen_1.append(path)

        # Scenario 2: different weight decay
        elif config_model["model::nlin"] == source_config["model::nlin"] \
        and config_model["model::init_type"] == source_config["model::init_type"] \
        and config_model["optim::lr"] == source_config["optim::lr"] \
        and config_model["optim::wd"] != source_config["optim::wd"] \
        and config_model["model::dropout"] == source_config["model::dropout"]:
            scen_2.append(path)

        # Scenario 3: different lr
        elif config_model["model::nlin"] == source_config["model::nlin"] \
        and config_model["model::init_type"] == source_config["model::init_type"] \
        and config_model["optim::lr"] != source_config["optim::lr"] \
        and config_model["optim::wd"] == source_config["optim::wd"] \
        and config_model["model::dropout"] == source_config["model::dropout"]:
            scen_3.append(path)

        # Scenario 4: different init_type
        elif config_model["model::nlin"] == source_config["model::nlin"] \
        and config_model["model::init_type"] != source_config["model::init_type"] \
        and config_model["optim::lr"] == source_config["optim::lr"] \
        and config_model["optim::wd"] == source_config["optim::wd"] \
        and config_model["model::dropout"] == source_config["model::dropout"]:
            scen_4.append(path)

        # Scenario 5: different activation
        elif config_model["model::nlin"] != source_config["model::nlin"] \
        and config_model["model::init_type"] == source_config["model::init_type"] \
        and config_model["optim::lr"] == source_config["optim::lr"] \
        and config_model["optim::wd"] == source_config["optim::wd"] \
        and config_model["model::dropout"] == source_config["model::dropout"]:
            scen_5.append(path)

    return [scen_1, scen_2, scen_3, scen_4, scen_5]


def black_box_evaluation(perturbed_data, scen, config, tune_config):

    total_acc, total_loss, n_models = 0, 0, 0

    for i, path in enumerate(scen):
        print(f"SCENARIO: {tune_config['experiment']}, checking model {i} of {len(scen)}")
        # Read in config containing params for the i-th model
        model_config_path = os.path.join(tune_config["checkpoint_path"], path, "params.json")
        config_model = json.load(open(model_config_path, ))

        # Define model and load in state
        model = ConvNetSmall(
            channels_in=config_model["model::channels_in"],
            nlin=config_model["model::nlin"],
            dropout=config_model["model::dropout"],
            init_type=config_model["model::init_type"]
        )
        try:
            model.load_state_dict(
                torch.load(os.path.join(tune_config["checkpoint_path"], path, "checkpoint_000050", "checkpoints"))
            )
        except RuntimeError:
            model = ConvNetSmall(
                channels_in=config_model["model::channels_in"],
                nlin=config_model["model::nlin"],
                dropout=0,
                init_type=config_model["model::init_type"]
            )
            model.load_state_dict(
                torch.load(os.path.join(tune_config["checkpoint_path"], path, "checkpoint_000050", "checkpoints"))
            )
        model.to(config["device"])

        # Calculate performance
        loss_avg, acc_avg = evaluate(perturbed_data, model, config_model, config)

        total_acc += acc_avg
        total_loss += loss_avg
        n_models += 1


    # Calculate loss and accuracy averages
    total_acc /= n_models
    total_loss /= n_models

    return total_loss, total_acc, n_models