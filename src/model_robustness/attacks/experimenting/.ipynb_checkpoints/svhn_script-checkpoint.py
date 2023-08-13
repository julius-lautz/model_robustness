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

from ray import tune, air
import ray
from ray.air.integrations.wandb import WandbLoggerCallback

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.dataset import random_split

from advertorch.attacks import LinfPGDAttack, GradientSignAttack

from model_robustness.attacks.networks import ConvNetLarge, ConvNetSmall

ROOT = Path("")

_logger = logging.getLogger(__name__)


def setup_logging(logLevel):
    logformat= "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=logLevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args(args):
    parser = argparse.ArgumentParser(description="SVHN experiments")
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


def black_box_evaluation(perturbed_data, source_model, source_config, config, tune_config):

    total_acc, total_loss, n_models = 0, 0, 0

    for i, path in enumerate(tune_config["path_list"]):
        print(f"checking model {i} of {len(tune_config["path_list"])}")
        # Read in config containing params for the i-th model
        model_config_path = os.path.join(tune_config["checkpoint_path"], path, "params.json")
        config_model = json.load(open(model_config_path, ))

        if config_model["model::nlin"] != source_config["model::nlin"] and config_model["model::init_type"] != source_config["model::init_type"]:
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
            loss_avg, acc_avg = evaluate(perturbed_data, source_model, source_config, config)

            total_acc += acc_avg
            total_loss += loss_avg
            n_models += 1

        else:
            continue

    # Calculate loss and accuracy averages
    total_acc /= n_models
    total_loss /= n_models

    return total_loss, total_acc, n_models


def transferability_experiments(tune_config, dataset, perturbed_data, source_model, source_config, config):

    # Normal evaluation
    if tune_config["experiment"] == "normal":
        loss_avg, acc_avg = evaluate(dataset, source_model, source_config, config)
        air.session.report(metrics={"acc_avg": acc_avg, "loss_avg": loss_avg})

    # White-box evaluation
    elif tune_config["experiment"] == "white_box":
        loss_avg, acc_avg = evaluate(perturbed_data, source_model, source_config, config)
        air.session.report(metrics={"acc_avg": acc_avg, "loss_avg": loss_avg})

    # Black-box evaluation
    elif tune_config["experiment"] == "black_box":
        loss_avg, acc_avg, n_models = black_box_evaluation(perturbed_data, source_model, source_config, config, tune_config)
        air.session.report(metrics={"acc_avg": acc_avg, "loss_avg": loss_avg, "n_models": n_models})



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

    # Delete max_index from path list
    path_list_wo_max = path_list
    del path_list_wo_max[max_index]

    # Load best model
    _logger.info("Loading best model")
    source_model, source_config = load_model(best_model_path, checkpoint_path, config)

    # Create perturbed images
    _logger.info("Attacking")
    perturbed_data = attack(dataset, source_model, config)

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
        "path_list": path_list_wo_max,
        "experiment": tune.grid_search(["normal", "white_box", "black_box"])
    }

    transferability_experiments_w_resources = tune.with_resources(transferability_experiments, resources_per_trial)

    tuner = tune.Tuner(
        tune.with_parameters(
            transferability_experiments_w_resources, dataset=dataset,
            perturbed_data=perturbed_data,
            source_model=source_model,
            source_config=source_config,
            config=config),
        tune_config=tune.TuneConfig(num_samples=1),
        run_config=air.RunConfig(
            callbacks=[WandbLoggerCallback(
                project="master_thesis", api_key="7fe80de0b53b0ab265297295a37223f3e9cb1215"
            )],
            storage_path="/netscratch2/jlautz/ray_results"
        ),
        param_space=search_space
    )
    results = tuner.fit()

    ray.shutdown()

    # # Evaluate normal performance
    # _logger.info("Evaluating normal performance")
    # normal_loss, normal_acc = evaluate(dataset, source_model, source_config, config)
    # _logger.info(f"Normal performance: Loss = {normal_loss}, Accuracy = {normal_acc}")

    # # Evaluate white-box attack
    # _logger.info("Evaluating white-box performance")
    # white_loss, white_acc = evaluate(perturbed_data, source_model, source_config, config)
    # _logger.info(f"White-box performance: Loss = {white_loss}, Accuracy = {white_acc}")


def run():
    main(sys.argv[1:])


if __name__=="__main__":
    run()

