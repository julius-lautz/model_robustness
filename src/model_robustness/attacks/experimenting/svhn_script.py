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

ROOT = Path("")

################ SETTING UP LOGGING ################

_logger = logging.getLogger(__name__)

def setup_logging(logLevel):
    logformat= "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=logLevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )

################ CLI STUFF ################

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

################ HELPER FUNCTIONS ################

def trial_str_creator(trial):
    return f"{trial.config['experiment']}_setup"


def load_zoo(zoo_name, setup):
    
    ROOT = Path("")

    if setup == "fixed":
        s = "fix"
    elif setup == "random":
        s = "rand"

    checkpoint_path = ROOT.joinpath(f'/netscratch2/dtaskiran/zoos/{zoo_name}/tune_zoo_{zoo_name.lower()}_hyperparameter_10_{setup}_seeds')
    _logger.debug(checkpoint_path)
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
    
    _logger.debug(f"best model has accuracy = {max(acc_list)}")

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
        # if j % 250 == 0:
        #     _logger.debug(f"Batch {j} of {len(dataset)/config_model['training::batchsize']} done.")

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

        if i%100==0:
            print(f"Checked {i} out of {len(path_list)} models")

    return [scen_1, scen_2, scen_3, scen_4, scen_5]


def black_box_evaluation(perturbed_data, scen, config, tune_config):

    total_acc, total_loss, n_models = 0, 0, 0

    for i, path in enumerate(scen):
        print(f"SCENARIO: {tune_config['experiment']}, checking model {i+1} of {len(scen)}")
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


################ TUNE EXPERIMENTS ################


def transferability_experiments(tune_config, dataset, perturbed_data, source_model, source_config, all_scens, config):

    # Normal evaluation
    if tune_config["experiment"] == "normal":
        normal_loss, normal_acc = evaluate(dataset, source_model, source_config, config)
        n_models_normal = 1
        air.session.report(metrics={"acc_avg": normal_acc, "loss_avg": normal_loss, "n_models": n_models_normal})

    # White-box evaluation
    elif tune_config["experiment"] == "white_box":
        white_loss, white_acc = evaluate(perturbed_data, source_model, source_config, config)
        n_models_white = 1
        air.session.report(metrics={"acc_avg": white_acc, "loss_avg": white_loss, "n_models": n_models_white})

    # Black-box 1 evaluation
    elif tune_config["experiment"] == "black_box_1":
        black_loss_1, black_acc_1, n_models_black_1 = black_box_evaluation(perturbed_data, all_scens[0], config, tune_config)
        air.session.report(metrics={"acc_avg": black_acc_1, "loss_avg": black_loss_1, "n_models": n_models_black_1})

    # Black-box 2 evaluation
    elif tune_config["experiment"] == "black_box_2":
        black_loss_2, black_acc_2, n_models_black_2 = black_box_evaluation(perturbed_data, all_scens[1], config, tune_config)
        air.session.report(metrics={"acc_avg": black_acc_2, "loss_avg": black_loss_2, "n_models": n_models_black_2})

    # Black-box 3 evaluation
    elif tune_config["experiment"] == "black_box_3":
        black_loss_3, black_acc_3, n_models_black_3 = black_box_evaluation(perturbed_data, all_scens[2], config, tune_config)
        air.session.report(metrics={"acc_avg": black_acc_3, "loss_avg": black_loss_3, "n_models": n_models_black_3})

    # Black-box 4 evaluation
    elif tune_config["experiment"] == "black_box_4":
        black_loss_4, black_acc_4, n_models_black_4 = black_box_evaluation(perturbed_data, all_scens[3], config, tune_config)
        air.session.report(metrics={"acc_avg": black_acc_4, "loss_avg": black_loss_4, "n_models": n_models_black_4})

    # Black-box 5 evaluation
    elif tune_config["experiment"] == "black_box_5":
        black_loss_5, black_acc_5, n_models_black_5 = black_box_evaluation(perturbed_data, all_scens[4], config, tune_config)
        air.session.report(metrics={"acc_avg": black_acc_5, "loss_avg": black_loss_5, "n_models": n_models_black_5})



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
    #del path_list_wo_max[max_index]

    # Load best model
    _logger.info("Loading best model")
    source_model, source_config = load_model(best_model_path, checkpoint_path, config)

    # Create perturbed images
    _logger.info("Attacking")
    perturbed_data = attack(dataset, source_model, config)

    # Getting the path lists for the black box scenarios
    _logger.info("Getting the path lists for the black box scenarios")
    all_scens = black_scenarios_path_lists(source_config, path_list, checkpoint_path)

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
        "experiment": tune.grid_search(["normal", "white_box", "black_box_1", "black_box_2", "black_box_3", "black_box_4", "black_box_5"]),
    }

    transferability_experiments_w_resources = tune.with_resources(transferability_experiments, resources_per_trial)

    tuner = tune.Tuner(
        tune.with_parameters(
            transferability_experiments_w_resources, dataset=dataset,
            perturbed_data=perturbed_data,
            source_model=source_model,
            source_config=source_config,
            all_scens=all_scens,
            config=config),
        tune_config=tune.TuneConfig(
            num_samples=1,
            trial_name_creator=trial_str_creator),
        run_config=air.RunConfig(
            #storage_path="/netscratch2/jlautz/ray_results",
            callbacks=[WandbLoggerCallback(
                project="master_thesis",
                api_key="7fe80de0b53b0ab265297295a37223f3e9cb1215",
                #group="SVHN FGSM hyp-10-r"
            )]
        ),
        param_space=search_space
    )
    results = tuner.fit()

    ray.shutdown()


def run():
    main(sys.argv[1:])


if __name__=="__main__":
    run()

