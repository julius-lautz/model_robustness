import configargparse
import logging
import yaml
import sys
import os
import random
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.dataset import random_split

# import sys 
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname("/netscratch2/jlautz/model_robustness/src/model_robustness/attacks/model_soups"), '..')))

from networks import ResNet18, ConvNetLarge, ConvNetSmall

from advertorch.attacks import LinfPGDAttack

################ SETTING UP LOGGING ################

_logger = logging.getLogger(__name__)

def setup_logging(logLevel):
    logformat= "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=logLevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


################ CLI STUFF ################

def parse_args(args):
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    parser.add_argument("-c", "--config", required=True, is_config_file=True, help="config file path")
    parser.add_argument('--path_to_zoos', type=yaml.safe_load)
    parser.add_argument('--zoos', help='zoo to use', type=yaml.safe_load)
    parser.add_argument('--setups', help='setups', action='append')
    parser.add_argument('--model_list_paths', type=yaml.safe_load)
    parser.add_argument('--perturbed_datasets', type=yaml.safe_load)

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

def load_model(best_model_path, checkpoint_path, config):

    # Some models don't have 50 checkpoints, so check that we always take the last available one
    checkpoints = []
    for p in os.listdir(os.path.join(checkpoint_path, best_model_path)):
        if p.startswith("checkpoint"):
            checkpoints.append(p)
    checkpoints.sort()

    model_config_path = os.path.join(checkpoint_path, best_model_path, "params.json")
    config_model = json.load(open(model_config_path, ))

    if config["dataset"] == "CIFAR10":
        model = ConvNetLarge(
            channels_in=config_model["model::channels_in"],
            nlin=config_model["model::nlin"],
            dropout=config_model["model::dropout"],
            init_type=config_model["model::init_type"]
        )
        try:
            model.load_state_dict(
                torch.load(os.path.join(checkpoint_path, best_model_path, checkpoints[-1], "checkpoints"))
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
                    torch.load(os.path.join(checkpoint_path, best_model_path, checkpoints[-1], "checkpoints"))
                )
            except RuntimeError:
                model = ConvNetLarge(
                    channels_in=config_model["model::channels_in"],
                    nlin=config_model["model::nlin"],
                    dropout=0.5,
                    init_type=config_model["model::init_type"]
                )
                model.load_state_dict(
                    torch.load(os.path.join(checkpoint_path, best_model_path, checkpoints[-1], "checkpoints"))
                )
    else:
        model = ConvNetSmall(
            channels_in=config_model["model::channels_in"],
            nlin=config_model["model::nlin"],
            dropout=config_model["model::dropout"],
            init_type=config_model["model::init_type"]
        )
        try:
            model.load_state_dict(
                torch.load(os.path.join(checkpoint_path, best_model_path, checkpoints[-1], "checkpoints"))
            )
        except RuntimeError:
            model = ConvNetSmall(
                channels_in=config_model["model::channels_in"],
                nlin=config_model["model::nlin"],
                dropout=0,
                init_type=config_model["model::init_type"]
            )
            model.load_state_dict(
                torch.load(os.path.join(checkpoint_path, best_model_path, checkpoints[-1], "checkpoints"))
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

    adversary = LinfPGDAttack(
        model,
        loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        eps=0.1,
        nb_iter=10,
        eps_iter=0.01,
        rand_init=False,
        clip_min=0.0,
        clip_max=1.0,
        targeted=False
    )


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


def get_black_box_scenario_paths(config, df_root):

    scenario_paths = {
        "dropout": None,
        "nlin": None,
        "init_type": None,
        "optimizer": None
    }

    # Scenario 1: different dropout
    aux_df = df_root[(df_root.nlin==config["nlin"]) \
        & (df_root.optimizer==config["optimizer"]) \
        & (df_root.init_type==config["init_type"])]
    aux_list = []
    for k in aux_df.dropout.value_counts().keys():
        if k == config["dropout"]:
            continue
        temp_df = aux_df[aux_df.dropout==k].sort_values(by="old_acc", ascending=False).reset_index(drop=True)
        aux_dict = {k: temp_df.loc[0, "name"]}
        aux_list.append(aux_dict)
    scenario_paths["dropout"] = aux_list

    # Scenario 2: different nlin
    aux_df = df_root[(df_root.dropout==config["dropout"]) \
        & (df_root.optimizer==config["optimizer"]) \
        & (df_root.init_type==config["init_type"])]
    aux_list = []
    for k in aux_df.nlin.value_counts().keys():
        if k == config["nlin"]:
            continue
        temp_df = aux_df[aux_df.nlin==k].sort_values(by="old_acc", ascending=False).reset_index(drop=True)
        aux_dict = {k: temp_df.loc[0, "name"]}
        aux_list.append(aux_dict)
    scenario_paths["nlin"] = aux_list

    # Scenario 3: different init type
    aux_df = df_root[(df_root.nlin==config["nlin"]) \
        & (df_root.optimizer==config["optimizer"]) \
        & (df_root.dropout==config["dropout"])]
    aux_list = []
    for k in aux_df.init_type.value_counts().keys():
        if k == config["init_type"]:
            continue
        temp_df = aux_df[aux_df.init_type==k].sort_values(by="old_acc", ascending=False).reset_index(drop=True)
        aux_dict = {k: temp_df.loc[0, "name"]}
        aux_list.append(aux_dict)
    scenario_paths["init_type"] = aux_list

    # Scenario 4: different optimizer
    if config["dataset"] == "SVHN":
        pass
    else:
        aux_df = df_root[(df_root.nlin==config["nlin"]) \
            & (df_root.init_type==config["init_type"]) \
            & (df_root.dropout==config["dropout"])]
        aux_list = []
        for k in aux_df.optimizer.value_counts().keys():
            if k == config["optimizer"]:
                continue
            temp_df = aux_df[aux_df.optimizer==k].sort_values(by="old_acc", ascending=False).reset_index(drop=True)
            aux_dict = {k: temp_df.loc[0, "name"]}
            aux_list.append(aux_dict)
        scenario_paths["optimizer"] = aux_list

    return scenario_paths


def black_box_evalution(result_df, scenario_paths, perturbed_data, index_counter, config, root_path):

    scenario_list = ["normal", "white_box"]

    for i, category in enumerate(scenario_paths):

        # For SVHN there is only one optimizer, skip that iteration
        if config["dataset"] == "SVHN" and i==3:
            continue
        
        for j, param in enumerate(scenario_paths[category]):
            scenario_list.append(f"{category}_{next(iter(param))}")

            for path_key in param:

                # load model
                model, model_config = load_model(param[path_key], root_path, config)

                # evaluate on dataset
                loss, acc = evaluate(perturbed_data, model, model_config, config)

                result_df.loc[index_counter, f"{config['dataset']}"] = acc
                index_counter += 1

    result_df["scenarios"] = scenario_list
    
    return result_df

################ MAIN ################

def main(args):
    
    args = parse_args(args)
    setup_logging(args.loglevel)

    result_path = Path("/netscratch2/jlautz/model_robustness/src/model_robustness/data/all_results.csv")
    results_old = pd.read_csv(result_path, index_col=0)

    config = {}
    config["dropout"] = 0.0
    config["init_type"] = "kaiming_normal"
    config["nlin"] = "tanh"
    config["optimizer"] = "adam"
    config["lr"] = 0.001
    config["wd"] = 0.001
    config["seed"] = 8
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    # scenarios = ['normal', 'white_box', 'black_box_1','black_box_2','black_box_3','black_box_4','black_box_5']

    # result_df["scenarios"] = scenarios

    for ds in args.zoos["normal"]:

        # if ds == "MNIST" or ds == "CIFAR10":
        #     continue

        setup = "hyp-10-f"

        if ds == "SVHN":
            config["optimizer"] = "sgd"
        config["dataset"] = ds

        _logger.info(f"[{ds}]")

        result_df = pd.DataFrame()

        # Getting paths
        root_path = Path(args.path_to_zoos["normal"][ds][setup])
        data_path = os.path.join(root_path, "dataset.pt")
        dataset = torch.load(data_path)["testset"]

        df_root = results_old[
            (results_old.dataset==ds) \
            & (results_old.attack=="PGD") \
            & (results_old.setup==setup) \
            & (results_old.eps==0.1) \
            & (results_old.lr==config["lr"]) \
            & (results_old.wd==config["wd"])]

        # Get best model
        df = df_root[(df_root.dropout==config["dropout"]) \
            & (df_root.init_type==config["init_type"]) \
            & (df_root.nlin==config["nlin"]) \
            & (df_root.optimizer==config["optimizer"])]

        df_best = df.sort_values(by="old_acc", ascending=False).reset_index(drop=True)

        best_model_path = df_best.loc[0, "name"]

        # Get normal performance of best model
        _logger.info(f"[{ds}] - Getting normal performance")
        index_counter = 0
        result_df.loc[index_counter, f"{ds}"] = df_best.loc[0, "old_acc"]
        index_counter += 1

        # Load best model
        best_model, best_model_config = load_model(best_model_path, root_path, config)

        # Create perturbed images
        _logger.info(f"[{ds}] - Attacking")
        perturbed_data = attack(dataset, best_model, config)

        # Evaluate white box
        _logger.info(f"[{ds}] - White Box Evaluation")
        white_loss, white_acc = evaluate(perturbed_data, best_model, best_model_config, config)
        result_df.loc[index_counter, f"{ds}"] = white_acc
        index_counter += 1

        # Define black box scenarios
        scenario_paths = get_black_box_scenario_paths(config, df_root)

        # Evaluate black box scenarios
        _logger.info(f"[{ds}] - Black Box Evaluations")
        result_df = black_box_evalution(result_df, scenario_paths, perturbed_data, index_counter, config, root_path)

        # Save results in dataframe
        result_df.to_csv(f"/netscratch2/jlautz/model_robustness/src/model_robustness/results/transferability/{ds}_results.csv")


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()