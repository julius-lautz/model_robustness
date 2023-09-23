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

import sys 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname("/netscratch2/jlautz/model_robustness/src/model_robustness/attacks/model_soups"), '..')))

from attacks.networks import ResNet18, ConvNetLarge, ConvNetSmall

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


################ HELPER FUNCTION ################

def evaluate(dataset, model, config_model):

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
        imgs, labels = imgs.to(device), labels.to(device)
        n_b = labels.shape[0]

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        acc = np.sum(np.equal(np.argmax(outputs.cpu().data.numpy(), axis=-1),
            labels.cpu().data.numpy()))

        loss_avg += loss.item()
        acc_avg += acc
        num_exp += n_b
        # if j % 250 == 0:
        #     print(f"Batch {j} of {len(dataset)/config_model['training::batchsize']} done.")

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg


def get_model_w_state(type, ds, config_model, root_path, path, uniform_soup, device):

    if type == "normal":
        if ds == "CIFAR10":
           # Define model and load in state
            model = ConvNetLarge(
                channels_in=config_model["model::channels_in"],
                nlin=config_model["model::nlin"],
                dropout=config_model["model::dropout"],
                init_type=config_model["model::init_type"]
            )
            try:
                model.load_state_dict(uniform_soup)
            except RuntimeError:
                model = ConvNetLarge(
                    channels_in=config_model["model::channels_in"],
                    nlin=config_model["model::nlin"],
                    dropout=0,
                    init_type=config_model["model::init_type"]
                )
                try:
                    model.load_state_dict(uniform_soup)
                except RuntimeError:
                    model = ConvNetLarge(
                        channels_in=config_model["model::channels_in"],
                        nlin=config_model["model::nlin"],
                        dropout=0.5,
                        init_type=config_model["model::init_type"]
                    )
                    model.load_state_dict(uniform_soup)
        else:
            model = ConvNetSmall(
                channels_in=config_model["model::channels_in"],
                nlin=config_model["model::nlin"],
                dropout=config_model["model::dropout"],
                init_type=config_model["model::init_type"]
            )
            try:
                model.load_state_dict(uniform_soup)
            except RuntimeError:
                model = ConvNetSmall(
                    channels_in=config_model["model::channels_in"],
                    nlin=config_model["model::nlin"],
                    dropout=0,
                    init_type=config_model["model::init_type"]
                )
                model.load_state_dict(uniform_soup)

    else: 
        model = ResNet18(
            channels_in=config_model["model::channels_in"],
            out_dim=config_model["model::o_dim"],
            nlin=config_model["model::nlin"],
            dropout=config_model["model::dropout"],
            init_type=config_model["model::init_type"]
        )
        model.load_state_dict(uniform_soup)
    
    return model.to(device)


device = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path("")


def main(args):

    args = parse_args(args)
    setup_logging(args.loglevel)

    for z in args.zoos.keys():
        for ds in args.zoos[z]:
            for s in args.setups:

                # make distinction between resnet and normal
                if z == "resnet" and s != "seed":
                    continue
                
                if z == "resnet":

                    _logger.info(f"[{z}][{ds}]")

                    root_path = args.path_to_zoos[z][ds]
                    data_path = os.path.join(args.path_to_zoos[z][ds], "dataset.pt")
                    model_list_path = args.model_list_paths[z][ds]
                    perturbed_data_path = args.perturbed_datasets[z][ds]
                    result_path = Path("../../data/resnet/all_results_resnet_df.csv")

                else:
                    _logger.info(f"[{z}][{ds}][{s}]")
                    root_path = args.path_to_zoos[z][ds][s]
                    data_path = os.path.join(args.path_to_zoos[z][ds][s], "dataset.pt")
                    model_list_path = args.model_list_paths[z][ds][s]
                    perturbed_data_path = args.perturbed_datasets[z][ds][s]
                    result_path = Path("../../data/all_results.csv")

                # Read in dataset
                dataset = torch.load(data_path)["testset"]

                # Read in new results
                if z == "normal":
                    df = pd.read_csv(result_path, index_col=0)
                else:
                    df = pd.read_csv(result_path)
                df = df[df.dataset == ds]
                if z == "normal":
                    df = df[df.setup == s]
                df = df[df.attack == "PGD"]
                df = df[df.eps == 0.1]

                model_paths = df["name"].tolist()

                # Creating empty result dataframe
                results = pd.DataFrame(columns=["name", "normal_acc", "normal_acc_attack", "soup_acc", "soup_acc_attack"])

                # Create model_soup
                # Average the weights for the last 5 epochs
                for i, path in enumerate(model_paths):

                    # Get config and load model
                    config_path_temp = os.path.join(ROOT, root_path, df[df.name == path].iloc[0, 0])
                    config_temp = json.load(open(os.path.join(config_path_temp, "params.json"), ))

                    # Get the paths for the last five checkpoints
                    checkpoints = []
                    for p in os.listdir(os.path.join(root_path, path)):
                        if p.startswith("checkpoint"):
                            checkpoints.append(p)
                        checkpoints.sort()
                    # Only take the last 5 checkpoints
                    checkpoints = checkpoints[-5:]

                    for j, epoch in enumerate(checkpoints):
                        
                        aux_path = os.path.join(root_path, path, epoch, "checkpoints")
                        state_dict = torch.load(aux_path)

                        if j == 0:
                            uniform_soup = {k: v * (1./(len(checkpoints))) for k, v in state_dict.items()}

                        else: 
                            uniform_soup = {k: v * (1./(len(checkpoints))) + uniform_soup[k] for k, v in state_dict.items()}

                    model = get_model_w_state(z, ds, config_temp, root_path, path, uniform_soup, device)

                    _logger.info(f"[{z}][{ds}][{s}] - Evaluate model soup for Model {i+1}/{len(model_paths)}.")
                    # Evaluate Model Soup
                    soup_loss, soup_acc = evaluate(dataset, model, config_temp)

                    # Getting perturbed dataset
                    _logger.info(f"[{z}][{ds}][{s}] - Evaluate perturbed model soup for Model {i+1}/{len(model_paths)}.")
                    perturbed_data = torch.load(perturbed_data_path)
                    soup_loss_attack, soup_acc_attack = evaluate(perturbed_data, model, config_temp)

                    # Getting old acc and loss from df
                    aux_df = df[df.name==path]
                    normal_acc = aux_df["old_acc"].item()
                    normal_acc_attack = aux_df["new_acc"].item()

                    results.loc[i, "name"] = path
                    results.loc[i, "normal_acc"] = normal_acc
                    results.loc[i, "normal_acc_attack"] = normal_acc_attack
                    results.loc[i, "soup_acc"] = soup_acc
                    results.loc[i, "soup_acc_attack"] = soup_acc_attack

                    # if i % 50 == 0:
                    #     print(f"Model {i+1}/{len(model_paths)} done.")

                    # print(f"MODEL SOUP {i+1}/{len(model_paths)}: New Accuracy = {soup_acc} ({old_acc}); \n After Attack Accuracy = {perturbed_acc}({old_attack_acc})")

                df_save_path = Path("/netscratch2/jlautz/model_robustness/src/model_robustness/results/soups")
                results.to_csv(os.path.join(df_save_path, f"{z}_{ds}_{s}_df.csv"))


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()

