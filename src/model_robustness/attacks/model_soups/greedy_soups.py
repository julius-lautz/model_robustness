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


def get_model_w_state(type, ds, config_model, uniform_soup, device):

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

    index_counter = 0
    for z in args.zoos.keys():
        for ds in args.zoos[z]:
            for s in args.setups:

                if ds == "MNIST":
                    continue
                elif s != "seed":
                    continue

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
                

                # Read in datasets
                dataset = torch.load(data_path)["testset"]
                valset = torch.load(data_path)["valset"]
                perturbed_data = torch.load(perturbed_data_path)

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

                # Rank models according to the decreasing validation accuracy
                # Iterate through the model_paths and get their validation accuracy
                ranked = pd.DataFrame(columns=["name", "val_acc", "last_checkpoint"])
                for k, path in enumerate(model_paths):
                    # take last checkpoint
                    checkpoints = []
                    for p in os.listdir(os.path.join(root_path, path)):
                        if p.startswith("checkpoint"):
                            checkpoints.append(p)
                    checkpoints.sort()
                    for l, line in enumerate(open(os.path.join(root_path, path, "result.json"), "r")):

                        if l == len(checkpoints)-1:
                            aux_dic = json.loads(line)
                            ranked.loc[k, "name"] = path
                            ranked.loc[k, "val_acc"] = aux_dic["validation_acc"]
                            ranked.loc[k, "last_checkpoint"] = checkpoints[-1]
                ranked = ranked.sort_values(by="val_acc", ascending=False).reset_index()
                ranked_model_paths = ranked["name"].tolist()
                last_checkpoints = ranked["last_checkpoint"].tolist()

                # Creating empty result dataframe
                results = pd.DataFrame(columns=["type", "ds", "setup", "normal_acc", "normal_acc_attack", "soup_acc", "soup_acc_attack"])

                # Create model_soup
                # Take first model as first ingredient
                greedy_soup_ingredients = [ranked_model_paths[0]]
                # Load parameters
                greedy_soup_params = torch.load(os.path.join(root_path, ranked_model_paths[0], last_checkpoints[0], "checkpoints"))
                best_val_acc_so_far = ranked.loc[0, "val_acc"]

                # Iterate through all models and consider adding them to the greedy soup
                for i, path in enumerate(ranked_model_paths):
                    
                    # Skip first model
                    if i == 0:
                        continue
                    _logger.info(f"Testing model {i} of {len(ranked_model_paths)}")

                    # Get config and load model
                    config_path_temp = os.path.join(ROOT, root_path, path)
                    config_temp = json.load(open(os.path.join(config_path_temp, "params.json"), ))

                    new_ingredient_params = torch.load(os.path.join(root_path, path, last_checkpoints[i], "checkpoints"))
                    num_ingredients = len(greedy_soup_ingredients)
                    potential_greedy_soup_params = {
                        k: greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.)) +
                        new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
                        for k in new_ingredient_params
                    }

                    # Evaluate the potential greedy soup on the valset
                    model = get_model_w_state(z, ds, config_temp, potential_greedy_soup_params, device)
                    held_out_val_loss, held_out_val_acc = evaluate(valset, model, config_temp)
                    _logger.info(f"Potential greedy soup val acc: {held_out_val_acc}, best so far: {best_val_acc_so_far}")
                    if held_out_val_acc > best_val_acc_so_far:
                        greedy_soup_ingredients.append(ranked_model_paths[i])
                        best_val_acc_so_far = held_out_val_acc
                        greedy_soup_params = potential_greedy_soup_params
                        _logger.info(f"Adding to soup, new soup is {greedy_soup_ingredients}")

                # Evaluate soup on the testset before and after attack
                best_model_config = json.load(open(os.path.join(ROOT, root_path, ranked_model_paths[0], "params.json"), ))
                model = get_model_w_state(z, ds, best_model_config, greedy_soup_params, device)

                results.loc[index_counter, "z"] = z
                results.loc[index_counter, "ds"] = ds
                results.loc[index_counter, "setup"] = s
                results.loc[index_counter, "normal_acc"] = df[df.name==ranked_model_paths[0]]["old_acc"].item()
                results.loc[index_counter, "normal_acc_attack"] = df[df.name==ranked_model_paths[0]]["new_acc"].item()

                _logger.info(f"[{z}][{ds}][{s}] - Evaluate greedy soup on normal dataset")
                greedy_loss, greedy_acc = evaluate(dataset, model, best_model_config)

                _logger.info(f"[{z}][{ds}][{s}] - Evaluate greedy soup on perturbed dataset")
                greedy_loss_attack, greedy_acc_attack = evaluate(perturbed_data, model, best_model_config)

                results.loc[index_counter, "soup_acc"] = greedy_acc
                results.loc[index_counter, "soup_acc_attack"] = greedy_acc_attack

                _logger.info(results.loc[index_counter, :])

                index_counter += 1

        #         break
        #     break
        # break
    

                    
    df_save_path = Path("/netscratch2/jlautz/model_robustness/src/model_robustness/results/soups")
    results.to_csv(os.path.join(df_save_path, f"greedy_soup_df.csv"))


def run():
    main(sys.argv[1:])


if __name__ == "__main__":
    run()

