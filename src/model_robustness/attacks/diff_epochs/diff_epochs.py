######### Script for analyzing robustness - accuracy tradeoff #########

import os
import json
from pathlib import Path
import configargparse
import yaml
import logging 

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

import sys 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname("/netscratch2/jlautz/model_robustness/src/model_robustness/attacks/diff_epochs"), '..')))

from attacks.networks import ResNet18, ConvNetLarge, ConvNetSmall

ROOT = Path("")
device = "cuda" if torch.cuda.is_available() else "cpu"


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

def get_model_w_state(type, ds, config_model, root_path, path, checkpoints, ep, device):

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
                model.load_state_dict(
                    torch.load(os.path.join(root_path, path, checkpoints[ep], "checkpoints"))
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
                        torch.load(os.path.join(root_path, path, checkpoints[ep], "checkpoints"))
                    )
                except RuntimeError:
                    model = ConvNetLarge(
                        channels_in=config_model["model::channels_in"],
                        nlin=config_model["model::nlin"],
                        dropout=0.5,
                        init_type=config_model["model::init_type"]
                    )
                    model.load_state_dict(
                        torch.load(os.path.join(root_path, path, checkpoints[ep], "checkpoints"))
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
                    torch.load(os.path.join(root_path, path, checkpoints[ep], "checkpoints"))
                )
            except RuntimeError:
                model = ConvNetSmall(
                    channels_in=config_model["model::channels_in"],
                    nlin=config_model["model::nlin"],
                    dropout=0,
                    init_type=config_model["model::init_type"]
                )
                model.load_state_dict(
                    torch.load(os.path.join(root_path, path, checkpoints[ep], "checkpoints"))
                )

    else: 
        model = ResNet18(
            channels_in=config_model["model::channels_in"],
            out_dim=config_model["model::o_dim"],
            nlin=config_model["model::nlin"],
            dropout=config_model["model::dropout"],
            init_type=config_model["model::init_type"]
        )
        model.load_state_dict(
            torch.load(os.path.join(root_path, path, checkpoints[ep], "checkpoints"))
        )
    
    return model.to(device)


def evaluate(criterion, loader, model, device):
    loss_avg, acc_avg, num_exp = 0, 0, 0
                        
    for k, data in enumerate(loader):
        
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

    return loss_avg, acc_avg


def compute_kendalls_tau(df, df_save_path, z, ds, s):
    kendall_df = pd.DataFrame(columns=["epoch", "kendalls_tau", "p-value"])
    epochs = ["epoch_10", "epoch_20", "epoch_30", "epoch_40", "epoch_50"]

    for i, ep in enumerate(epochs):
        aux_df = df[[ep, f"{ep}_attack"]]
        
        perturbed_df = aux_df.sort_values(f"{ep}_attack", ascending=False)
        aux_df = aux_df.sort_values(ep, ascending=False)
        
        perturbed_df.insert(0, "order", range(1, 1+len(perturbed_df)))
        aux_df.insert(0, "order", range(1, 1+len(aux_df)))
        
        # Sort perturbed_df according to names of aux_df again
        aux_list = list(aux_df.index)
        perturbed_df["index"] = perturbed_df.index
        perturbed_df.sort_values(by="index", key=lambda column:column.map(lambda e: aux_list.index(e)), inplace=True)

        
        # Calculating kendall's tau
        tau, p_value = stats.kendalltau(aux_df["order"], perturbed_df["order"])
        kendall_df.loc[i, "epoch"] = ep
        kendall_df.loc[i, "kendalls_tau"] = tau
        kendall_df.loc[i, "p-value"] = p_value

    kendall_df.to_csv(os.path.join(df_save_path, f"kendalls/{z}_{ds}_{s}_df.csv"))

    return kendall_df

    

### Steps ###
# Read in zoo

# For all 50 models from model list
# - get normal results for epochs 10, 20, 30, 40, 50
# - evaluate on perturbed dataset for epochs 10, 20, 30, 40, 50
# - plot results (boxplot, lineplot) for all epochs
# - compute kendalls tau
# - save result from kendalls tau in dataframe
# - get plot for only epochs where kendalls tau i


def main(args):
    args = parse_args(args)
    setup_logging(args.loglevel)

    for z in args.zoos.keys():
        for ds in args.zoos[z]:
            for s in args.setups:

                # added to make sure we don't rerun some combinations
                if z == "normal" and ds == "CIFAR10" and s != "seed":
                    continue

                # make distinction between resnet and normal
                if z == "resnet" and s != "seed":
                    continue
                
                if z == "resnet":

                    _logger.info(f"[{z}][{ds}]")

                    root_path = args.path_to_zoos[z][ds]
                    model_list_path = args.model_list_paths[z][ds]
                    data_path = args.perturbed_datasets[z][ds]

                else:
                    _logger.info(f"[{z}][{ds}][{s}]")
                    root_path = args.path_to_zoos[z][ds][s]
                    model_list_path = args.model_list_paths[z][ds][s]
                    data_path = args.perturbed_datasets[z][ds][s]


                # Getting model_list
                with open(model_list_path, "r") as items:
                    model_paths = items.readlines()

                    for i, l in enumerate(model_paths):
                        model_paths[i] = l.replace("\n", "")
                
                # Loading dataset
                dataset = torch.load(data_path, map_location=torch.device(device))

                # Instantiating dataframe
                df = pd.DataFrame(columns=["epoch_10", "epoch_20", "epoch_30", "epoch_40", "epoch_50", "epoch_10_attack", "epoch_20_attack", "epoch_30_attack", "epoch_40_attack", "epoch_50_attack"])
                epochs_iter = [-41, -31, -21, -11, -1]
                epochs_df = [10, 20, 30, 40, 50]
                for j, path in enumerate(model_paths):
                    
                    _logger.info(f"[{z}][{ds}][{s}] - Getting normal results for Model {j+1}/50.")
                    for i, line in enumerate(open(os.path.join(root_path, path, "result.json"), 'r')):
                        
                        if i == 10:
                            aux_dic = json.loads(line)
                            df.loc[j, "epoch_10"] = aux_dic["test_acc"]
                        elif i == 20:
                            aux_dic = json.loads(line)
                            df.loc[j, "epoch_20"] = aux_dic["test_acc"]
                        elif i == 30:
                            aux_dic = json.loads(line)
                            df.loc[j, "epoch_30"] = aux_dic["test_acc"]
                        elif i==40:
                            aux_dic = json.loads(line)
                            df.loc[j, "epoch_40"] = aux_dic["test_acc"]
                        elif i == 50:
                            aux_dic = json.loads(line)
                            df.loc[j, "epoch_50"] = aux_dic["test_acc"]
                            
                            
                    model_config_path = os.path.join(root_path, path, "params.json")
                    config_model = json.load(open(model_config_path, ))
                    
                    
                    # Some models don't have 50 checkpoints, so check that we always take the last available one
                    checkpoints = []
                    for p in os.listdir(os.path.join(root_path, path)):
                        if p.startswith("checkpoint"):
                            checkpoints.append(p)
                    checkpoints.sort()
                    
                    
                    _logger.info(f"[{z}][{ds}][{s}] - Evaluating Model {j+1}/50 on attacks.")
                    for i, ep in enumerate(epochs_iter):

                        # Getting the correct model
                        model = get_model_w_state(z, ds, config_model, root_path, path, checkpoints, ep, device)
                        
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
                        
                        loss_avg, acc_avg = evaluate(criterion, loader, model, device)

                        df.loc[j, f"epoch_{epochs_df[i]}_attack"] = acc_avg
                        
                    _logger.info(f"[{z}][{ds}][{s}] - Model {j+1}/50 done.")

                df_save_path = Path("/netscratch2/jlautz/model_robustness/src/model_robustness/results/diff_epochs_dfs")
                df.to_csv(os.path.join(df_save_path, f"normal/{z}_{ds}_{s}_df.csv"))

                # Make plots
                plot_path = Path("/netscratch2/jlautz/model_robustness/src/model_robustness/plots/diff_epochs")
                sns.displot(data=df, kde=True)
                plt.savefig(
                    os.path.join(plot_path, f"normal/{z}_{ds}_{s}_displot.png"), 
                    bbox_inches="tight"
                )
                plt.clf()
                _logger.info(f"Full plot done")

                # Compute kendall's tau
                kendall_df = compute_kendalls_tau(df, df_save_path, z, ds, s)

                # Filter out all rows where p-value bigger than 0.05
                kendall_df = kendall_df[kendall_df["p-value"] <= 0.05]

                # Skip if no epochs are relevant
                if len(kendall_df) == 0:
                    _logger.info(f"[{z}][{ds}][{s}] - No relevant epochs")
                    continue
                
                # Make plots with only relevant columns
                plot_path = Path("/netscratch2/jlautz/model_robustness/src/model_robustness/plots/diff_epochs")
                sns.displot(data=df, kde=True)
                plt.savefig(
                    os.path.join(plot_path, f"kendalls/{z}_{ds}_{s}_displot.png"), 
                    bbox_inches="tight"
                )
                plt.clf()
                _logger.info(f"Kendall plot done")




def run():
    main(sys.argv[1:])


if __name__=="__main__":
    run()