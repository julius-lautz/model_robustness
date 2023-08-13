import argparse
import logging
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
from networks import ResNet18, ConvNetLarge, ConvNetSmall


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


device = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path("")
# Define paths
dataset_path = os.path.join(ROOT, "/ds2/model_zoos/zoos_v2/SVHN/tune_zoo_svhn_hyperparameter_10_fixed_seeds/dataset.pt")
old_result_path = os.path.join(ROOT, "/ds2/model_zoos/zoos_v2/SVHN/tune_zoo_svhn_hyperparameter_10_fixed_seeds/")

# Read in dataset
ds = torch.load(dataset_path)["testset"]

# Read in new results
df = pd.read_csv("../data/all_results.csv")
df = df[df.dataset == "SVHN"]
df = df[df.setup == "hyp-10-f"]
df = df[df.attack == "PGD"]
df = df[df.eps == 0.1]

# Only take seed=1, init_type=kaiming_uniform, nlin=tanh
#df = df[(df.seed==1) & (df.init_type=="kaiming_uniform") & (df.nlin=="tanh")]

model_paths = df["name"].tolist()

# # Get best model and calculate old accuracy
# best_model_config_path = os.path.join(ROOT, old_result_path, df[df.old_acc == df.old_acc.max()].iloc[0,1])
# best_model_config = json.load(open(os.path.join(best_model_config_path, "params.json"), ))

# best_model = ConvNetLarge(
#     channels_in=best_model_config["model::channels_in"],
#     nlin=best_model_config["model::nlin"],
#     dropout=0.5,
#     init_type=best_model_config["model::init_type"]
# )
# best_model.load_state_dict(
#     torch.load(os.path.join(best_model_config_path, "checkpoint_000050", "checkpoints"))
# )
# best_model.to(device)

# # Evaluate best model
# best_loss, best_acc = evaluate(ds, best_model, best_model_config)
# print(f"Best Model: Loss = {best_loss}, Accuracy = {best_acc}")

# Creating empty result dataframe
results = pd.DataFrame(columns=["name", "normal_acc", "normal_acc_attack", "soup_acc", "soup_acc_attack"])

# Create model_soup
# Average the weights for the last 5 epochs
for i, path in enumerate(model_paths):

    # Get config and load model
    config_path_temp = os.path.join(ROOT, old_result_path, df[df.name == path].iloc[0, 1])
    config_temp = json.load(open(os.path.join(config_path_temp, "params.json"), ))

    # Get the paths for the last five checkpoints
    checkpoints = []
    for p in os.listdir(os.path.join(old_result_path, path)):
        if p.startswith("checkpoint"):
            checkpoints.append(p)
        checkpoints.sort()
    # Only take the last 5 checkpoints
    checkpoints = checkpoints[-5:]

    for j, epoch in enumerate(checkpoints):
        
        aux_path = os.path.join(old_result_path, path, epoch, "checkpoints")
        state_dict = torch.load(aux_path)

        if j == 0:
            uniform_soup = {k: v * (1./(len(checkpoints))) for k, v in state_dict.items()}

        else: 
            uniform_soup = {k: v * (1./(len(checkpoints))) + uniform_soup[k] for k, v in state_dict.items()}

     # Define model and load in state
    model = ConvNetSmall(
        channels_in=config_temp["model::channels_in"],
        nlin=config_temp["model::nlin"],
        dropout=config_temp["model::dropout"],
        init_type=config_temp["model::init_type"]
    )
    try:
        model.load_state_dict(uniform_soup)
    except RuntimeError:
        model = ConvNetSmall(
            channels_in=config_temp["model::channels_in"],
            nlin=config_temp["model::nlin"],
            dropout=0,
            init_type=config_temp["model::init_type"]
        )
        model.load_state_dict(uniform_soup)
        try:
           model.load_state_dict(uniform_soup)
        except RuntimeError:
            model = ConvNetSmall(
                channels_in=config_temp["model::channels_in"],
                nlin=config_temp["model::nlin"],
                dropout=0.5,
                init_type=config_temp["model::init_type"]
            )
            model.load_state_dict(uniform_soup)
    model.to(device)

    # Evaluate Model Soup
    soup_loss, soup_acc = evaluate(ds, model, config_temp)

    # Getting perturbed dataset
    perturbed_data_path = "/netscratch2/jlautz/model_robustness/src/model_robustness/data/SVHN/PGD/hyp-10-f/eps_0.1/perturbed_dataset.pt"
    perturbed_data = torch.load(perturbed_data_path)
    soup_loss_attack, soup_acc_attack = evaluate(perturbed_data, model, config_temp)

    # Getting old acc and loss from df
    aux_df = df[df.name==path]
    normal_acc = aux_df["old_acc"].item()
    normal_acc_attack = aux_df["new_acc"].item()

    results.loc[i, "name"] = path#
    results.loc[i, "normal_acc"] = normal_acc
    results.loc[i, "normal_acc_attack"] = normal_acc_attack
    results.loc[i, "soup_acc"] = soup_acc
    results.loc[i, "soup_acc_attack"] = soup_acc_attack

    if i % 50 == 0:
        print(f"Model {i+1}/{len(model_paths)} done.")

    # print(f"MODEL SOUP {i+1}/{len(model_paths)}: New Accuracy = {soup_acc} ({old_acc}); \n After Attack Accuracy = {perturbed_acc}({old_attack_acc})")

results.to_csv("/netscratch2/jlautz/model_robustness/src/model_robustness/data/soups/SVHN_hyp_10_f_results.csv")

