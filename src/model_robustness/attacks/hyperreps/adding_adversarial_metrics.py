import sys 
import os 
import random 
import json
from pathlib import Path
import numpy as np 
import pandas as pd

import torch
from shrp.datasets.dataset_tokens import DatasetTokens
from shrp.git_re_basin.git_re_basin import (
    zoo_cnn_large_permutation_spec,
    zoo_cnn_permutation_spec
)

from shrp.models.def_AE_module import AEModule
from shrp.models.def_downstream_module import DownstreamTaskLearner

def return_names_for_path(dataset, setup):
    if setup=="hyp-10-r":
        abbr = "random"
    elif setup=="hyp-10-f":
        abbr = "fixed"

    if dataset == "CIFAR10":
        part1 = os.path.join(dataset, "large")

        if setup == "seed":
            part2 = f"tune_zoo_{dataset.lower()}_uniform_large"
        else:
            part2 = f"tune_zoo_{dataset.lower()}_large_hyperparameter_10_{abbr}_seeds"
    
    else:
        part1 = dataset

        if setup == "seed":
            part2 = f"tune_zoo_{dataset.lower()}_uniform"
        else:
            part2 = f"tune_zoo_{dataset.lower()}_hyperparameter_10_{abbr}_seeds"
    
    return part1, part2




# Define paths
zoo_root = Path("/ds2/model_zoos/zoos_v2/")
result_path = Path("/netscratch2/jlautz/model_robustness/src/model_robustness/data/all_results.csv")
hyper_root = Path("/netscratch2/kschuerholt/code/shrp/experiments/02_representation_learning/03_kde_sampling/tune")
full_df = pd.read_csv(result_path)

# Define params to loop over
datasets = ["MNIST", "CIFAR10", "SVHN"]
setups = ["seed", "hyp-10-f", "hyp-10-r"]

iter = 0

column_list = ['test_acc_train', 'test_acc_test', 'test_acc_val', 'test_loss_train', 'test_loss_test', 'test_loss_val', 'attack_acc_train',
       'attack_acc_test', 'attack_acc_val', 'attack_loss_train', 'attack_loss_test', 'attack_loss_val', 'ds', 'setup']
results_df = pd.DataFrame(columns=column_list)

# Big loop
for ds in datasets:
    for setup in setups:
        print(f"Getting results for: dataset={ds} setup={setup}")

        dataset_path1, dataset_path2 = return_names_for_path(ds, setup)
        zoo_path = zoo_root.joinpath(dataset_path1, dataset_path2)

        # Define params for loading dataset
        result_key_list = ["test_acc", "test_loss"]
        config_key_list = []
        property_keys = {
            "result_keys": result_key_list,
            "config_keys": config_key_list
        }
        tokensize=0
        if ds == "CIFAR10":
            permutation_spec = zoo_cnn_large_permutation_spec
        else:
            permutation_spec = zoo_cnn_permutation_spec

        # Defining dataset config
        dataset_config = {
            'epoch_lst' : 50,
            'mode' : "vector",
            'permutation_spec' : permutation_spec,
            'map_to_canonical' : False,
            'standardize' : "l2_ind",
            'ds_split' : [0.7, 0.15, 0.15],  
            'max_samples' : None,
            'weight_threshold' : 15000,
            'precision' : "32",
            'filter_function' : None,  # gets sample path as argument and returns True if model needs to be filtered out
            'num_threads' : 8,
            'shuffle_path' : True,
            'verbosity' : 3,
            'getitem' : "tokens+props",
            'ignore_bn' : False,
            'tokensize' : tokensize
        }

        # Load in dataset
        trainset = DatasetTokens(root=[zoo_path.absolute()], train_val_test='train',  property_keys=property_keys, **dataset_config)
        testset = DatasetTokens(root=[zoo_path.absolute()], train_val_test='test',  property_keys=property_keys, **dataset_config)
        valset = DatasetTokens(root=[zoo_path.absolute()], train_val_test='val', property_keys=property_keys, **dataset_config)

        aux_df = full_df[(full_df.dataset==ds) & (full_df.setup==setup)]

        # instantiate empty list to read in results from attacks
        train_attack = [0 for i in range(len(trainset.properties["test_acc"]))]
        train_loss = [0 for i in range(len(trainset.properties["test_loss"]))]
        test_attack = [0 for i in range(len(testset.properties["test_acc"]))]
        test_loss = [0 for i in range(len(testset.properties["test_loss"]))]        
        val_attack = [0 for i in range(len(valset.properties["test_acc"]))]
        val_loss = [0 for i in range(len(valset.properties["test_loss"]))]


        for i, elem in enumerate(trainset.properties["test_acc"]):

            path = str(trainset.get_paths(i)[0]).split("/")[-1]
            
            att = aux_df[aux_df.name == str(trainset.get_paths(i)[0]).split("/")[-1]].iloc[0,17]
            train_attack[i] = [att]
            
            lo = aux_df[aux_df.name == str(trainset.get_paths(i)[0]).split("/")[-1]].iloc[0,16]
            train_loss[i] = [lo]


        for i, elem in enumerate(testset.properties["test_acc"]):

            path = str(testset.get_paths(i)[0]).split("/")[-1]
    
            att = aux_df[aux_df.name == str(testset.get_paths(i)[0]).split("/")[-1]].iloc[0,17]
            test_attack[i] = [att]
            
            lo = aux_df[aux_df.name == str(testset.get_paths(i)[0]).split("/")[-1]].iloc[0,16]
            test_loss[i] = [lo]
        

        for i, elem in enumerate(valset.properties["test_acc"]):

            path = str(valset.get_paths(i)[0]).split("/")[-1]
    
            att = aux_df[aux_df.name == str(valset.get_paths(i)[0]).split("/")[-1]].iloc[0,17]
            val_attack[i] = [att]

            lo = aux_df[aux_df.name == str(valset.get_paths(i)[0]).split("/")[-1]].iloc[0,16]
            val_loss[i] = [lo]

        trainset.properties["attack_acc"] = train_attack
        testset.properties["attack_acc"] = test_attack
        valset.properties["attack_acc"] = val_attack

        trainset.properties["attack_loss"] = train_loss
        testset.properties["attack_loss"] = test_loss
        valset.properties["attack_loss"] = val_loss

        if ds == "MNIST":
            hyper_path = hyper_root.joinpath("mnist_kde_sampling_ep21-25_v7/AE_trainable_879a1_00000_0_2023-08-15_13-56-27")
        elif ds == "CIFAR10":
            hyper_path = hyper_root.joinpath("cifar_kde_sampling_ep21-25_v7/AE_trainable_74edd_00000_0_2023-08-08_15-30-08")
        elif ds == "SVHN":
            hyper_path = hyper_root.joinpath("svhn_kde_sampling_ep21-25_v72/AE_trainable_5e9d8_00000_0_2023-08-17_11-29-46")

        config = json.load(hyper_path.joinpath("params.json").open("r"))
        config["device"] = "cpu"
        config["training::steps_per_epoch"] = 123
        module = AEModule(config)

        checkpoint = torch.load(
            hyper_path.joinpath("checkpoint_000100/state.pt"), map_location=config["device"]
        )
        module.model.load_state_dict(checkpoint["model"])

        dstk = DownstreamTaskLearner()

        performance = dstk.eval_dstasks(
            model=module,
            trainset=trainset,
            testset=testset,
            valset=valset,
            task_keys=['test_acc', 'test_loss', 'attack_acc', 'attack_loss'],
            batch_size=config["trainset::batchsize"],
        )

        # Add results to dataframe
        results_df.loc[iter, :] = performance
        results_df.loc[iter, "ds"] = ds
        results_df.loc[iter, "setup"] = setup

        iter += 1

results_df.to_csv("hyper_results.csv")

                
