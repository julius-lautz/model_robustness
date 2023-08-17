from pathlib import Path

import torch

from torch.utils.data import Dataset


from .dataset_auxiliaries import (
    test_checkpoint_for_nan,
    test_checkpoint_with_threshold,
)

import random
import copy
import json
import tqdm
from collections import OrderedDict
import numpy as np


import ray
from .progress_bar import ProgressBar

import logging


CLEAN_MODEL_PROPERTY_PADDING_KEYS = [
    "final_triggered_val_acc",
    "final_triggered_val_loss",
    "final_triggered_data_test_acc",
]


class ModelDatasetBase(Dataset):
    """
    This dataset class loads checkpoints from path, stores them in the memory
    It considers different task, but does not implement the __getitem__ function or any augmentation
    """

    # init
    def __init__(
        self,
        root,
        train_val_test="train",  # determines whcih dataset split to use
        ds_split=[0.7, 0.3],  #
        max_samples=None,
        weight_threshold=float("inf"),
        filter_function=None,  # gets sample path as argument and returns True if model needs to be filtered out
        property_keys=None,
        num_threads=4,
        shuffle_path=True,
        verbosity=0,
    ):
        self.verbosity = verbosity
        self.weight_threshold = weight_threshold
        self.property_keys = copy.deepcopy(property_keys)
        self.train_val_test = train_val_test
        self.ds_split = ds_split

        ### prepare directories and path list ################################################################

        ## check if root is list. if not, make root a list
        if not isinstance(root, list):
            root = [root]

        ## make path an absolute pathlib Path
        for rdx in root:
            if isinstance(rdx, str):
                rdx = Path(rdx)
        self.root = root

        # get list of folders in directory
        self.path_list = []
        for rdx in self.root:
            pth_lst_tmp = [f for f in rdx.iterdir() if f.is_dir()]
            self.path_list.extend(pth_lst_tmp)

        # shuffle self.path_list
        if shuffle_path:
            random.seed(42)
            random.shuffle(self.path_list)

        ### Split Train and Test set ###########################################################################
        if max_samples is not None:
            self.path_list = self.path_list[:max_samples]

        ### get reference model ###########################################################################
        # iterate over path list
        for pdx in self.path_list:
            # try to load model at last epoch
            # first successsful load becomes reference checkpoint
            ref_check, ref_lab, ref_path = load_checkpoint(
                path=pdx,  # path to model
                weight_threshold=self.weight_threshold,
                filter_function=filter_function,
            )
            if ref_check is not None:
                break

        self.reference_checkpoint = copy.deepcopy(ref_check)
        self.reference_label = copy.deepcopy(ref_lab)
        self.reference_path = copy.deepcopy(ref_path)

        assert self.reference_checkpoint is not None, "no reference checkpoint found"
        logging.info(f"reference checkpoint found at {self.reference_path}")

        ### Split Train and Test set ###########################################################################
        assert sum(self.ds_split) == 1.0, "dataset splits do not equal to 1"
        # two splits
        if len(self.ds_split) == 2:
            if self.train_val_test == "train":
                idx1 = int(self.ds_split[0] * len(self.path_list))
                self.path_list = self.path_list[:idx1]
            elif self.train_val_test == "test":
                idx1 = int(self.ds_split[0] * len(self.path_list))
                self.path_list = self.path_list[idx1:]
            else:
                logging.error(
                    "validation split requested, but only two splits provided."
                )
                raise NotImplementedError(
                    "validation split requested, but only two splits provided."
                )
        # three splits
        elif len(self.ds_split) == 3:
            if self.train_val_test == "train":
                idx1 = int(self.ds_split[0] * len(self.path_list))
                self.path_list = self.path_list[:idx1]
            elif self.train_val_test == "val":
                idx1 = int(self.ds_split[0] * len(self.path_list))
                idx2 = idx1 + int(self.ds_split[1] * len(self.path_list))
                self.path_list = self.path_list[idx1:idx2]
            elif self.train_val_test == "test":
                idx1 = int(self.ds_split[0] * len(self.path_list))
                idx2 = idx1 + int(self.ds_split[1] * len(self.path_list))
                self.path_list = self.path_list[idx2:]
        else:
            logging.warning(f"dataset splits are unintelligble. Load 100% of dataset")
            pass

        ### prepare data lists ###############
        data = []
        labels = []
        paths = []

        ## init multiprocessing environment ############
        ray.init(num_cpus=num_threads)

        ### gather data #############################################################################################
        logging.info(f"loading checkpoints from {self.root}")
        pb = ProgressBar(total=len(self.path_list))
        pb_actor = pb.actor
        for idx, path in enumerate(self.path_list):
            # # call function in parallel
            (
                ddx,
                ldx,
                path_dx,
            ) = load_checkpoints_remote.remote(
                path=path,
                weight_threshold=self.weight_threshold,
                filter_function=filter_function,
                pba=pb_actor,
            )
            # append returns to lists
            data.append(ddx)
            labels.append(ldx)
            paths.append(path_dx)

        pb.print_until_done()

        # collect actual data
        data = ray.get(data)
        labels = ray.get(labels)
        paths = ray.get(paths)

        ray.shutdown()

        # remove None values
        data = [ddx for ddx in data if ddx]
        labels = [ddx for ddx in labels if ddx]
        paths = [pdx for pdx in paths if pdx]

        self.data = copy.deepcopy(data)
        self.labels = copy.deepcopy(labels)
        self.paths = copy.deepcopy(paths)

        logging.info(
            f"Data loaded. found {len(self.data)} usable samples out of potential {len(self.path_list)} samples."
        )

        if self.property_keys is not None:
            logging.info(f"Load properties for samples from paths.")

            # get propertys from path
            result_keys = self.property_keys.get("result_keys", [])
            config_keys = self.property_keys.get("config_keys", [])
            # figure out offset
            try:
                self.read_properties(
                    results_key_list=result_keys,
                    config_key_list=config_keys,
                    idx_offset=0,
                )
            except AssertionError as e:
                logging.error(f"Exception occurred: {e}", exc_info=True)
                self.read_properties(
                    results_key_list=result_keys,
                    config_key_list=config_keys,
                    idx_offset=1,
                )
            logging.info(f"Properties loaded.")
        elif "tdc_datasets" in str(self.root[0]):
            logging.info(
                f"infer neurips trojai competition-type from path. attempting to load data."
            )
            self.read_properties_neurips_trojai()
        elif "icsi_resnet" in str(self.root[0]):
            logging.info(
                f"infer neurips trojai competition-type from path. attempting to load data."
            )
            self.read_properties_icsi_resnet()
        else:
            self.properties = None

    ## getitem ####################################################################################################################################################################
    def __getitem__(self, index):
        # not implemented in base class
        raise NotImplementedError(
            "the __getitem__ function is not implemented in the base class. "
        )
        pass

    ## len ####################################################################################################################################################################
    def __len__(self):
        return len(self.data)

    ## read properties from path ##############################################################################################################################################
    def read_properties_icsi_resnet(
        self,
    ):
        """
        reads model info from info.json as per neurips trojai competition dataset format
        """

        properties = {
            # "id": [],
            # "model": [],
            # "dataset": [],
            # "trigger_id": [],
            "trigger_ratio": [],
            # "trigger_stealness": [],
            # "batchsize": [],
            # "lr": [],
            "acc_train": [],
            "acc_test": [],
            "asr": [],
            # "attack_start": [],
            "epoch": [],
            # "source_class": [],
            # "target_class": [],
            # "num_trigger": [],
            "poisoned": [],
        }
        for iidx, ppdx in tqdm.tqdm(enumerate(self.paths), desc="read model info.json"):
            fname = ppdx.joinpath("info.json")
            with fname.open("r") as f:
                info = json.load(f)
            # for key in properties.keys():
            #     properties[key].append(info[key])
            properties["trigger_ratio"].append(float(info["trigger_ratio"]))
            properties["acc_train"].append(float(info["acc_train"]))
            properties["acc_test"].append(float(info["acc_test"]))
            properties["asr"].append(float(info["asr"]))
            properties["epoch"].append(int(info["epoch"]))
            properties["poisoned"].append(info["poisoned"])
        self.properties = properties

    ## read properties from path ##############################################################################################################################################
    def read_properties_neurips_trojai(
        self,
    ):
        """
        reads model info from info.json as per neurips trojai competition dataset format
        """
        properties = {
            "test_acc": [],
            "poisoned": [],
            "trigger_type": [],
            "attack_success_rate": [],
        }
        for iidx, ppdx in tqdm.tqdm(enumerate(self.paths), desc="read model info.json"):
            fname = ppdx.joinpath("info.json")
            with fname.open("r") as f:
                info = json.load(f)
            properties["test_acc"].append(info["test_accuracy"])
            properties["poisoned"].append(
                "poisoned" if info.get("trigger_type", None) is not None else "clean"
            )
            properties["attack_success_rate"].append(info.get("attack_success_rate", 0))
            properties["trigger_type"].append(info.get("trigger_type", "none"))

        self.properties = properties

    def read_properties(self, results_key_list, config_key_list, idx_offset=1):
        # copy results_key_list to prevent kickback of delete to upstream function
        results_key_list = [key for key in results_key_list]
        # init dict
        properties = {}
        for key in results_key_list:
            properties[key] = []
        for key in config_key_list:
            properties[key] = []
        # remove ggap from results_key_list -> cannot be read, has to be computed.
        read_ggap = False
        if "ggap" in results_key_list:
            results_key_list.remove("ggap")
            read_ggap = True

        logging.info(f"### load data for {properties.keys()}")

        # iterate over samples
        for iidx, ppdx in tqdm.tqdm(enumerate(self.paths)):
            res_tmp = read_properties_from_path(ppdx)
            for key in results_key_list:
                properties[key].append(res_tmp[key])
            for key in config_key_list:
                properties[key].append(res_tmp["config"][key])
            # # compute ggap TODO: there is no training accuracy only validation/test acc
            # if read_ggap:
            #     gap = res_tmp["train_acc"] - res_tmp["test_acc"]
            #     properties["ggap"].append(gap)
            # assert epoch == training_iteration -> match correct data
            if iidx == 123:
                logging.debug(f"check existance of keys")
                for key in properties.keys():
                    logging.debug(
                        f"key: {key} - len {len(properties[key])} - last entry: {properties[key][-1]}"
                    )

        # change the poisoned accuracy/loss values for clean models to mean value
        for padding_k in CLEAN_MODEL_PROPERTY_PADDING_KEYS:
            vals_with_padding = properties[padding_k]
            avg_val = np.nanmean(vals_with_padding)
            properties[padding_k] = np.nan_to_num(
                vals_with_padding, nan=avg_val
            ).tolist()

        self.properties = properties


## helper function for property reading
def read_properties_from_path(path):
    """
    #TODO Jialin: currently contain all model architectures
    set model_architecture_level = 5 to include only mobile_net_v2
    reads path/result.json
    returns the dict
    """
    # read json
    try:
        res_tmp = {}
        config_file = json.load(open(str(Path(path).joinpath("config.json")), "r"))
        results_file = json.load(open(str(Path(path).joinpath("stats.json")), "r"))

        res_tmp["config"] = {}
        for config_k, config_v in config_file.items():
            # include only number types
            if isinstance(config_v, (float, int, bool)):
                res_tmp["config"][config_k] = config_v

        for result_k, result_v in results_file.items():
            # there are repetitive keys from config file in the result file
            if result_k in res_tmp["config"]:
                continue
            if isinstance(result_v, (float, int)):
                res_tmp[result_k] = result_v

        if res_tmp["config"]["POISONED"] == False:
            # fix a mistake in the config file, clean models doesn't have triggers
            res_tmp["config"]["NUMBER_TRIGGERED_CLASSES"] = 0
            # add poisoned accuracy/loss to results for clean models for consisitency in the property file
            for padding_k in CLEAN_MODEL_PROPERTY_PADDING_KEYS:
                res_tmp[padding_k] = np.nan

    except Exception as e:
        logging.error(
            f"error loading files from model directory {path} - {e}", exc_info=True
        )
    # pick results
    return res_tmp


############## load_checkpoint_remote ########################################################
@ray.remote(num_returns=3)
def load_checkpoints_remote(
    path,  # path to model
    weight_threshold,
    filter_function,
    pba,
):
    ## get full path to files ################################################################
    # TODO JIALIN - use correct checkpoint name
    # chkpth = path.joinpath(f"model.pt")
    chkpth = path.joinpath("checkpoint.pt")
    ## load checkpoint #######################################################################
    chkpoint = {}
    try:
        # load chkpoint to cpu memory
        device = torch.device("cpu")
        chkpoint = torch.load(str(chkpth), map_location=device)

    except Exception as e:
        logging.debug(f"error while loading {chkpth}: {e}")
        # instead of appending empty stuff, jump to next
        pba.update.remote(1)
        return None, None, None
    ## create label ##########################################################################
    label = f"{path}"

    ### check for NAN values #################################################################
    nan_flag = test_checkpoint_for_nan(copy.deepcopy(chkpoint))
    if nan_flag == True:
        # jump to next sample
        logging.warning(f"found nan values in checkpoint {label}")
        print(f"found nan values in checkpoint {label}")
        pba.update.remote(1)
        return None, None, None

    #### apply filter function #################################################################
    if filter_function is not None:
        filter_flag = filter_function(path)
        if filter_flag == True:  # model needs to be filtered
            logging.warning(f"filtered out checkpoint {label} via filter function")
            print(f"filtered out checkpoint {label} via filter function")
            pba.update.remote(1)
            return None, None, None

    #### apply threhold #################################################################
    thresh_flag = test_checkpoint_with_threshold(
        copy.deepcopy(chkpoint), threshold=weight_threshold
    )
    if thresh_flag == True:
        # jump to next sample
        logging.warning(f"found values above threshold in checkpoint {label}")
        print(f"found values above threshold in checkpoint {label}")
        pba.update.remote(1)
        return None, None, None

    ### clean data #################################################################
    else:  # use data
        ddx = copy.deepcopy(chkpoint)
        ldx = copy.deepcopy(label)

    # return
    pba.update.remote(1)
    return ddx, ldx, path

    ############## load_checkpoint ########################################################


############## load_checkpoint_remote ########################################################
def load_checkpoint(
    path,  # path to model
    weight_threshold,
    filter_function,
):
    ## get full path to files ################################################################
    # TODO JIALIN - use correct checkpoint name
    # chkpth = path.joinpath(f"model.pt")
    chkpth = path.joinpath("checkpoint.pt")
    ## load checkpoint #######################################################################
    chkpoint = {}
    try:
        # load chkpoint to cpu memory
        device = torch.device("cpu")
        chkpoint = torch.load(str(chkpth), map_location=device)

    except Exception as e:
        logging.debug(f"error while loading {chkpth}: {e}")
        # instead of appending empty stuff, jump to next

        return None, None, None
    ## create label ##########################################################################
    label = f"{path}"

    ### check for NAN values #################################################################
    nan_flag = test_checkpoint_for_nan(copy.deepcopy(chkpoint))
    if nan_flag == True:
        # jump to next sample
        logging.warning(f"found nan values in checkpoint {label}")
        print(f"found nan values in checkpoint {label}")

        return None, None, None

    #### apply filter function #################################################################
    if filter_function is not None:
        filter_flag = filter_function(path)
        if filter_flag == True:  # model needs to be filtered
            logging.warning(f"filtered out checkpoint {label} via filter function")
            print(f"filtered out checkpoint {label} via filter function")

            return None, None, None

    #### apply threhold #################################################################
    thresh_flag = test_checkpoint_with_threshold(
        copy.deepcopy(chkpoint), threshold=weight_threshold
    )
    if thresh_flag == True:
        # jump to next sample
        logging.warning(f"found values above threshold in checkpoint {label}")
        print(f"found values above threshold in checkpoint {label}")

        return None, None, None

    ### clean data #################################################################
    else:  # use data
        ddx = copy.deepcopy(chkpoint)
        ldx = copy.deepcopy(label)

    # return
    return ddx, ldx, path

    ############## load_checkpoint ########################################################
