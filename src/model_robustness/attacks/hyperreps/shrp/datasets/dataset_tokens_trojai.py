import torch
from torch.utils.data import Dataset

from pathlib import Path
import random
import copy

import itertools
from math import factorial

from shrp.datasets.dataset_base_trojai import ModelDatasetBase
from shrp.git_re_basin.git_re_basin import (
    PermutationSpec,
    zoo_cnn_permutation_spec,
    weight_matching,
    apply_permutation,
)

from .dataset_auxiliaries import (
    # tokens_to_checkpoint,
    tokenize_checkpoint,
)


import logging

from typing import List, Union

from shrp.datasets.random_erasing import RandomErasingVector

import ray
from .progress_bar import ProgressBar

import tqdm


#####################################################################
# Define Dataset class
#####################################################################
class DatasetTokens(ModelDatasetBase):
    """
    This class inherits from the base ModelDatasetBaseEpochs class.
    It extends it by permutations of the dataset in the init function.
    """

    # init
    def __init__(
        self,
        root,
        mode="vector",  # "vector", "checkpoint", "tokenize"
        permutation_spec: PermutationSpec = zoo_cnn_permutation_spec,
        map_to_canonical: bool = False,
        standardize: bool = True,  # wether or not to standardize the data
        tokensize: int = 0,
        train_val_test="train",
        ds_split=[0.7, 0.3],
        weight_threshold: float = float("inf"),
        max_samples: int = 0,  # limit the number of models to integer number (full model trajectory, all epochs)
        filter_function=None,  # gets sample path as argument and returns True if model needs to be filtered out
        property_keys=None,
        shuffle_path: bool = True,
        num_threads=4,
        verbosity=0,
        precision="32",
        supersample: int = 1,  # if supersample > 1, the dataset will be supersampled by this factor
        getitem: str = "tokens",  # "tokens", "tokens+props"
        ignore_bn: bool = False,
        reference_model_path: Union[Path, str] = None,
    ):
        # call init of base class
        super().__init__(
            root=root,
            train_val_test=train_val_test,
            ds_split=ds_split,
            weight_threshold=weight_threshold,
            max_samples=max_samples,
            filter_function=filter_function,
            property_keys=property_keys,
            num_threads=num_threads,
            verbosity=verbosity,
            shuffle_path=shuffle_path,
        )
        self.mode = mode
        self.permutation_spec = permutation_spec
        self.standardize = standardize
        self.tokensize = tokensize
        self.precision = precision
        self.supersample = supersample
        assert self.supersample >= 1

        self.num_threads = num_threads

        self.map_to_canonical = map_to_canonical

        self.ignore_bn = ignore_bn

        self.getitem = getitem

        ### prepare canonical form ###########################################################################################
        if self.map_to_canonical:
            logging.info("prepare canonical form")
            self.map_models_to_canonical()

        ### init len ###########################################################################################
        logging.info("init dataset length")
        self.init_len()

        if reference_model_path:
            logging.info(f"load reference model from {reference_model_path}")
            self.reference_checkpoint = torch.load(reference_model_path)

        # ### standardize ###################################################################################################################################################################
        if self.mode == "tokenize":
            if self.standardize == True:
                logging.info("standardize data")
                self.standardize_data_checkpoints()
            elif self.standardize == "minmax":
                logging.info("min max normalization of data")
                self.normalize_data_checkpoints()
            elif self.standardize == "l2_ind":
                self.normalize_checkpoints_separately()
            # TODO: Jialin maybe check potential per-model standardization / normalization

        ### tokenize
        if self.mode == "tokenize":
            logging.info("tokenize data")
            self.tokenize_data()

        ### set transforms ########################
        logging.info("set transforms")
        self.transforms = None

        # set precision
        if self.precision != "32":
            logging.info("set precision")
            if not self.standardize:
                logging.warning(
                    "using lower precision for non-standardized data may cause loss of information"
                )
            self.set_precision(self.precision)

    def tokenize_data(self):
        """
        cast samples as list of tokens to tensors to speed up processing
        """
        # iterate over all samlpes
        self.weights = []
        self.pos = []
        self.masks = []
        max_len = 0
        for idx in tqdm.tqdm(range(len(self.data))):
            ddx, mask, pos = tokenize_checkpoint(
                checkpoint=self.data[idx],
                tokensize=self.tokensize,
                return_mask=True,
                ignore_bn=self.ignore_bn,
            )
            self.weights.append(ddx)
            # keep only largest for convenience.. cast to bool to save space
            self.masks.append(mask.to(torch.bool))
            self.pos.append(pos.to(torch.int))
            if pos.shape[0] > max_len:
                max_len = pos.shape[0]
        # zero pad weights
        for idx, ddx in enumerate(self.weights):
            if ddx.shape[0] < max_len:
                # zero pad
                w_tmp = torch.zeros(max_len, ddx.shape[1])
                m_tmp = torch.zeros(max_len, ddx.shape[1])
                p_tmp = torch.zeros(max_len, self.pos[0].shape[1])
                # copy data to padded tensor
                w_tmp[: ddx.shape[0], :] = ddx
                w_tmp[: ddx.shape[0], :] = self.masks[idx]
                p_tmp[: ddx.shape[0], :] = self.pos[idx]
                # replace data
                self.masks[idx] = m_tmp
                self.weights[idx] = w_tmp
                self.pos[idx] = p_tmp

    def set_precision(self, precision: str = "32"):
        """ """
        self.precision = precision
        if self.precision == "16":
            dtype = torch.float16
        elif self.precision == "b16":
            dtype = torch.bfloat16
        elif self.precision == "32":
            dtype = torch.float32
        elif self.precision == "64":
            dtype = torch.float64
        else:
            raise NotImplementedError(
                f"precision {self.precision} is not implemented. use 32 or 64"
            )
        if self.mode == "tokenize" or self.mode == "checkpoint":
            self.data = [
                self.data[idx][key].to(dtype)
                for key in self.data[idx].keys()
                for idx in range(len(self.data))
            ]

    ## get_weights ####################################################################################################################################################################
    def __get_weights__(
        self,
    ):
        """
        Returns:
            torch.Tensor with full dataset as sequence of components [n_samples,n_tokens_per_sample,token_dim]
        """
        if self.mode == "vector":
            data_out = [self.data[idx] for idx in range(len(self.data))]
            mask_out = [self.masks[idx] for idx in range(len(self.data))]
            data_out = torch.stack(data_out)
            mask_out = torch.stack(mask_out)
            logging.debug(f"shape of weight tensor: {data_out.shape}")
            return data_out, mask_out
        elif self.mode == "tokenize":
            # stack weights
            data_out = torch.stack(self.weights)
            mask_out = torch.stack(self.masks)
            logging.debug(f"shape of weight tensor: {data_out.shape}")
            return data_out, mask_out

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample to be retrieved
        Returns:
            ddx: torch.Tensor of neuron tokens with shape [n_tokens_per_sample/windowsize,token_dim]
            mask: torch.Tensor of the same shape as ddx_idx indicating the nonzero elements
            pos: positions [layer,token_in_layer] of sample
            props: properties of sample
        """
        # remove supersample (over index, if exists)
        idx = index % self._len

        # get raw data, assume view 1 and view 2 are the same
        chdx = self.data[idx]  # we assume we always get checkpoints here.

        ### set getitem to base function ########################
        if self.getitem == "tokens":
            if not self.transforms:
                return chdx
            else:
                return self.transforms(chdx)
        elif self.getitem == "tokens+props":
            # get properties
            props = []
            for key in self.properties.keys():
                props.append(self.properties[key][idx])
            props = torch.tensor(props)

            if not self.transforms:
                return chdx, props
            else:
                return self.transforms(chdx, props)
        else:
            raise NotImplementedError(f"getitem {self.getitem} is not implemented")

    ### len ##################################################################################################################################################################
    def init_len(self):
        """
        helper function that sets the virtual lenght of the dataset, in case that's more complicated than len(self.data)
        """
        self._len = len(self.data)

    def __len__(self):
        if self.supersample:
            # supersampling extends the len of the dataset by the sumpersample factor
            # the dataset will be iterated over several times.
            # this only makes sense if transformations are used
            # that way, it extends one epoch and reduces dataloading / synchronization epoch
            # motivation: if samples are sliced to small subsets, using the same sample multiple times may reduce overhead
            return self._len * self.supersample
        return self._len

    ### standardize data #########################################################################################################################################################
    def standardize_data(self):
        """
        standardize data to zero-mean / unit std. per layer
        store per-layer mean / std
        """
        logging.info("Get layer mapping")
        # step 1: get token-layer index relation
        layers = {}
        # get layers
        layer_list = torch.unique(
            self.positions[0][:, 1]
        )  # assuming that the number of layers is always the same
        # iterate over layers and get statistics
        logging.info("Get layer-wise mean and std")
        for ldx in layer_list:
            # collect all tokens within the layer for all models
            means = []
            stds = []
            # iterate over models
            for idx in range(len(self.data)):
                # get index for current layer (returned as tuple with 1 entry)
                index_l = torch.where(self.positions[idx][:, 1] == ldx)[0]
                # get mask for this layer
                mdx = torch.index_select(input=self.masks[idx], index=index_l, dim=0)
                # slice token sample with torch.index_select()
                tmp2 = torch.index_select(input=self.data[idx], index=index_l, dim=0)
                # cast to masked tensor
                tmp2 = torch.masked.masked_tensor(tmp2, mdx)
                # compute masked mean / std
                tmp_mean = tmp2.mean()
                tmp_std = tmp2.std()
                # append values to lists
                means.append(tmp_mean.item())
                stds.append(tmp_std.item())

            # compute mean / std assuming all samples have the same numels
            mu = torch.mean(torch.tensor(means))
            sigma = torch.sqrt(torch.mean(torch.tensor(stds) ** 2))

            # store in layer
            layer = str(ldx)
            layers[layer] = {}
            layers[layer]["mean"] = mu
            layers[layer]["std"] = sigma

        self.layers = layers

        logging.info("Apply standardization")
        # standardize:
        # # iterate over models
        for idx in tqdm.tqdm(range(len(self.data))):
            # # iterate over tokens of that layer
            for ldx in layer_list:
                # get index for current layer (returned as tuple with 1 entry)
                index_l = torch.where(self.positions[idx][:, 1] == ldx)[0]
                # get mean / std
                layer = str(ldx)
                mu = layers[layer]["mean"]
                sigma = layers[layer]["std"]
                # apply standardization on slice of layer
                self.data[idx][index_l] = (self.data[idx][index_l] - mu) / sigma

    def standardize_data_checkpoints(self):
        """
        standardize data to zero-mean / unit std. per layer
        store per-layer mean / std
        """
        logging.info("Get layer mapping")
        # step 1: get token-layer index relation
        layers = {}
        # iterate over layers
        chkpt = self.reference_checkpoint
        for key in tqdm.tqdm(chkpt.keys()):
            if "weight" in key:
                #### get weights ####
                if self.ignore_bn and ("bn" in key or "downsample.1" in key):
                    continue

                # first iteration over data: get means / std
                means = []
                stds = []
                for idx in range(len(self.data)):
                    w = self.data[idx][key]
                    # flatten to out_channels x n
                    w = w.view(w.shape[0], -1)
                    # cat biases to channels if they exist in self.data[idx]
                    if key.replace("weight", "bias") in self.data[idx]:
                        b = self.data[idx][key.replace("weight", "bias")]
                        w = torch.cat([w, b.unsqueeze(dim=1)], dim=1)
                    # compute masked mean / std
                    tmp_mean = w.mean()
                    tmp_std = w.std()
                    # append values to lists
                    means.append(tmp_mean.item())
                    stds.append(tmp_std.item())

                # aggregate means / stds
                # compute mean / std assuming all samples have the same numels
                mu = torch.mean(torch.tensor(means))
                sigma = torch.sqrt(torch.mean(torch.tensor(stds) ** 2))

                # store in layer
                layers[key] = {}
                layers[key]["mean"] = mu
                layers[key]["std"] = sigma

                # secon iteration over data: standardize
                for idx in range(len(self.data)):
                    # normalize weights
                    self.data[idx][key] = (self.data[idx][key] - mu) / sigma
                    # normalize biases if they exist
                    if key.replace("weight", "bias") in self.data[idx]:
                        self.data[idx][key.replace("weight", "bias")] = (
                            self.data[idx][key.replace("weight", "bias")] - mu
                        ) / sigma

        self.layers = layers

    def normalize_data_checkpoints(self):
        """
        min-max normalization of data to zero-mean / unit std. per layer
        store per-layer mean / std
        """
        logging.info("Get layer mapping")
        # step 1: get token-layer index relation
        layers = {}
        # iterate over layers
        chkpt = self.reference_checkpoint
        for key in tqdm.tqdm(chkpt.keys()):
            if "weight" in key:
                #### get weights ####
                if self.ignore_bn and ("bn" in key or "downsample.1" in key):
                    continue

                # first iteration over data: get means / std
                mins = []
                maxs = []
                for idx in range(len(self.data)):
                    w = self.data[idx][key]
                    # flatten to out_channels x n
                    w = w.view(w.shape[0], -1)
                    # cat biases to channels if they exist in self.data[idx]
                    if key.replace("weight", "bias") in self.data[idx]:
                        b = self.data[idx][key.replace("weight", "bias")]
                        w = torch.cat([w, b.unsqueeze(dim=1)], dim=1)
                    # compute masked mean / std
                    tmp_min = w.flatten().min()
                    tmp_max = w.flatten().max()
                    # append values to lists
                    mins.append(tmp_min.item())
                    maxs.append(tmp_max.item())

                # aggregate means / stds
                # compute mean / std assuming all samples have the same numels
                min_glob = torch.min(torch.tensor(mins))
                max_glob = torch.max(torch.tensor(maxs))

                # store in layer
                layers[key] = {}
                layers[key]["min"] = min_glob
                layers[key]["max"] = max_glob

                # secon iteration over data: min-max normalization
                for idx in range(len(self.data)):
                    # normalize weights
                    self.data[idx][key] = (self.data[idx][key] - min_glob) / (
                        max_glob - min_glob
                    ) * 2 - 1
                    # normalize biases if they exist
                    if key.replace("weight", "bias") in self.data[idx]:
                        self.data[idx][key.replace("weight", "bias")] = (
                            self.data[idx][key.replace("weight", "bias")] - min_glob
                        ) / (max_glob - min_glob) * 2 - 1

        self.layers = layers

    def normalize_checkpoints_separately(self):
        """
        l2-normalization of all checkoints and layers, individually (note: no shared normalization coeff)
        we currently don't keep the normalization coefficients, as we don't plan to reconstruct ever.
        This may need fixing in the future.
        """
        logging.info("apply l2 normalization")
        # iterate over data points
        for idx in tqdm.tqdm(range(len(self.data))):
            # iterate over layers:
            for key in self.data[idx].keys():
                if "weight" in key:
                    #### get weights ####
                    if self.ignore_bn and ("bn" in key or "downsample.1" in key):
                        continue
                    w = self.data[idx][key]
                    # flatten to out_channels x n
                    w = w.view(w.shape[0], -1)
                    # cat biases to channels if they exist in self.data[idx]
                    if key.replace("weight", "bias") in self.data[idx]:
                        b = self.data[idx][key.replace("weight", "bias")]
                        w = torch.cat([w, b.unsqueeze(dim=1)], dim=1)
                    # compute l2 norm of flattened weights
                    l2w = torch.norm(w.flatten(), p=2, dim=0)
                    # normalize weights with l2w
                    self.data[idx][key] = self.data[idx][key] / l2w
                    # normalize biases with l2w
                    if key.replace("weight", "bias") in self.data[idx]:
                        self.data[idx][key.replace("weight", "bias")] = (
                            self.data[idx][key.replace("weight", "bias")] / l2w
                        )

    ### map data to canoncial #############################################################################################
    def map_models_to_canonical(self):
        """
        define reference model
        iterate over all models
        get permutation w.r.t last epoch (best convergence)
        apply same permutation on all epochs (on raw data)
        """
        # use first model / last epoch as reference model (might be sub-optimal)
        reference_model = self.reference_checkpoint

        ## init multiprocessing environment ############
        ray.init(num_cpus=self.num_threads)

        ### gather data #############################################################################################
        print(f"preparing computing canon form...")
        pb = ProgressBar(total=len(self.data))
        pb_actor = pb.actor

        for idx in range(len(self.data)):
            # align models using git-re-basin
            perm_spec = self.permutation_spec
            # get second model
            model_curr = self.data[idx]
            # align using git re-basin
            model_curr = compute_single_canon_form.remote(
                reference_model=reference_model,
                model=model_curr,
                perm_spec=perm_spec,
                pba=pb_actor,
            )
            # put back in self.data
            self.data[idx] = model_curr

        # update progress bar
        pb.print_until_done()

        self.data = [ray.get(self.data[idx]) for idx in range(len(self.data))]
        ray.shutdown()


### helper parallel function #############################################################################################
@ray.remote(num_returns=1)
def compute_single_canon_form(reference_model, model, perm_spec, pba):
    """
    finds git-rebasin style permutation for model, so that it's aligned with reference model
    Args:
        reference_model: checkpoint (ordered_dict) of reference model (to be aligned to)
        model: checkpoint (ordered_dict) of model (to be aligned to the reference model)
        perm_spec: permutation_spec that identifies how both models can be permuted to match each other
        pba: tracks the progress of the calling parallel jobs
    """
    # find permutation to match params_b to params_a
    logging.debug(
        f"compute canonical form: params a {type(reference_model)} params b {type(model)}"
    )
    match_permutation = weight_matching(
        ps=perm_spec, params_a=reference_model, params_b=model
    )
    # apply permutation on all epochs
    model = apply_permutation(ps=perm_spec, perm=match_permutation, params=model)

    # update counter
    pba.update.remote(1)
    # return list
    return model


""" may be deprecated, who knows..
### helper parallel function #############################################################################################
@ray.remote(num_returns=1)
def compute_single_perm(reference_checkpoint, permutation_dict, perm_spec, pba):
    # copy reference checkpoint
    index_check = copy.deepcopy(reference_checkpoint)
    # apply permutation on checkpoint
    index_check_perm = apply_permutation(
        ps=perm_spec, perm=permutation_dict, params=index_check
    )
    # vectorize
    index_perm = vectorize_checkpoint(index_check_perm)
    # update counter
    pba.update.remote(1)
    # return list
    return index_perm
"""
