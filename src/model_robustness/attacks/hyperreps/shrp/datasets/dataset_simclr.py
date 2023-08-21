import torch
from torch.utils.data import Dataset

from pathlib import Path
import random
import copy

import itertools
from math import factorial

from shrp.datasets.dataset_epochs import ModelDatasetBaseEpochs
from shrp.git_re_basin.git_re_basin import (
    PermutationSpec,
    zoo_cnn_permutation_spec,
    weight_matching,
    apply_permutation,
)

import logging

from typing import List

from shrp.datasets.random_erasing import RandomErasingVector

import ray
from .progress_bar import ProgressBar

#####################################################################
# Define Dataset class
#####################################################################
class SimCLRDataset(ModelDatasetBaseEpochs):
    """
    This class inherits from the base ModelDatasetBaseEpochs class.
    It extends it by permutations of the dataset in the init function.
    """

    # init
    def __init__(
        self,
        root,
        epoch_lst=10,
        mode="vector",  # "vector", "checkpoint"
        permutations_number=10,
        permutation_spec: PermutationSpec = zoo_cnn_permutation_spec,
        view_1_canonical: bool = False,
        view_2_canonical: bool = False,
        add_noise_view_1: float = 0.0,  # [('input', 0.15), ('output', 0.013)]
        add_noise_view_2: float = 0.0,  # [('input', 0.15), ('output', 0.013)]
        noise_multiplicative: bool = True,
        erase_augment=None,  # {"p": 0.5,"scale":(0.02,0.33),"value":0,"mode":"block"}
        windowsize: int = 5,
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
    ):
        # call init of base class
        super().__init__(
            root=root,
            epoch_lst=epoch_lst,
            mode="checkpoint",
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
        self.permutations_number = permutations_number
        self.permutation_spec = permutation_spec
        self.standardize = standardize
        self.tokensize = tokensize

        self.add_noise_view_1 = add_noise_view_1
        assert isinstance(self.add_noise_view_1, float)
        self.add_noise_view_2 = add_noise_view_2
        assert isinstance(self.add_noise_view_1, float)
        self.use_multiplicative_noise = noise_multiplicative

        self.num_threads = num_threads

        self.view_1_canonical = view_1_canonical
        self.view_2_canonical = view_2_canonical
        if view_1_canonical and view_2_canonical:
            logging.info(
                f"both view 1 and view 2 are set to canonical. number of permutations is set to 0"
            )
            self.permutations_number = 0

        # set erase augmnet
        self.set_erase(erase_augment)

        if self.view_1_canonical or self.view_2_canonical:
            logging.info("prepare canonical form")
            # TODO: load reference checkpoint in base class, share between train/val/test
            self.map_models_to_canonical()

        ### init len ###########################################################################################
        logging.info("init dataset length")
        self.init_len()

        ### get positions ###########################################################################################
        logging.info("Get Positions")
        reference_checkpoint = self.reference_checkpoint
        self.positions = get_position_mapping_from_checkpoint(reference_checkpoint)

        ### vectorize data ###########################################################################################
        if self.mode == "vector":
            logging.info("vectorize data")
            # keep reference checkpoint
            self.vectorize_data()

        ### initialize permutations ##########################################################################################################################################
        # list of permutations (list of list with indexes)
        if self.permutations_number > 0:
            logging.info("init permutations")
            self.precompute_permutations(
                permutation_number=self.permutations_number,
                perm_spec=self.permutation_spec,
                num_threads=num_threads,
            )

        if self.standardize:
            # TODO: add normalization with l2 norm (ben style)
            self.standardize_data()

        ### set module window ##################################################################################################################################################################
        self.set_module_window(windowsize=windowsize)

        ### set tokensize ###################################################################################################################################################################
        self.set_token_size(tokensize=self.tokensize)

    def set_module_window(self, windowsize: int = 5):
        # check that window is within range 1,no-tokens
        assert 0 < windowsize <= len(self.data[0][-1])
        self.window = windowsize

    def set_token_size(self, tokensize: int = 0):
        """
        tokens are zero-padded to all have the same size at __getitem__
        this function sets the size of the token either to a specific length (rest is cut off)
        or discovers the maximum size
        """
        if tokensize > 0:
            self.tokensize = tokensize
        else:
            logging.info("Discover tokensize")
            # assumes data is already vectorized
            max_len = 0
            for tdx in self.data[0][-1]:
                if tdx.shape[0] > max_len:
                    max_len = tdx.shape[0]
            self.tokensize = max_len

    ## get_weights ####################################################################################################################################################################
    def __get_weights__(
        self,
    ):
        """
        Returns:
            torch.Tensor with full dataset as sequence of components [n_samples,n_tokens_per_sample,token_dim]
        """
        if not self.mode == "vector":
            data_tmp = copy.deepcopy(self.data)
            self.vectorize_data()
            data_out = [
                tokenize(
                    self.data[idx][jdx],
                    tokensize=self.tokensize,
                    return_mask=True,
                )[0]
                for idx in range(len(self.data))
                for jdx in range(len(self.data[idx]))
            ]
            mask_out = [
                tokenize(
                    self.data[idx][jdx],
                    tokensize=self.tokensize,
                    return_mask=True,
                )[1]
                for idx in range(len(self.data))
                for jdx in range(len(self.data[idx]))
            ]
            data_out = torch.stack(data_out)
            mask_out = torch.stack(mask_out)
            self.data = data_tmp
            return data_out, mask_out
        data_out = [
            tokenize(
                self.data[idx][jdx],
                tokensize=self.tokensize,
                return_mask=True,
            )[0]
            for idx in range(len(self.data))
            for jdx in range(len(self.data[idx]))
        ]
        mask_out = [
            tokenize(
                self.data[idx][jdx],
                tokensize=self.tokensize,
                return_mask=True,
            )[1]
            for idx in range(len(self.data))
            for jdx in range(len(self.data[idx]))
        ]
        data_out = torch.stack(data_out)
        mask_out = torch.stack(mask_out)
        logging.debug(f"shape of weight tensor: {data_out.shape}")
        return data_out, mask_out

    ## getitem ####################################################################################################################################################################
    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample to be retrieved
        Returns:
            ddx_idx: torch.Tensor of neuron tokens with shape [n_tokens_per_sample/windowsize,token_dim] in view 1
            mask_idx: torch.Tensor of the same shape as ddx_idx indicating the nonzero elements
            label_idx: label for sample in view 1
            ddx_jdx: torch.Tensor of neuron tokens with shape [n_tokens_per_sample/windowsize,token_dim] in view 2
            mask_jdx: torch.Tensor of the same shape as ddx_jdx indicating the nonzero elements
            label_jdx: label for sample in view 2
            pos: positions [layer,token_in_layer] of sample in view 1 / view 2
        """
        # get model and epoch index
        mdx, edx = self._index[index]

        # get permutation index -> pick random number from available perms
        if self.permutations_number > 0:
            perm_idx, perm_jdx = random.choices(
                list(range(self.permutations_number)), k=2
            )

        ## mode "vector has different workflow"
        if self.mode == "vector":
            # get raw data, assume view 1 and view 2 are the same
            ddx = self.data[mdx][edx]
            # ddx = copy.deepcopy(ddx)
            label = self.labels[mdx][edx]
            label = copy.deepcopy(label)

            # slice window of size windowsize
            # idx_start = torch.randint(low=0, high=len(ddx) - self.window)
            idx_start = random.randint(0, len(ddx) - self.window)

            # get position
            pos = self.positions[idx_start : idx_start + self.window]
            # pos = [
            #     self.positions[idx] for idx in range(idx_start, idx_start + self.window)
            # ]

            # get permutation index -> pick random number from available perms
            if self.permutations_number > 0:
                perm_idx, perm_jdx = random.choices(
                    list(range(self.permutations_number)), k=2
                )

            # permutation
            # permute data idx
            if self.view_1_canonical:
                ddx_idx = copy.deepcopy(ddx)
                ddx_idx = ddx_idx[idx_start : idx_start + self.window]
                label_idx = f"{label}#_#canon"
            elif self.permutations_number > 0:
                # get current permutation for slice of window
                ddx_idx = copy.deepcopy(ddx)
                permdx = self.permutations[perm_idx]
                ddx_idx = permute_model_vector(
                    vec=ddx_idx, perm=permdx, idx_start=idx_start, window=self.window
                )
                # update label
                label_idx = f"{label}#_#per_{perm_idx}"
            else:
                ddx_idx = copy.deepcopy(ddx)
                ddx_idx = ddx_idx[idx_start : idx_start + self.window]
                label_idx = copy.deepcopy(label)

            # permute data jdx
            if self.view_2_canonical:
                ddx_jdx = copy.deepcopy(ddx)
                ddx_jdx = ddx_jdx[idx_start : idx_start + self.window]
                label_jdx = f"{label}#_#canon"
            elif self.permutations_number > 0:
                # get current permutation for slice of window
                ddx_jdx = copy.deepcopy(ddx)
                permdx = self.permutations[perm_jdx]
                ddx_jdx = permute_model_vector(
                    vec=ddx_jdx, perm=permdx, idx_start=idx_start, window=self.window
                )
                # update label
                label_jdx = f"{label}#_#per_{perm_jdx}"
            else:
                ddx_jdx = copy.deepcopy(ddx)
                ddx_jdx = ddx_jdx[idx_start : idx_start + self.window]
                label_jdx = copy.deepcopy(label)

            # noise
            if not self.add_noise_view_1 == False:
                # add noise to input
                # check sigma is larger than 0
                if self.add_noise_view_1 > 0:
                    if self.use_multiplicative_noise:
                        # multiply each token with noise (uniform distribution around 1)
                        # noise idx
                        ddx_idx = [
                            iddx
                            * (1.0 + self.add_noise_view_1 * torch.randn(iddx.shape))
                            for iddx in ddx_idx
                        ]
                    else:
                        # add to each token noise (uniform distribution around 0)
                        # noise idx
                        ddx_idx = [
                            iddx + self.add_noise_view_1 * torch.randn(iddx.shape)
                            for iddx in ddx_idx
                        ]

            if not self.add_noise_view_2 == False:
                # check sigma is number
                assert isinstance(self.add_noise_view_2, float)
                # add noise to input
                # check sigma is larger than 0
                if self.add_noise_view_2 > 0:
                    if self.use_multiplicative_noise:
                        # multiply each token with noise (uniform distribution around 1)
                        # noise jdx
                        ddx_jdx = [
                            iddx
                            * (1.0 + self.add_noise_view_2 * torch.randn(iddx.shape))
                            for iddx in ddx_jdx
                        ]
                    else:
                        # add to each token noise (uniform distribution around 0)
                        # noise jdx
                        ddx_jdx = [
                            iddx + self.add_noise_view_2 * torch.randn(iddx.shape)
                            for iddx in ddx_jdx
                        ]

            # erase_input/output augmentation
            if self.erase_augment is not None:
                # apply erase_augment on each token
                ddx_idx = [self.erase_augment(iddx) for iddx in ddx_idx]
                ddx_jdx = [self.erase_augment(iddx) for iddx in ddx_jdx]

            # tokenize
            ddx_idx, mask_idx = tokenize(t=ddx_idx, tokensize=self.tokensize)
            ddx_jdx, mask_jdx = tokenize(t=ddx_jdx, tokensize=self.tokensize)

            # return ddx_idx, mask_idx, label_idx, ddx_jdx, mask_jdx, label_jdx, pos
            return ddx_idx, mask_idx, ddx_jdx, mask_jdx, pos
        ### end mode=="vector"

        # TODO implement version on checkpoints, relatively straight-forward
        raise NotImplementedError(
            "simclr dataset is not yet implemented for checkpoints. hang in there.."
        )

    ### len ##################################################################################################################################################################
    def init_len(self):
        index = []
        for idx, ddx in enumerate(self.data):
            idx_tmp = [(idx, jdx) for jdx in range(len(ddx))]
            index.extend(idx_tmp)
        self._len = len(index)
        self._index = index

    def __len__(self):
        return self._len

    ### set erase ############################################################
    def set_erase(self, erase=None):
        if erase is not None:
            assert (
                self.mode == "vectorize" or self.mode == "vector"
            ), "erasing is only for vectorized mode implemenetd"
            erase = RandomErasingVector(
                p=erase["p"],
                scale=erase["scale"],
                value=erase["value"],
                mode=erase["mode"],
            )
        else:
            erase = None
        self.erase_augment = erase

    ### vectorize_data #########################################################################################################################################################
    def vectorize_data(self):
        # iterate over models
        for idx in range(len(self.data)):
            # iterate over epochs
            for jdx in range(len(self.data[idx])):
                # get checkpoint
                checkpoint = copy.deepcopy(self.data[idx][jdx])
                # get vectorize
                ddx = vectorize_checkpoint(checkpoint)
                # overwrite data
                self.data[idx][jdx] = ddx

    ### standardize data #########################################################################################################################################################
    def standardize_data(self):
        """
        standardize data to zero-mean / unit std. per layer
        store per-layer mean / std
        """
        logging.info("Get layer mapping")
        # step 1: get token-layer index relation
        layers = {}
        # init vals
        cur_layer = 0
        cur_layer_start = 0
        # iterate over models
        for idx in range(self.positions.shape[0]):
            ldx, kdx = self.positions[idx, 0].item(), self.positions[idx, 1].item()
            # if new layer
            if not ldx == cur_layer:
                # add previous layer to dict
                layer = str(cur_layer)
                cur_layer_end = idx - 1
                layers[layer] = {
                    "start_idx": cur_layer_start,
                    "end_idx": cur_layer_end,
                }
                # start new layer
                cur_layer = ldx
                cur_layer_start = idx
        # last layer
        layer = str(ldx)
        cur_layer_end = idx
        layers[layer] = {
            "start_idx": cur_layer_start,
            "end_idx": cur_layer_end,
        }
        logging.debug(f"layer mapping: {layers}")

        logging.info("Get layer-wise mean and std")

        # iterate over layers
        for layer in layers:
            idx_start = layers[layer]["start_idx"]
            idx_end = layers[layer]["end_idx"]

            # collect all tokens within the layer for all models
            tmp = []
            # iterate over models
            for idx in range(len(self.data)):
                for jdx in range(len(self.data[idx])):
                    tmp2 = [
                        self.data[idx][jdx][ldx]
                        for ldx in range(idx_start, idx_end + 1)
                    ]
                    tmp.extend(tmp2)

            # stack / cat
            tmp = torch.stack(tmp, dim=0)

            # compute mean / std
            mu = torch.mean(tmp)
            sigma = torch.std(tmp)

            # store in layer
            layers[layer]["mean"] = mu
            layers[layer]["std"] = sigma

            # free memory
            del tmp

        self.layers = layers

        logging.info("Apply standardization")
        # TODO: make more efficient, cut out the first two for loops with list expression?
        # standardize:
        # # iterate over models
        for idx in range(len(self.data)):
            for jdx in range(len(self.data[idx])):
                # # iterate over tokens of that layer
                for tdx in range(len(self.data[idx][jdx])):
                    # get position
                    pos = self.positions[tdx]
                    ldx = pos[0].item()
                    # get mu/sigma for that token
                    mu = layers[str(ldx)]["mean"]
                    std = layers[str(ldx)]["std"]
                    # # standardize with mean / std
                    self.data[idx][jdx][tdx] = (self.data[idx][jdx][tdx] - mu) / std

    ### precompute_permutation_index #########################################################################################################################################################
    def precompute_permutation_index(self):
        # ASSUMES THAT DATA IS ALREADY VECTORIZED
        permutation_index_list = []
        # create index vector
        # print(f"vector shape: {self.data_in[0].shape}")
        index_vector = torch.tensor(list(range(self.data_in[0].shape[0])))
        # cast index vector to double
        index_vector = index_vector.double()
        # print(f"index vector: {index_vector}")
        # reference checkpoint
        reference_checkpoint = copy.deepcopy(self.reference_checkpoint)
        # cast index vector to checkpoint
        index_checkpoint = vector_to_checkpoint(
            checkpoint=copy.deepcopy(reference_checkpoint),
            vector=copy.deepcopy(index_vector),
            layer_lst=self.layer_lst,
            use_bias=self.use_bias,
        )

        ## init multiprocessing environment ############
        ray.init(num_cpus=self.num_threads)

        ### gather data #############################################################################################
        logging.info(f"preparing permutation indices from {self.root}")
        pb = ProgressBar(total=self.permutations_number)
        pb_actor = pb.actor

        # loop over all permutations in self.permutations_number
        for pdx in range(self.permutations_number):
            # get perm dict
            prmt_dct = self.permutations_dct_lst[pdx]
            #
            index_p = compute_single_index_vector_remote.remote(
                index_checkpoint=copy.deepcopy(index_checkpoint),
                prmt_dct=prmt_dct,
                layer_lst=self.layer_lst,
                permute_layers=self.permute_layers,
                use_bias=self.use_bias,
                pba=pb_actor,
            )
            # append to permutation_index_list
            permutation_index_list.append(index_p)

        # update progress bar
        pb.print_until_done()

        # collect actual data
        permutation_index_list = ray.get(permutation_index_list)

        ray.shutdown()

        self.permutation_index_list = permutation_index_list

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
            # get second
            model_curr = self.data[idx]
            model_curr = compute_single_canon_form.remote(
                reference_model=reference_model,
                data_curr=model_curr,
                perm_spec=perm_spec,
                pba=pb_actor,
            )
            self.data[idx] = model_curr

        # update progress bar
        pb.print_until_done()

        self.data = [ray.get(self.data[idx]) for idx in range(len(self.data))]
        ray.shutdown()

    ### precompute_permutations #############################################################################################
    def precompute_permutations(self, permutation_number, perm_spec, num_threads=6):
        """
        - get permutation_dict as template
        - generate random permutations
        - generate index checkpoint
            - copy actual checkpoint
            - flatten tensor
            - generate list with indices of same shape
            - put on tensor
            - reshape to original view
        - apply permutation on checkpoint: store checkpoint as permutation dict.

        - permuting modules translates to:
            - get right index for current module
            - .flatten()
            - apply permutation /slice
            - .view()
        """
        logging.info("start precomputing permutations")
        # model_curr = self.data[0][-1]
        model_curr = self.reference_checkpoint
        # find permutation of model to itself as reference
        reference_permutation = weight_matching(
            ps=perm_spec, params_a=model_curr, params_b=model_curr
        )

        logging.info("get random permutation dicts")
        # compute random permutations
        permutation_dicts = []
        for ndx in range(permutation_number):
            perm = copy.deepcopy(reference_permutation)
            for key in perm.keys():
                # get permuted indecs for current layer
                perm[key] = torch.randperm(perm[key].shape[0]).float()
            # append to list of permutation dicts
            permutation_dicts.append(perm)

        self.permutation_dicts = permutation_dicts

        if self.mode == "vector":
            logging.info("get permutation indices")
            # get permutation data
            ## get reference checkpoint
            ref_checkpoint = copy.deepcopy(model_curr)
            ## vectoirze
            ref_vec_global = vectorize_checkpoint(ref_checkpoint)
            ref_vec_kernel = copy.deepcopy(ref_vec_global)
            ## get reference index vec
            for idx, module in enumerate(ref_vec_global):
                # get global index of permutation between kernels
                index_global = torch.ones(module.numel()) * idx
                index_global = index_global.view(module.shape)
                ref_vec_global[idx] = index_global
                # got local index of permutation within kernels
                index_kernel = torch.tensor(list(range(module.numel())))
                index_kernel = index_kernel.view(module.shape)
                ref_vec_kernel[idx] = index_kernel
            ## map to checkpoint
            ref_checkpoint_global = vector_to_checkpoint(
                vector=ref_vec_global, reference_checkpoint=ref_checkpoint
            )
            ref_checkpoint_kernel = vector_to_checkpoint(
                vector=ref_vec_kernel, reference_checkpoint=ref_checkpoint
            )

            ## init multiprocessing environment ############
            ray.init(num_cpus=num_threads)
            pb = ProgressBar(total=permutation_number)
            pb_actor = pb.actor
            # get permutations
            permutations_global = []
            permutations_kernel = []
            for perm_dict in permutation_dicts:
                perm_curr_global = compute_single_perm.remote(
                    reference_checkpoint=ref_checkpoint_global,
                    permutation_dict=perm_dict,
                    perm_spec=perm_spec,
                    pba=pb_actor,
                )

                perm_curr_kernel = compute_single_perm.remote(
                    reference_checkpoint=ref_checkpoint_kernel,
                    permutation_dict=perm_dict,
                    perm_spec=perm_spec,
                    pba=pb_actor,
                )

                permutations_global.append(perm_curr_global)
                permutations_kernel.append(perm_curr_kernel)

            permutations_global = ray.get(permutations_global)
            permutations_kernel = ray.get(permutations_kernel)

            permutations_global = [
                torch.tensor([perm[0].item() for perm in perm_g]).int()
                for perm_g in permutations_global
            ]

            permutations = [
                (perm_g, perm_k)
                for (perm_g, perm_k) in zip(permutations_global, permutations_kernel)
            ]

            ray.shutdown()

            self.permutations = permutations


def permute_model_vector(
    vec: List[torch.Tensor], perm: List[tuple], idx_start: int, window: int
):
    """
    performs permutation on vectorized model and returns slice of tokens
    input vec: list(torch.tensor) with the weights per output channel of the full model
    input perm: contains two pieces of information for permutation. perm[0] is the global permutation of the tokens, perm[1] contains the permutaion mappings within tokens per token
    input idx_start: int marks the start of a slice of tokens to keep
    input window: int marks the size of the slice to keep
    return vec: list(torch.tensor) of permuted and sliced tokens
    """
    # create index vector of tokens
    index = list(range(len(vec)))

    # apply global permutation on index
    # using (slices of) the permuted index to access tokens equals permuting all tokens and slicing after
    perm_glob = perm[0]
    index = [index[idx] for idx in perm_glob]

    # slice index
    idx_end = idx_start + window
    index = index[idx_start:idx_end]

    # slice token sequence
    vec = [vec[idx] for idx in index]

    # slice permutations
    perm_loc = perm[1]
    perm_loc = [perm_loc[idx] for idx in index]

    # apply token permutation
    vec = [vecdx[permdx] for (vecdx, permdx) in zip(vec, perm_loc)]

    # return tokens
    return vec


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


### helper parallel function #############################################################################################
@ray.remote(num_returns=1)
def compute_single_canon_form(reference_model, data_curr, perm_spec, pba):
    # get second
    model_curr = data_curr[-1]
    # find permutation to match params_b to params_a
    logging.debug(
        f"compute canonical form: params a {type(reference_model)} params b {type(model_curr)}"
    )
    match_permutation = weight_matching(
        ps=perm_spec, params_a=reference_model, params_b=model_curr
    )
    # apply permutation on all epochs
    for jdx in range(len(data_curr)):
        model_curr = data_curr[jdx]
        model_curr_perm = apply_permutation(
            ps=perm_spec, perm=match_permutation, params=model_curr
        )
        # put back in data
        data_curr[jdx] = model_curr_perm

    # update counter
    pba.update.remote(1)
    # return list
    return data_curr


### helper parallel function #############################################################################################
@ray.remote(num_returns=1)
def compute_single_index_vector_remote(
    index_checkpoint, prmt_dct, layer_lst, permute_layers, use_bias, pba
):
    # apply permutation on copy of unit checkpoint
    # TODO: fix function,
    chkpt_p = permute_checkpoint(
        index_checkpoint,
        layer_lst,
        permute_layers,
        prmt_dct,
    )

    # cast back to vector
    vector_p = vectorize_checkpoint(copy.deepcopy(chkpt_p), layer_lst, use_bias)
    # cast vector back to int
    vector_p = vector_p.int()
    # we specifically don't check for uniqueness of indices. we'd rather let this run into index errors to catch the issue
    index_p = copy.deepcopy(vector_p.tolist())
    # update counter
    pba.update.remote(1)
    # return list
    return index_p


def vectorize_checkpoint(checkpoint):
    out = []
    # use only weights and biases
    for key in checkpoint.keys():
        if "weight" in key:
            w = checkpoint[key]
            # flatten to out_channels x n
            w = w.view(w.shape[0], -1)
            # cat biases to channels if they exist in checkpoint
            if key.replace("weight", "bias") in checkpoint:
                b = checkpoint[key.replace("weight", "bias")]
                w = torch.cat([w, b.unsqueeze(dim=1)], dim=1)
            # split weights in slices along output channel dims
            w = torch.split(w, w.shape[0])
            # extend out with new tokens, zero's (and only entry) is a list
            out.extend(w[0])

    return out


def vector_to_checkpoint(vector, reference_checkpoint):
    # make copy to prevent memory management issues
    checkpoint = copy.deepcopy(reference_checkpoint)
    # use only weights and biases
    idx_start = 0
    for key in checkpoint.keys():
        if "weight" in key:
            # get correct slice of modules out of vec sequence
            out_channels = checkpoint[key].shape[0]
            idx_end = idx_start + out_channels
            w = vector[idx_start:idx_end]
            # get weight matrix from list of vectors
            try:
                w = torch.stack(w, dim=0)
            except Exception as e:
                logging.error(
                    f"vector_to_checkpoint: layer {key}, idx: [{idx_start},{idx_end}] created weight {w.shape} for checkpoint weight {checkpoint[key].shape}"
                )
                logging.error(e)
            # extract bias
            if key.replace("weight", "bias") in checkpoint:
                b = w[:, -1]
                checkpoint[key.replace("weight", "bias")] = b
                w = w[:, :-1]
            # reshape weight vector
            w = w.view(checkpoint[key].shape)
            logging.debug(
                f"vector_to_checkpoint: layer {key}, idx: [{idx_start},{idx_end}] tried to create weights from {[wdx.shape] for wdx in w} for checkpoint weight {checkpoint[key].shape}"
            )
            checkpoint[key] = w
            # update start
            idx_start = idx_end
    return checkpoint


def get_position_mapping_from_checkpoint(checkpoint) -> list:
    """
    Args:
        checkpoint: Collections.OrderedDict model checkpoint
    Returns:
        output tensor with 2d positions for every token in the vectorized model sequence
    """
    out = []
    # start layer index counter at 0
    idx = 0
    # iterate over modules
    for key in checkpoint.keys():
        # if module with weights is found -> add index
        if "weight" in key:
            w = checkpoint[key]
            # create tuple (layer_idx, channel_idx) for every channel
            idx_layer = [torch.tensor([idx, jdx]) for jdx in range(w.shape[0])]
            # add to overall position
            out.extend(idx_layer)
            # increase layer counter
            idx += 1
    out = torch.stack(out, dim=0)
    return out


def tokenize(t: List[torch.tensor], tokensize: int, return_mask: bool = True):
    """
    transforms list of tokens of differen lenght to tensor
    Args:
        t: List[torch.tensor]: list of 1d input tokens of different lenghts
        tokensize: int output dimension of each token
        return_mask: bool wether to return the mask of nonzero values
    Returns:
        tokens: torch.tensor with tokens stacked along dim=0
        mask: torch.tensor indicating the shape of the original tokens
    """
    # init output with zeros
    tokens = torch.zeros(len(t), tokensize)
    mask = torch.zeros(len(t), tokensize)
    # iterate over inputs
    for idx, tdx in enumerate(t):
        # get end of token, either the length of the input or them maximum length
        tdx_end = min(tdx.shape[0], tokensize)
        # put at position idx
        tokens[idx, :tdx_end] = tdx[:tdx_end]
        mask[idx, :tdx_end] = torch.ones(tdx_end)

    # return
    if return_mask:
        return tokens, mask
    else:
        return tokens
