from ray.tune import Trainable
from ray.tune.utils import wait_for_gpu
import torch
import sys

import os

import json

from pathlib import Path

from shrp.datasets.dataset_simclr import SimCLRDataset

from shrp.models.def_AE_module import AEModule

from torch.utils.data import DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from shrp.models.def_downstream_module import DownstreamTaskLearner

from ray.air.integrations.wandb import setup_wandb

import logging

from shrp.datasets.augmentations import (
    AugmentationPipeline,
    TwoViewSplit,
    WindowCutter,
    ErasingAugmentation,
    NoiseAugmentation,
)


###############################################################################
# define Tune Trainable
###############################################################################
class AE_DDP_trainable(Trainable):
    """
    DDP courtesy of https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html
    tune trainable wrapper around AE model experiments
    Loads datasets, configures model, performs (training) steps and returns performance metrics
    Args:
        config (dict): config dictionary
        data (dict): data dictionary (optional)
    """

    def setup(self, config):
        """
        Init function to set up experiment. Configures data, augmentaion, and module
        Args:
            config (dict): config dictionary
            data (dict): data dictionary (optional)
        """
        logging.info("Set up AE Trainable")
        logging.debug(f"Trainable Config: {config}")

        # set trainable properties
        self.config = config
        self.seed = config["seed"]

        # figure out how much of the GPU to wait for
        logging.info("Wait for resources to become available")
        resources = config.get("resources", None)
        target_util = None
        if resources is not None:
            gpu_resource_share = resources.get("gpu", 0)
            # more than at least one gpu
            if gpu_resource_share > 1.0 - 1e-5:
                target_util = 0.01
            else:
                # set target util maximum full load minus share - buffer
                target_util = 1.0 - gpu_resource_share - 0.01
        else:
            target_util = 0.01
        # wait for gpu memory to be available
        if target_util is not None:
            logging.info("cuda detected: wait for gpu memory to be available")
            wait_for_gpu(gpu_id=None, target_util=target_util, retry=20, delay_s=5)

        # load config if restore from previous checkpoint
        if config.get("model::checkpoint_path", None):
            config_path = config.get("model::checkpoint_path", None).joinpath(
                "..", "params.json"
            )
            logging.info(
                f"restore model from previous checkpoint. load config from {config_path}"
            )
            config_old = json.load(config_path.open("r"))
            # transfer all 'model' keys to
            for key in config_old.keys():
                if "model::" in key:
                    self.config[key] = config_old[key]

        # initiate ddp 
        self.rank = 0
        self.world_size = torch.cuda.device_count()
        ddp_setup(rank=self.rank, world_size=self.world_size)


        # instanciate Model
        logging.info("instanciate model")
        # set distributed flag to make sure model is configured for distributed training
        config["distributed"] = "ddp"
        # call constructor
        self.module = AEModule(config=config)


        # load checkpoint
        if config.get("model::checkpoint_path", None):
            logging.info(
                f'restore model state from {config.get("model::checkpoint_path",None)}'
            )
            # load all state dicts
            self.load_checkpoint(config.get("model::checkpoint_path", None))
            # reset optimizer
            self.module.set_optimizer(config)

        # set windowsize
        logging.info("adjust augmentations")
        windowsize = config.get("training::windowsize", 15)
        logging.info(f"adjust windowsize to {windowsize}")

        ### START TRANSFORAMTIONS ######
        # TRAIN AUGMENTATIONS
        stack_1 = []
        if self.config.get("trainset::add_noise_view_1", 0.0) > 0.0:
            stack_1.append(
                NoiseAugmentation(self.config.get("trainset::add_noise_view_1", 0.0))
            )
        if self.config.get("trainset::erase_augment", None) is not None:
            stack_1.append(ErasingAugmentation(**config["trainset::erase_augment"]))
        stack_2 = []
        if self.config.get("trainset::add_noise_view_2", 0.0) > 0.0:
            stack_2.append(
                NoiseAugmentation(self.config.get("trainset::add_noise_view_2", 0.0))
            )
        if self.config.get("trainset::erase_augment", None) is not None:
            stack_2.append(ErasingAugmentation(**config["trainset::erase_augment"]))
        stack_train = [
            WindowCutter(windowsize=windowsize),
            TwoViewSplit(
                stack_1=stack_1,
                stack_2=stack_2,
            ),
        ]

        trafo_train = AugmentationPipeline(stack=stack_train)

        # test AUGMENTATIONS
        stack_1 = []
        if self.config.get("testset::add_noise_view_1", 0.0) > 0.0:
            stack_1.append(
                NoiseAugmentation(self.config.get("testset::add_noise_view_1", 0.0))
            )
        if self.config.get("testset::erase_augment", None) is not None:
            stack_1.append(ErasingAugmentation(**config["testset::erase_augment"]))
        stack_2 = []
        if self.config.get("testset::add_noise_view_2", 0.0) > 0.0:
            stack_2.append(
                NoiseAugmentation(self.config.get("testset::add_noise_view_2", 0.0))
            )
        if self.config.get("testset::erase_augment", None) is not None:
            stack_2.append(ErasingAugmentation(**config["testset::erase_augment"]))
        stack_test = [
            WindowCutter(windowsize=windowsize),
            TwoViewSplit(
                stack_1=stack_1,
                stack_2=stack_2,
            ),
        ]

        trafo_test = AugmentationPipeline(stack=stack_test)

        # set trafos
        self.module.set_transforms(trafo_train, trafo_test)
        ### END TRANSFORAMTIONS ######

        # conventional dataset
        if "dataset.pt" in str(self.config["dataset::dump"]):
            # init dataloaders
            logging.info("Load Data")
            # load dataset from file
            dataset = torch.load(self.config["dataset::dump"])

            self.trainset = dataset["trainset"]
            self.testset = dataset["testset"]
            self.valset = dataset.get("valset", None)
            # get full dataset in tensors
            logging.info("set up dataloaders")
            self.trainloader = DataLoader(
                self.trainset,
                batch_size=self.config["trainset::batchsize"],
                shuffle=False,
                drop_last=True,  # important: we need equal batch sizes
                num_workers=self.config.get("trainloader::workers", 2),
                prefetch_factor=4,
                sampler=DistributedSampler(self.trainset),
            )

            # get full dataset in tensors
            self.testloader = torch.utils.data.DataLoader(
                self.testset,
                batch_size=self.config["trainset::batchsize"],
                shuffle=False,
                drop_last=True,  # important: we need equal batch sizes
                num_workers=self.config.get("testloader::workers", 2),
                prefetch_factor=4,
                sampler=DistributedSampler(self.testset),
            )
            if self.valset is not None:
                # get full dataset in tensors
                self.valloader = torch.utils.data.DataLoader(
                    self.valset,
                    batch_size=self.config["trainset::batchsize"],
                    shuffle=False,
                    drop_last=True,  # important: we need equal batch sizes
                    num_workers=self.config.get("testloader::workers", 2),
                    prefetch_factor=4,
                    sampler=DistributedSampler(self.valset),
                )
        # ffcv type dataset
        elif "dataset_beton" in str(self.config["dataset::dump"]):
            raise NotImplementedError
            # from ffcv.loader import Loader, OrderOption

            # # trainloader
            # batch_size = self.config["trainset::batchsize"]
            # num_workers = self.config.get("testloader::workers", 4)
            # ordering = OrderOption.QUASI_RANDOM
            # # Dataset ordering
            # path_trainset = str(config["dataset::dump"]) + ".train"
            # self.trainloader = Loader(
            #     path_trainset,
            #     batch_size=batch_size,
            #     num_workers=num_workers,
            #     order=ordering,
            #     drop_last=True,
            #     # pipelines=PIPELINES
            #     os_cache=False,
            # )
            # # trainloader
            # batch_size = self.config["trainset::batchsize"]
            # num_workers = self.config.get("testloader::workers", 4)
            # ordering = OrderOption.SEQUENTIAL
            # # Dataset ordering
            # path_testset = str(config["dataset::dump"]) + ".test"
            # self.testloader = Loader(
            #     path_testset,
            #     batch_size=batch_size,
            #     num_workers=num_workers,
            #     order=ordering,
            #     drop_last=True,
            #     # pipelines=PIPELINES
            #     os_cache=False,
            # )
            # # config
            # batch_size = self.config["trainset::batchsize"]
            # num_workers = self.config.get("testloader::workers", 4)
            # ordering = OrderOption.SEQUENTIAL
            # # Dataset ordering
            # path_valset = str(config["dataset::dump"]) + ".val"
            # self.valloader = Loader(
            #     path_valset,
            #     batch_size=batch_size,
            #     num_workers=num_workers,
            #     order=ordering,
            #     drop_last=True,
            #     # pipelines=PIPELINES
            #     os_cache=False,
            )

        # remove fabric for now
        # # set up fabric
        # self.module.initialize_fabric(
        #     config=self.config,
        #     trainloader=self.trainloader,
        #     testloader=self.testloader,
        #     valloader=self.valloader,
        # )

        # compute loss_mean
        logging.info(f"set normalization")
        weights_train = []
        masks_train = []
        for wdx, mdx, _ in self.trainloader:
            weights_train.append(wdx)
            masks_train.append(mdx)
        weights_train = torch.cat(weights_train, dim=0)
        masks_train = torch.cat(masks_train, dim=0)
        self.module.criterion.set_mean_loss(weights_train, masks_train)

        # run first test epoch and log results
        self._iteration = -1

        # DownstreamTask Learners
        if self.trainset.properties is not None:
            logging.info(
                "Found properties in dataset - downstream tasks are going to be evaluated at test time."
            )
            self.dstk = DownstreamTaskLearner()
        else:
            logging.info("No properties found in dataset - skip downstream tasks.")
            self.dstk = None

        # # print model summary
        # try:
        #     logging.info(f"generate model summary")
        #     import pytorch_model_summary as pms

        #     x_i, _, _, _, _, _, p = next(iter(self.testloader))
        #     inpts = (x_i, p)
        #     summary = pms.summary(
        #         self.module,
        #         inpts,
        #         show_input=False,
        #         show_hierarchical=True,
        #         print_summary=False,
        #         max_depth=5,
        #         show_parent_layers=False,
        #     )
        #     fname_summary = Path(self.logdir).joinpath("model_summary.txt")
        #     with fname_summary.open("w") as f:
        #         f.write(summary)
        # except Exception as e:
        #     print(e)

    # step ####
    def step(self):
        # run several training epochs before one test epoch
        if self._iteration < 0:
            print("test first validation mode")
            perf_train = self.module.test_epoch(self.trainloader)

        else:
            for _ in range(self.config["training::test_epochs"]):
                # run one training epoch
                perf_train = self.module.train_epoch(self.trainloader,epoch=self._iteration)
        # run one test epoch
        perf_test = self.module.test_epoch(self.testloader)
        result_dict = {}
        for key in perf_test.keys():
            result_dict[f"{key}_test"] = perf_test[key]

        for key in perf_train.keys():
            result_dict[f"{key}_train"] = perf_train[key]

        if self.valset is not None:
            # run one test epoch
            perf_val = self.module.test_epoch(
                self.valloader,
            )
            for key in perf_val.keys():
                result_dict[f"{key}_val"] = perf_val[key]

        # if DownstreamTaskLearner exist. apply downstream task
        if self.dstk is not None:
            performance = self.dstk.eval_dstasks(
                # model=self.module.model,
                model=self.module,
                trainset=self.trainset,
                testset=self.testset,
                valset=self.valset,
                batch_size=self.config["trainset::batchsize"],
            )
            # append performance values to result_dict
            for key in performance.keys():
                result_dict[key] = performance[key]
        return result_dict

    def set_loss_weights(
        self,
    ):
        index_dict = self.config["model::index_dict"]
        weights = []
        for idx, layer in enumerate(index_dict["layer"]):
            w = self.config.get(f"training::loss_weight_layer_{idx+1}", 1.0)
            weights.append(w)
        self.config["model::index_dict"]["loss_weight"] = weights
        print("#### weighting the loss per layer as follows:")
        for idx, layer in enumerate(index_dict["layer"]):
            print(
                f'layer {layer} - weight {self.config["model::index_dict"]["loss_weight"][idx]}'
            )

    def save_checkpoint(self, experiment_dir):
        """
        saves model checkpoint and optimizer state_dict
        Args:
            experiment_dir: path to experiment directory for model saving
        Returns:
            experiment_dir: path to experiment directory for model saving as per tune convention
        """
        self.module.save_model(experiment_dir)
        # tune apparently expects to return the directory
        return experiment_dir

    def load_checkpoint(self, experiment_dir):
        """
        loads model checkpoint and optimizer state_dict
        Uses self.reset_optimizer to decide if optimizer should be loaded
        Args:
            experiment_dir: path to experiment directory for model loading
        Returns:
            experiment_dir: path to experiment directory for model loading as per tune convention
        """
        self.module.load_model(experiment_dir)
        # tune apparently expects to return the directory
        return experiment_dir


def ddp_setup(rank: int, world_size: int):
    """
    Courtesy of https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
