from ray.tune.utils import wait_for_gpu
import torch
import sys

import json

# print(f"sys path in experiment: {sys.path}")
from pathlib import Path

from shrp.datasets.dataset_simclr import SimCLRDataset

# import model_definitions
from shrp.models.def_AE_module import AEModule

from torch.utils.data import DataLoader

import logging

from shrp.datasets.augmentations import (
    AugmentationPipeline,
    TwoViewSplit,
    WindowCutter,
    ErasingAugmentation,
    NoiseAugmentation,
    MultiWindowCutter,
    StackBatches,
    PermutationSelector,
)

from shrp.models.downstream_module_ffcv import (
    DownstreamTaskLearner as DownstreamTaskLearnerFFCV,
)
from shrp.models.def_downstream_module import (
    DownstreamTaskLearner as DownstreamTaskLearner,
)


###############################################################################
# define AE_trainer class
###############################################################################
class AE_trainer:
    """
    trainer wrapper around AE model experiments
    Loads datasets, configures model, performs (training) steps and returns performance metrics
    Args:
        config (dict): config dictionary
        data (dict): data dictionary (optional)
    """

    def __init__(self, config):
        """
        Init function to set up experiment. Configures data, augmentaion, and module
        Args:
            config (dict): config dictionary
        """
        logging.info("Set up AE Trainable")
        logging.debug(f"Trainable Config: {config}")

        # set trainable properties
        self.config = config
        self.seed = config["seed"]

        # callbacks
        self.callbacks = self.config.get("callbacks", None)
        # print(f'callbacks: {self.callbacks}')

        # figure out how much of the GPU to wait for
        # logging.info("Wait for resources to become available")
        # resources = config.get("resources", None)
        # target_util = None
        # if resources is not None:
        #     gpu_resource_share = resources.get("gpu", 0)
        #     # more than at least one gpu
        #     if gpu_resource_share > 1.0 - 1e-5:
        #         target_util = 0.01
        #     else:
        #         # set target util maximum full load minus share - buffer
        #         target_util = 1.0 - gpu_resource_share - 0.01
        # else:
        #     target_util = 0.01
        # # wait for gpu memory to be available
        # if target_util is not None:
        #     logging.info("cuda detected: wait for gpu memory to be available")
        #     wait_for_gpu(gpu_id=None, target_util=target_util, retry=20, delay_s=5)

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

        # instanciate Model
        logging.info("instanciate model")
        self.module = AEModule(config=config)
        self.module.model.eval()

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

        # MULTIWINDOW
        if self.config.get("trainset::multi_windows", None):
            trafo_dataset = MultiWindowCutter(
                windowsize=windowsize, k=self.config.get("trainset::multi_windows")
            )

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

        stack_train = []
        if self.config.get("trainset::multi_windows", None):
            stack_train.append(StackBatches())
        else:
            stack_train.append(WindowCutter(windowsize=windowsize))
        # put train stack together
        if config.get("training::permutation_number", 0) == 0:
            split_mode = "copy"
            view_1_canon = True
            view_2_canon = True
        else:
            split_mode = "permutation"
            view_1_canon = config.get("training::view_1_canon", True)
            view_2_canon = config.get("training::view_2_canon", False)
        stack_train.append(
            TwoViewSplit(
                stack_1=stack_1,
                stack_2=stack_2,
                mode=split_mode,
                view_1_canon=view_1_canon,
                view_2_canon=view_2_canon,
            ),
        )

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

        stack_test = []
        if self.config.get("trainset::multi_windows", None):
            stack_test.append(StackBatches())
        else:
            stack_test.append(WindowCutter(windowsize=windowsize))
        # put together
        if config.get("testing::permutation_number", 0) == 0:
            split_mode = "copy"
            view_1_canon = True
            view_2_canon = True
        else:
            split_mode = "permutation"
            view_1_canon = config.get("testing::view_1_canon", True)
            view_2_canon = config.get("testing::view_2_canon", False)
        stack_test.append(
            TwoViewSplit(
                stack_1=stack_1,
                stack_2=stack_2,
                mode=split_mode,
                view_1_canon=view_1_canon,
                view_2_canon=view_2_canon,
            ),
        )

        # TODO: pass through permutation / view_1/2 canonical
        trafo_test = AugmentationPipeline(stack=stack_test)

        # downstream task permutation (chose which permutationn to use for dstk)
        if config.get("training::permutation_number", 0) > 0:
            trafo_dst = PermutationSelector(mode="canonical", keep_properties=True)
        else:
            trafo_dst = PermutationSelector(mode="identity", keep_properties=True)
        # set trafos
        self.module.set_transforms(trafo_train, trafo_test, trafo_dst)
        ### END TRANSFORAMTIONS ######

        # conventional dataset
        if "dataset.pt" in str(self.config["dataset::dump"]):
            # import conventional downstream module

            # init dataloaders
            dataset = torch.load(self.config["dataset::dump"])

            self.trainset = dataset["trainset"]
            self.testset = dataset["testset"]
            self.valset = dataset.get("valset", None)

            # transfer trafo_dataset to datasets
            self.trainset.transforms = (
                trafo_dataset  # this applies multi-windowcutter, etc.
            )
            self.testset.transforms = trafo_dataset
            if self.valset is not None:
                self.valset.transforms = trafo_dataset

            # get full dataset in tensors
            logging.info("set up dataloaders")
            # correct dataloader batchsize with # of multi_window samples out of single __getitem__ call
            assert (
                self.config["trainset::batchsize"]
                % self.config.get("trainset::multi_windows", 1)
                == 0
            ), f'batchsize {self.config["trainset::batchsize"]} needs to be divisible by multi_windows {self.config["trainset::multi_windows"]}'
            bs_corr = int(
                self.config["trainset::batchsize"]
                / self.config.get("trainset::multi_windows", 1)
            )
            logging.info(f"corrected batchsize to {bs_corr}")
            #
            self.trainloader = DataLoader(
                self.trainset,
                batch_size=bs_corr,
                shuffle=True,
                drop_last=True,  # important: we need equal batch sizes
                num_workers=self.config.get("trainloader::workers", 2),
                prefetch_factor=4,
            )

            # get full dataset in tensors
            self.testloader = torch.utils.data.DataLoader(
                self.testset,
                batch_size=bs_corr,
                shuffle=False,
                drop_last=True,  # important: we need equal batch sizes
                num_workers=self.config.get("testloader::workers", 2),
                prefetch_factor=4,
            )
            if self.valset is not None:
                # get full dataset in tensors
                self.valloader = torch.utils.data.DataLoader(
                    self.valset,
                    batch_size=bs_corr,
                    shuffle=False,
                    drop_last=True,  # important: we need equal batch sizes
                    num_workers=self.config.get("testloader::workers", 2),
                    prefetch_factor=4,
                )
            else:
                self.valloader = None
        # ffcv type dataset
        elif "dataset_beton" in str(self.config["dataset::dump"]):
            from ffcv.loader import Loader, OrderOption
            from ffcv.fields.decoders import NDArrayDecoder
            from ffcv.transforms import ToTensor, Convert

            # trainloader
            batch_size = self.config["trainset::batchsize"]
            num_workers = self.config.get("testloader::workers", 4)
            ordering = OrderOption.QUASI_RANDOM
            weights_dtype = (
                torch.float32
                if config.get("training::precision", "32") == "32"
                else torch.float16
            )
            PIPELINES = {
                "w": [NDArrayDecoder(), ToTensor(), Convert(weights_dtype)],
                "m": [NDArrayDecoder(), ToTensor()],
                "p": [NDArrayDecoder(), ToTensor()],
                "props": [NDArrayDecoder(), ToTensor()],
            }

            # Dataset ordering
            path_trainset = str(config["dataset::dump"]) + ".train"
            self.trainloader = Loader(
                path_trainset,
                batch_size=batch_size,
                num_workers=num_workers,
                order=ordering,
                drop_last=True,
                pipelines=PIPELINES,
                os_cache=False,
            )
            # trainloader
            batch_size = self.config["trainset::batchsize"]
            num_workers = self.config.get("testloader::workers", 4)
            ordering = OrderOption.SEQUENTIAL
            # Dataset ordering
            path_testset = str(config["dataset::dump"]) + ".test"
            self.testloader = Loader(
                path_testset,
                batch_size=batch_size,
                num_workers=num_workers,
                order=ordering,
                drop_last=True,
                pipelines=PIPELINES,
                os_cache=False,
            )
            # config
            batch_size = self.config["trainset::batchsize"]
            num_workers = self.config.get("testloader::workers", 4)
            ordering = OrderOption.SEQUENTIAL
            # Dataset ordering
            path_valset = str(config["dataset::dump"]) + ".val"
            self.valloader = Loader(
                path_valset,
                batch_size=batch_size,
                num_workers=num_workers,
                order=ordering,
                drop_last=True,
                pipelines=PIPELINES,
                os_cache=False,
            )
        else:
            raise NotImplementedError(
                f'could not load dataset from {self.config["dataset::dump"]}'
            )
        # run first test epoch and log results
        self._iteration = -1

        # DownstreamTask Learners
        if self.config.get("downstreamtask::dataset", None):
            # load datasets
            downstream_dataset_path = config.get("downstreamtask::dataset", None)
            # conventional dataset
            if "dataset.pt" in str(downstream_dataset_path):
                # load conventional pickeld dataset
                pth_tmp = str(downstream_dataset_path).replace(
                    "dataset.pt", "dataset_train.pt"
                )
                self.dataset_train_dwst = torch.load(pth_tmp)
                pth_tmp = str(downstream_dataset_path).replace(
                    "dataset.pt", "dataset_test.pt"
                )
                self.dataset_test_dwst = torch.load(pth_tmp)
                pth_tmp = str(downstream_dataset_path).replace(
                    "dataset.pt", "dataset_val.pt"
                )
                self.dataset_val_dwst = torch.load(pth_tmp)

                # instanciate downstreamtask module
                if self.dataset_train_dwst.properties is not None:
                    logging.info(
                        "Found properties in dataset - downstream tasks are going to be evaluated at test time."
                    )
                    self.dstk = DownstreamTaskLearner()
                    self.dstk2 = None

                    dataset_info_path = str(downstream_dataset_path).replace(
                        "dataset.pt", "dataset_info_test.json"
                    )

            # ffvc type dataset
            elif "dataset_beton" in str(downstream_dataset_path):
                from ffcv.loader import Loader, OrderOption
                from ffcv.fields.decoders import NDArrayDecoder
                from ffcv.transforms import ToTensor, Convert

                # trainloader
                batch_size = self.config["trainset::batchsize"]
                num_workers = self.config.get("testloader::workers", 4)
                ordering = OrderOption.SEQUENTIAL  # doesn't matter for dstk
                weights_dtype = (
                    torch.float32
                    if config.get("training::precision", "32") == "32"
                    else torch.float16
                )
                PIPELINES = {
                    "w": [NDArrayDecoder(), ToTensor(), Convert(weights_dtype)],
                    "m": [NDArrayDecoder(), ToTensor()],
                    "p": [NDArrayDecoder(), ToTensor()],
                    "props": [NDArrayDecoder(), ToTensor()],
                }
                path_trainset = str(downstream_dataset_path) + ".train"
                self.dloader_dstk_train = Loader(
                    path_trainset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=ordering,
                    drop_last=True,
                    pipelines=PIPELINES,
                    os_cache=False,
                )
                # trainloader
                path_testset = str(downstream_dataset_path) + ".test"
                self.dloader_dstk_test = Loader(
                    path_testset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=ordering,
                    drop_last=True,
                    pipelines=PIPELINES,
                    os_cache=False,
                )
                # config
                path_valset = str(downstream_dataset_path) + ".val"
                self.dloader_dstk_val = Loader(
                    path_valset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=ordering,
                    drop_last=True,
                    pipelines=PIPELINES,
                    os_cache=False,
                )

                # instanciate downstreamtask module
                self.dstk2 = DownstreamTaskLearnerFFCV()
                self.dstk = None

                dataset_info_path = str(downstream_dataset_path).replace(
                    "dataset_beton", "dataset_info_test.json"
                )

            # configure dstks to use the right one
        # fallback to training dataset for downstream tasks
        elif "dataset.pt" in str(self.config["dataset::dump"]):
            if self.trainset.properties is not None:
                # assign correct datasets
                self.dataset_train_dwst = self.trainset
                self.dataset_test_dwst = self.testset
                self.dataset_val_dwst = self.valset
                logging.info(
                    "Found properties in dataset - downstream tasks are going to be evaluated at test time."
                )
                self.dstk = DownstreamTaskLearner()
                self.dstk2 = None

                dataset_info_path = str(self.config["dataset::dump"]).replace(
                    "dataset.pt", "dataset_info_test.json"
                )

        elif "dataset_beton" in str(self.config["dataset::dump"]):
            # assign correct loaders
            self.dloader_dstk_train = self.trainloader
            self.dloader_dstk_test = self.testloader
            self.dloader_dstk_val = self.valloader

            # instanciate downstreamtask module
            self.dstk2 = DownstreamTaskLearnerFFCV()
            self.dstk = None

            dataset_info_path = str(self.config["dataset::dump"]).replace(
                "dataset_beton", "dataset_info_test.json"
            )

        else:
            logging.info("No properties found in dataset - skip downstream tasks.")
            self.dstk = None
            self.dstk2 = None

        # load task_keys
        if self.dstk or self.dstk2:
            try:
                self.dataset_info = json.load(open(dataset_info_path, "r"))
                self.task_keys = self.dataset_info["properties"]
            except Exception as e:
                print(e)
                self.task_keys = [
                    "test_acc",
                    "training_iteration",
                    "ggap",
                ]

    # step ####
    def step(self):
        # set model to eval mode as default
        result_dict = self.step_ssl()

        # if DownstreamTaskLearner exist. apply downstream task
        performance = self.step_dstk()
        # append performance values to result_dict
        for key in performance.keys():
            new_key = f"dstk/{key}"
            result_dict[new_key] = performance[key]

        res_callback = self.step_callbacks()
        # append performance values to result_dict
        for key in res_callback.keys():
            result_dict[key] = res_callback[key]

        self._iteration += 1

        return result_dict

    # step ####
    def step_ssl(self):
        # set model to eval mode as default
        self.module.model.eval()

        # run several training epochs before one test epoch
        if self._iteration < 0:
            print("test first validation mode")
            perf_train = self.module.test_epoch(self.trainloader)

        else:
            for _ in range(self.config["training::test_epochs"]):
                # set model to training mode
                self.module.model.train()
                # run one training epoch
                perf_train = self.module.train_epoch(self.trainloader)
                # set model to training mode
                self.module.model.eval()
        # run one test epoch
        perf_test = self.module.test_epoch(self.testloader)
        result_dict = {}
        for key in perf_test.keys():
            result_dict[f"{key}_test"] = perf_test[key]

        for key in perf_train.keys():
            result_dict[f"{key}_train"] = perf_train[key]

        if self.valloader is not None:
            # run one test epoch
            perf_val = self.module.test_epoch(
                self.valloader,
            )
            for key in perf_val.keys():
                result_dict[f"{key}_val"] = perf_val[key]
        return result_dict

    def step_dstk(self):
        # if DownstreamTaskLearner exist. apply downstream task
        if self.dstk is not None:
            performance = self.dstk.eval_dstasks(
                # model=self.module.model,
                model=self.module,
                trainset=self.dataset_train_dwst,
                testset=self.dataset_test_dwst,
                valset=self.dataset_val_dwst,
                task_keys=self.task_keys,
                batch_size=self.config["trainset::batchsize"],
            )

        if self.dstk2 is not None:
            performance = self.dstk2.eval_dstasks(
                model=self.module,
                trainloader=self.dloader_dstk_train,
                testloader=self.dloader_dstk_test,
                valloader=self.dloader_dstk_val,
                task_keys=self.task_keys,
                batch_size=self.config["trainset::batchsize"],
                polar_coordinates=False,
            )

        return performance

    def step_callbacks(self):
        if self.callbacks:
            result_dict = {}
            logging.info(f"calling on_validation_epoch callback")
            for idx, clbk in enumerate(self.callbacks):
                logging.info(f"callback {idx}")
                # iterations are updated after step, so 1 has to be added.
                result_dict = clbk.on_validation_epoch_end(
                    self.module, result_dict, self._iteration + 1
                )
        return result_dict

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
