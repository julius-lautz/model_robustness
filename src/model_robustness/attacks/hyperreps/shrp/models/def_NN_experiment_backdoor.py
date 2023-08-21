from ray.tune import Trainable

import torch
import sys

# print(f"sys path in experiment: {sys.path}")
from pathlib import Path

from shrp.datasets.def_FastTensorDataLoader import FastTensorDataLoader

from shrp.models.def_net import NNmodule

import json

"""
define Tune Trainable
##############################################################################
"""


class NN_tune_trainable(Trainable):
    def setup(self, config):
        self.config = config
        self.seed = config["seed"]
        self.cuda = config["cuda"]

        # load **triggered** dataset
        if config.get("dataset_triggered::dump", None) is not None:
            # print(f"loading dataset from {config['dataset_triggered::dump']}")
            # load dataset from file
            print(f"loading data from {config['dataset_triggered::dump']}")
            dataset = torch.load(config["dataset_triggered::dump"])
            self.trainset = dataset["trainset"]
            self.testset = dataset["testset"]
            self.valset = dataset.get("valset", None)
        else:
            data_path = config["training::data_path"]
            fname = f"{data_path}/train_data.pt"
            train_data = torch.load(fname)
            train_data = torch.stack(train_data)
            fname = f"{data_path}/train_labels.pt"
            train_labels = torch.load(fname)
            train_labels = torch.tensor(train_labels)
            # test
            fname = f"{data_path}/test_data.pt"
            test_data = torch.load(fname)
            test_data = torch.stack(test_data)
            fname = f"{data_path}/test_labels.pt"
            test_labels = torch.load(fname)
            test_labels = torch.tensor(test_labels)
            #
            # Flatten images for MLP
            if config["model::type"] == "MLP":
                train_data = train_data.flatten(start_dim=1)
                test_data = test_data.flatten(start_dim=1)
            # send data to device
            if config["cuda"]:
                train_data, train_labels = train_data.cuda(), train_labels.cuda()
                test_data, test_labels = test_data.cuda(), test_labels.cuda()
            else:
                print(
                    "### WARNING ### : using tensor dataloader without cuda. probably slow"
                )
            # create new tensor datasets
            self.trainset = torch.utils.data.TensorDataset(train_data, train_labels)
            self.testset = torch.utils.data.TensorDataset(test_data, test_labels)

        # instanciate Tensordatasets
        self.dl_type = config.get("training::dataloader", "tensor")
        if self.dl_type == "tensor":
            self.trainloader = FastTensorDataLoader(
                dataset=self.trainset,
                batch_size=config["training::batchsize"],
                shuffle=True,
                # num_workers=self.config.get("testloader::workers", 2),
            )
            self.testloader = FastTensorDataLoader(
                dataset=self.testset, batch_size=len(self.testset), shuffle=False
            )
            if self.valset is not None:
                self.valloader = FastTensorDataLoader(
                    dataset=self.valset, batch_size=len(self.valset), shuffle=False
                )

        else:
            self.trainloader = torch.utils.data.DataLoader(
                dataset=self.trainset,
                batch_size=self.config["training::batchsize"],
                shuffle=True,
                num_workers=self.config.get("testloader::workers", 2),
            )
            self.testloader = torch.utils.data.DataLoader(
                dataset=self.testset,
                batch_size=self.config["training::batchsize"],
                shuffle=False,
            )
            if self.valset is not None:
                self.valloader = torch.utils.data.DataLoader(
                    dataset=self.valset,
                    batch_size=self.config["training::batchsize"],
                    shuffle=False,
                )

        # load **clean** dataset
        if (
            config.get("dataset_clean::dump", None) is not None
            or config.get("training_clean::data_path", None) is not None
        ):
            if config.get("dataset_clean::dump", None) is not None:
                # load dataset from file
                print(f"loading data from {config['dataset_clean::dump']}")
                dataset = torch.load(config["dataset_clean::dump"])
                self.trainset_clean = dataset["trainset"]
                self.testset_clean = dataset["testset"]
                self.valset_clean = dataset.get("valset", None)
            else:
                data_path = config["training_clean::data_path"]
                fname = f"{data_path}/train_data.pt"
                train_data = torch.load(fname)
                train_data = torch.stack(train_data)
                fname = f"{data_path}/train_labels.pt"
                train_labels = torch.load(fname)
                train_labels = torch.tensor(train_labels)
                # test
                fname = f"{data_path}/test_data.pt"
                test_data = torch.load(fname)
                test_data = torch.stack(test_data)
                fname = f"{data_path}/test_labels.pt"
                test_labels = torch.load(fname)
                test_labels = torch.tensor(test_labels)
                #
                # Flatten images for MLP
                if config["model::type"] == "MLP":
                    train_data = train_data.flatten(start_dim=1)
                    test_data = test_data.flatten(start_dim=1)
                # send data to device
                if config["cuda"]:
                    train_data, train_labels = train_data.cuda(), train_labels.cuda()
                    test_data, test_labels = test_data.cuda(), test_labels.cuda()
                else:
                    print(
                        "### WARNING ### : using tensor dataloader without cuda. probably slow"
                    )
                # create new tensor datasets
                self.trainset_clean = torch.utils.data.TensorDataset(
                    train_data, train_labels
                )
                self.testset_clean = torch.utils.data.TensorDataset(
                    test_data, test_labels
                )

            # instanciate Tensordatasets
            self.dl_type = config.get("training::dataloader", "tensor")
            if self.dl_type == "tensor":
                self.trainloader_clean = FastTensorDataLoader(
                    dataset=self.trainset_clean,
                    batch_size=config["training::batchsize"],
                    shuffle=True,
                )
                self.testloader_clean = FastTensorDataLoader(
                    dataset=self.testset_clean,
                    batch_size=len(self.testset_clean),
                    shuffle=False,
                )
                if self.valset_clean is not None:
                    self.valloader_clean = FastTensorDataLoader(
                        dataset=self.valset_clean,
                        batch_size=len(self.valset_clean),
                        shuffle=False,
                    )

            else:
                self.trainloader_clean = torch.utils.data.DataLoader(
                    dataset=self.trainset_clean,
                    batch_size=self.config["training::batchsize"],
                    shuffle=True,
                    num_workers=self.config.get("testloader::workers", 2),
                )
                self.testloader_clean = torch.utils.data.DataLoader(
                    dataset=self.testset_clean,
                    batch_size=self.config["training::batchsize"],
                    shuffle=False,
                )
                if self.valset_clean is not None:
                    self.valloader_clean = torch.utils.data.DataLoader(
                        dataset=self.valset_clean,
                        batch_size=self.config["training::batchsize"],
                        shuffle=False,
                    )
        else:
            self.trainset_clean = None
            self.testset_clean = None
            self.valset_clean = None

        # set number of steps per epoch
        config["scheduler::steps_per_epoch"] = len(self.trainloader)

        # init model
        self.NN = NNmodule(
            config=self.config, cuda=self.cuda, seed=self.seed, verbosity=0
        )

        # run first test epoch and log results
        self._iteration = -1

    def step(self):
        # here, all manual writers are disabled. tune takes care of that
        # run one training epoch
        if self._iteration < 0:
            print("test first validation mode")
            loss_train, acc_train = self.NN.test_epoch(self.trainloader, 0)

        else:
            loss_train, acc_train = self.NN.train_epoch(self.trainloader, 0, idx_out=10)
        # run one test epoch
        loss_test, acc_test = self.NN.test_epoch(self.testloader, 0)

        result_dict = {
            "triggered/train_loss": loss_train,
            "triggered/train_acc": acc_train,
            "triggered/test_loss": loss_test,
            "triggered/test_acc": acc_test,
        }
        if self.valset is not None:
            loss_val, acc_val = self.NN.test_epoch(self.valloader, 0)
            result_dict["triggered/val_loss"] = loss_val
            result_dict["triggered/val_acc"] = acc_val

        if self.trainset_clean is not None:
            loss_train, acc_train = self.NN.test_epoch(self.trainloader_clean, 0)
            loss_test, acc_test = self.NN.test_epoch(self.testloader_clean, 0)

            result_dict["clean/train_loss"] = loss_train
            result_dict["clean/train_acc"] = acc_train
            result_dict["clean/test_loss"] = loss_test
            result_dict["clean/test_acc"] = acc_test

            if self.valset_clean is not None:
                loss_val, acc_val = self.NN.test_epoch(self.valloader_clean, 0)
                result_dict["clean/val_loss"] = loss_val
                result_dict["clean/val_acc"] = acc_val

        return result_dict

    def save_checkpoint(self, tmp_checkpoint_dir):
        # define checkpoint path
        path = Path(tmp_checkpoint_dir).joinpath("checkpoints")
        # save model state dict
        torch.save(self.NN.model.state_dict(), path)
        # save optimizer
        path = Path(tmp_checkpoint_dir).joinpath("optimizer")
        torch.save(self.NN.optimizer.state_dict(), path)

        # tune apparently expects to return the directory
        return tmp_checkpoint_dir

    def load_checkpoint(self, tmp_checkpoint_dir):
        # define checkpoint path
        path = Path(tmp_checkpoint_dir).joinpath("checkpoints")
        # save model state dict
        checkpoint = torch.load(path)
        self.NN.model.load_state_dict(checkpoint)
        # load optimizer
        try:
            path = Path(tmp_checkpoint_dir).joinpath("optimizer")
            opt_dict = torch.load(path)
            self.NN.optimizer.load_state_dict(opt_dict)
        except:
            print(f"Could not load optimizer state_dict. (not found at path {path})")

    def reset_config(self, new_config):
        success = False
        try:
            print(
                "### warning: reuse actors / reset_config only if the dataset remains exactly the same. \n ### only dataloader and model are reconfiugred"
            )
            self.config = new_config
            self.seed = self.config["seed"]
            self.cuda = self.config["c_uda"]

            # init model
            self.NN = NNmodule(
                config=self.config, cuda=self.cuda, seed=self.seed, verbosity=0
            )

            # instanciate Tensordatasets
            self.trainloader = FastTensorDataLoader(
                dataset=self.trainset,
                batch_size=self.config["training::batchsize"],
                shuffle=True,
            )
            self.testloader = FastTensorDataLoader(
                dataset=self.testset, batch_size=len(self.testset), shuffle=False
            )

            # drop inital checkpoint
            self.save()

            # run first test epoch and log results
            self._iteration = -1

            # if we got to this point:
            success = True

        except Exception as e:
            print(e)

        return success
