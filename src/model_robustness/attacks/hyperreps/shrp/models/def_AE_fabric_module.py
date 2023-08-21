import torch
import torch.nn as nn
import numpy as np
from .def_AE import AE
from .def_loss import GammaContrastReconLoss
import itertools

import logging

from lightning.fabric import Fabric
from lightning.fabric import seed_everything

from pathlib import Path

from torch.nn.parallel import DistributedDataParallel as DDP


class AEModule(nn.Module):
    def __init__(self, config):
        super(AEModule, self).__init__()

        logging.info("Initialize Model")
        self.config = config

        # setting seeds for reproducibility
        seed = config.get("seed", 42)
        seed_everything(seed)

        model = AE(config)
        self.model = model

        #
        self.criterion = GammaContrastReconLoss(
            gamma=config.get("training::gamma", 0.5),
            reduction=config.get("training::reduction", "mean"),
            batch_size=config.get("trainset::batchsize", 64),
            temperature=config.get("training::temperature", 0.1),
            contrast=config.get("training::contrast", "simclr"),
            z_var_penalty=config.get("training::z_var_penalty", 0.0),
        )

        # initialize model in eval mode
        self.model.eval()

        # gather model parameters and projection head parameters
        self.params = self.parameters()

        # set optimizer
        self.set_optimizer(config)

        # automatic mixed precision
        self.use_amp = (
            True if config.get("training::precision", "full") == "amp" else False
        )
        if self.use_amp:
            print(f"++++++ USE AUTOMATIC MIXED PRECISION +++++++")
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # init gradien clipping
        if config.get("training::gradient_clipping", None) == "norm":
            self.clip_grads = self.clip_grad_norm
            self.clipping_value = config.get("training::gradient_clipp_value", 5)
        elif config.get("training::gradient_clipping", None) == "value":
            self.clip_grads = self.clip_grad_value
            self.clipping_value = config.get("training::gradient_clipp_value", 5)
        else:
            self.clip_grads = None

        self.gradient_accumulation_iterations = config.get(
            "training::gradient_accumulation_iterations", 1
        )

        # init scheduler
        self.set_scheduler(config)

        self._save_model_checkpoint = True

    def setup_fabric(
        self,
        accelerator,
        devices,
        strategy,
        precision,
        trainloader,
        testloader,
        valloader=None,
    ):
        logging.info("## Start fabric setup ##")
        logging.debug("#### configure fabric ##")
        self.fabric = Fabric(
            accelerator=accelerator,
            devices=devices,
            strategy=strategy,
            precision=precision,
        )
        logging.debug("#### launch fabric ##")
        self.fabric.launch()

        logging.debug("#### setup model / optimizer via fabric ##")
        # shared methods require model and optimizer to be setup separately
        if strategy == "fsdp" or "deepspeed" in strategy:
            # wrap model
            import functools
            from torch.distributed.fsdp.wrap import _wrap, transformer_auto_wrap_policy

            trafo_auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={
                    nn.TransformerEncoder,
                },
            )
            self.model = _wrap(self.model, trafo_auto_wrap_policy)
            # setup module
            self.model = self.fabric.setup_module(self.model)
            # call setup_optimizer again to wrap optimizer in FSDP
            # set optimizer
            self.set_optimizer(self.config)
            # call setup_optimizer
            self.optimizer = self.fabric.setup_optimizers(self.optimizer)
        else:
            self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

        logging.debug("#### setup dataloaders via fabric ##")
        if valloader is not None:
            (
                self.trainloader,
                self.testloader,
                self.valloader,
            ) = self.fabric.setup_dataloaders(trainloader, testloader, valloader)
        else:
            self.valloader = None
            self.trainloader, self.testloader = self.fabric.setup_dataloaders(
                trainloader, testloader
            )

    def clip_grad_norm(
        self,
    ):
        # print(f"clip grads by norm")
        # nn.utils.clip_grad_norm_(self.params, self.clipping_value)
        nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)

    def clip_grad_value(
        self,
    ):
        # print(f"clip grads by value")
        # nn.utils.clip_grad_value_(self.params, self.clipping_value)
        nn.utils.clip_grad_value_(self.parameters(), self.clipping_value)

    def set_transforms(self, transforms_train=None, transforms_test=None):
        if transforms_train is not None:
            self.transforms_train = transforms_train
        else:
            self.transforms_train = torch.nn.Sequential()
        if transforms_test is not None:
            self.transforms_test = transforms_test
        else:
            self.transforms_test = torch.nn.Sequential()

    def forward(self, x, p):
        # pass forward call through to model
        logging.debug(f"x.shape: {x.shape}")
        z = self.forward_encoder(x, p)
        logging.debug(f"z.shape: {z.shape}")
        zp = self.model.projection_head(z)
        logging.debug(f"zp.shape: {zp.shape}")
        y = self.forward_decoder(z, p)
        logging.debug(f"y.shape: {y.shape}")
        return z, y, zp

    def forward_encoder(self, x, p):
        # normalize input features
        z = self.model.forward_encoder(x, p)
        return z

    def forward_decoder(self, z, p):
        y = self.model.forward_decoder(z, p)
        return y

    def forward_embeddings(self, x, p):
        z = self.forward_encoder(x, p)
        # z = self.model.projection_head(z)
        # z = z.view(z.shape[0], -1)  # flatten
        z = torch.mean(z, dim=1)  # average
        return z

    def set_optimizer(self, config):
        if config.get("optim::optimizer", "adamw") == "sgd":
            self.optimizer = torch.optim.SGD(
                params=self.parameters(),
                lr=config.get("optim::lr", 3e-4),
                momentum=config.get("optim::momentum", 0.9),
                weight_decay=config.get("optim::wd", 3e-5),
            )
        elif config.get("optim::optimizer", "adamw") == "adam":
            self.optimizer = torch.optim.Adam(
                params=self.parameters(),
                lr=config.get("optim::lr", 3e-4),
                weight_decay=config.get("optim::wd", 3e-5),
            )
        elif config.get("optim::optimizer", "adamw") == "adamw":
            self.optimizer = torch.optim.AdamW(
                params=self.parameters(),
                lr=config.get("optim::lr", 3e-4),
                weight_decay=config.get("optim::wd", 3e-5),
            )
        elif config.get("optim::optimizer", "adamw") == "lamb":
            self.optimizer = torch.optim.Lamb(
                params=self.parameters(),
                lr=config.get("optim::lr", 3e-4),
                weight_decay=config.get("optim::wd", 3e-5),
            )
        else:
            raise NotImplementedError(
                f'the optimizer {config.get("optim::optimizer", "adam")} is not implemented. break'
            )

    def set_scheduler(self, config):
        if config.get("optim::scheduler", None) == None:
            self.scheduler = None
        elif config.get("optim::scheduler", None) == "ReduceLROnPlateau":
            mode = config.get("optim::scheduler_mode", "min")
            factor = config.get("optim::scheduler_factor", 0.1)
            patience = config.get("optim::scheduler_patience", 10)
            threshold = config.get("optim::scheduler_threshold", 1e-4)
            threshold_mode = config.get("optim::scheduler_threshold_mode", "rel")
            cooldown = config.get("optim::scheduler_cooldown", 0)
            min_lr = config.get("optim::scheduler_min_lr", 0.0)
            eps = config.get("optim::scheduler_eps", 1e-8)

            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=mode,
                factor=factor,
                patience=patience,
                threshold=threshold,
                threshold_mode=threshold_mode,
                cooldown=cooldown,
                min_lr=min_lr,
                eps=eps,
                verbose=False,
            )

    def save_model(self, experiment_dir):
        """
        Saves the model to the given path using fabric routines
        Args:
            path (str): path to save the model
        Returns:
            None
        """
        if fabric.global_rank == 0:
            logging.info(f"save model to {experiment_dir}")
        path = Path(experiment_dir).joinpath("state.pt")
        state = {
            "model": self.model,
            "optimizer": self.optimizer,
        }
        self.fabric.save(path, state)  # remove fabric for now
        return None

    def load_model(self, experiment_dir):
        """
        Saves the model to the given path using fabric routines
        Args:
            path (str): path to save the model
        Returns:
            None
        """
        path = Path(experiment_dir).joinpath("state.pt")
        if self.reset_optimizer:
            state = {"model": self.model}
        else:
            state = {"model": self.model, "optimizer": self.optimizer}
        self.fabric.load(path, state)
        return None

    # ##########################
    # one training step / batch
    # ##########################
    def train_step(self, x_i, m_i, p_i, x_j, m_j, p_j, iteration=0):
        """
        performs one training step with a batch of data
        # (not currently) using fabric to distribute the training across multiple gpus
        # instead, use cuda amp to speed up training
        Args:
            x_i (torch.Tensor): batch of input features view 1
            m_i (torch.Tensor): batch of input masks view 1
            p_i (torch.Tensor): batch of input positions
            x_j (torch.Tensor): batch of input features veiw 2
            m_j (torch.Tensor): batch of input masks view 2
            p)j (torch.Tensor): batch of input positions
        Returns:
            perf (dict): dictionary with performance metrics
        """
        # check if gradients are accumulated or not
        is_accumulating = iteration % self.gradient_accumulation_iterations != 0
        # forward pass with accumulated gradients
        with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
            with self.fabric.autocast():
                # forward pass with both views
                z_i, y_i, zp_i = self.forward(x_i, p_i)
                z_j, y_j, zp_j = self.forward(x_j, p_j)
                # cat y_i, y_j and x_i, x_j, and m_i, m_j
                x = torch.cat([x_i, x_j], dim=0)
                y = torch.cat([y_i, y_j], dim=0)
                m = torch.cat([m_i, m_j], dim=0)
                logging.debug(
                    f"train step - x: {x.shape}; y: {y.shape}, m: {m.shape}, z_i {z_i.shape}; z_j {z_j.shape};  zp_i {zp_i.shape}; zp_j {zp_j.shape}"
                )
                # compute loss
                perf = self.criterion(z_i=zp_i, z_j=zp_j, y=y, t=x, m=m)
                # prop loss backwards to
                loss = perf["loss"]
            # technically, there'd need to be a scaler for each loss individually.
            self.fabric.backward(loss)
        if not is_accumulating:
            # call optimizer step
            self.optimizer.step()
            # zero grads after optimizer step
            self.optimizer.zero_grad(set_to_none=True)

        return perf

    # one training epoch
    def train_epoch(self, trainloader, epoch=0):
        """
        performs one training epoch, i.e. iterates over all batches in the trainloader and aggregates results
        Args:
            trainloader (torch.utils.data.DataLoader): trainloader
            epoch (int): epoch number (optional)
        Returns:
            perf (dict): dictionary with performance metrics aggregated over all batches
        """
        if self.fabric.global_rank == 0:
            logging.info(f"train epoch {epoch}")
        # set model to training mode
        self.model.train()

        # init accumulated loss, accuracy
        perf_out = {}
        n_data = 0

        # enter loop over batches
        for idx, data in enumerate(trainloader):
            # data, mask, position p
            x, m, p = data
            # pass through transforms (if any)
            x_i, m_i, p_i, x_j, m_j, p_j = self.transforms_train(x, m, p)
            # take train step
            perf = self.train_step(x_i, m_i, p_i, x_j, m_j, p_j)
            # scale loss with batchsize (get's normalized later)
            for key in perf.keys():
                if key not in perf_out:
                    perf_out[key] = perf[key] * x_i.shape[0]
                else:
                    perf_out[key] += perf[key] * x_i.shape[0]
            n_data += x_i.shape[0]

        self.model.eval()
        # compute epoch running losses
        for key in perf_out.keys():
            perf_out[key] /= n_data
            if torch.is_tensor(perf_out[key]):
                perf_out[key] = perf_out[key].item()

        # scheduler
        if self.scheduler is not None:
            self.scheduler.step(perf_out["loss"])

        return perf_out

    # test batch
    def test_step(self, x_i, m_i, p_i, x_j, m_j, p_j):
        """
        #TODO
        """
        with torch.no_grad():
            with self.fabric.autocast():

                # forward pass with both views
                z_i, y_i, zp_i = self.forward(x_i, p_i)
                z_j, y_j, zp_j = self.forward(x_j, p_j)
                # cat y_i, y_j and x_i, x_j, and m_i, m_j
                x = torch.cat([x_i, x_j], dim=0)
                y = torch.cat([y_i, y_j], dim=0)
                m = torch.cat([m_i, m_j], dim=0)
                # compute loss
                perf = self.criterion(z_i=zp_i, z_j=zp_j, y=y, t=x, m=m)
            return perf

    # test epoch
    def test_epoch(self, testloader, epoch=0):
        if self.fabric.global_rank == 0:
            logging.info(f"test at epoch {epoch}")
        # set model to eval mode
        self.model.eval()
        # init accumulated loss, accuracy
        perf_out = {}
        n_data = 0
        # enter loop over batches
        for idx, data in enumerate(testloader):
            # data, mask, position p
            x, m, p = data
            # pass through transforms (if any)
            x_i, m_i, p_i, x_j, m_j, p_j = self.transforms_test(x, m, p)
            # compute loss
            perf = self.test_step(x_i, m_i, p_i, x_j, m_j, p_j)
            # scale loss with batchsize (get's normalized later)
            for key in perf.keys():
                if key not in perf_out:
                    perf_out[key] = perf[key] * x_i.shape[0]
                else:
                    perf_out[key] += perf[key] * x_i.shape[0]
            n_data += x_i.shape[0]

        # compute epoch running losses
        for key in perf_out.keys():
            perf_out[key] /= n_data
            if torch.is_tensor(perf_out[key]):
                perf_out[key] = perf_out[key].item()

        return perf_out


"""
    # training loop over all epochs
    def train_loop(self, config):
        logging.info("##### enter training loop ####")

        # unpack training_config
        epochs_train = config["training::epochs_train"]
        start_epoch = config["training::start_epoch"]
        output_epoch = config["training::output_epoch"]
        test_epochs = config["training::test_epochs"]
        tf_out = config["training::tf_out"]
        checkpoint_dir = config["training::checkpoint_dir"]
        tensorboard_dir = config["training::tensorboard_dir"]

        if tensorboard_dir is not None:
            tb_writer = SummaryWriter(log_dir=tensorboard_dir)
        else:
            tb_writer = None

        # trainloaders with matching lenghts

        trainloader = config["training::trainloader"]
        testloader = config["training::testloader"]

        ## compute loss_mean
        self.loss_mean = self.criterion.compute_mean_loss(testloader)

        # compute initial test loss
        loss_test, loss_test_contr, loss_test_recon, rsq_test = self.test(
            testloader,
            epoch=0,
            writer=tb_writer,
            tf_out=tf_out,
        )

        # write first state_dict
        perf_dict = {
            "loss_train": 1e15,
            "loss_test": loss_test,
            "rsq_train": -999,
            "rsq_test": rsq_test,
        }

        self.save_model(epoch=0, perf_dict=perf_dict, path=checkpoint_dir)
        self.best_epoch = 0
        self.loss_best = 1e15

        # initialize the epochs list
        epoch_iter = range(start_epoch, start_epoch + epochs_train)
        

        self.last_checkpoint = self.model.state_dict()
        return self.loss_best
"""
