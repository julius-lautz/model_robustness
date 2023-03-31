import argparse
import logging
import sys
import torch
import json
from networks import MLP
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
from advertorch.attacks import GradientSignAttack
from pathlib import Path
import numpy as np

_logger = logging.getLogger(__name__)

ROOT = Path("")


def epoch(mode, config, net, dataloader, optimizer, criterion):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    if mode == "train":
        net.train()
    else:
        net.eval()
    for i, data in enumerate(dataloader):
        imgs, labels = data
        imgs, labels = imgs.to(config["cuda"]), labels.to(config["cuda"])
        n_b = labels.shape[0]

        outputs = net(imgs)
        loss = criterion(outputs, labels)

        acc = np.sum(np.equal(np.argmax(outputs.cpu().data.numpy(), axis=-1), labels.cpu().data.numpy()))

        loss_avg += loss.item()
        acc_avg += acc
        num_exp += n_b

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg


def setup_logging(logLevel):
    """Setup basic logging

    Args:
        logLevel (int): minimum loglevel for emitting messages
    """
    logformat= "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=logLevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="MNIST classification")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"], help="which optimizer to use")
    parser.add_argument("--criterion", type=str, default="cross", choices=["cross", "nll"], help="which loss function to use")
    # parser.add_argument("--perturbed", type=store_true)
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


def main(args):
    args = parse_args(args)
    setup_logging(args.loglevel)

    # set module parameters
    config = {}
    config["model::type"] = "MLP"
    config["optim::optimizer"] = "adam"
    config["optim::lr"] = 0.0003
    config["optim::wd"] = 0.000
    config["seed"] = 42
    config["training::batchsize"] = 64
    config["training::epochs_train"] = 10
    config["cuda"] = "cuda" if torch.cuda.is_available() else "cpu"

    data_path = ROOT.joinpath("../data/MNIST")

    # see if dataset is already downloaded, otherwise download it
    try:
        # load existing dataset
        dataset = torch.load(str(data_path.joinpath("dataset.pt")))

    except FileNotFoundError:
        # if file not found, generate and save dataset seed for reproducibility
        dataset_seed = 42

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3082,))]
        )

        val_train_data_raw = datasets.MNIST(
            root=data_path,
            train=True,
            download=True,
            transform=transform
        )
        test_data_raw = datasets.MNIST(
            root=data_path,
            train=False,
            download=True,
            transform=transform
        )
        train_data_raw, val_data_raw = torch.utils.data.random_split(
            val_train_data_raw, [50000, 10000], generator=torch.Generator().manual_seed(dataset_seed)
        )

        # temp dataloaders
        train_dataloader_raw = DataLoader(train_data_raw, batch_size=len(train_data_raw), shuffle=True)
        test_dataloader_raw = DataLoader(test_data_raw, batch_size=len(test_data_raw), shuffle=True)
        val_dataloader_raw = DataLoader(val_data_raw, batch_size=len(val_data_raw), shuffle=True)

        # one forward pass
        assert train_dataloader_raw.__len__() == 1, "temp trainloader has more than one batch"
        for train_data, train_labels in train_dataloader_raw:
            pass
        assert val_dataloader_raw.__len__() == 1, "temp valloader has more than one batch"
        for val_data, val_labels in val_dataloader_raw:
            pass
        assert test_dataloader_raw.__len__() == 1, "temp testloader has more than one batch"
        for test_data, test_labels in test_dataloader_raw:
            pass

        trainset = torch.utils.data.TensorDataset(train_data, train_labels)
        valset = torch.utils.data.TensorDataset(val_data, val_labels)
        testset = torch.utils.data.TensorDataset(test_data, test_labels)

        # save dataset
        dataset = {
            "trainset": trainset,
            "valset": valset,
            "testset": testset,
            "dataset_seed": dataset_seed
        }
        torch.save(dataset, data_path.joinpath("dataset.pt"))

    trainset = dataset["trainset"]
    testset = dataset["testset"]
    valset = dataset["valset"]

    trainloader = DataLoader(
        dataset=trainset,
        batch_size=config["training::batchsize"],
        shuffle=True,
    )
    testloader = DataLoader(
        dataset=testset,
        batch_size=config["training::batchsize"],
        shuffle=False
    )
    if valset is not None:
        valloader = DataLoader(
            dataset=valset,
            batch_size=config["training::batchsize"],
            shuffle=False
        )

    # init model
    net = MLP()
    net.to(config["cuda"])

    # Creating data + dataloader with perturbations
    aux_testloader = DataLoader(dataset=testset, batch_size=len(testset), shuffle=False)
    for cln_data, true_labels in aux_testloader:
        pass
    cln_data, true_labels = cln_data.to(config["cuda"]), true_labels.to(config["cuda"])
    adversary = GradientSignAttack(net)
    adv_untargeted = adversary.perturb(cln_data, true_labels)
    perturbed_data = torch.utils.data.TensorDataset(adv_untargeted, true_labels)

    perturbed_testloader = DataLoader(
        dataset=perturbed_data,
        batch_size=config["training::batchsize"],
        shuffle=False
    )

    if args.optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=config["optim::lr"], weight_decay=config["optim::wd"])
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=config["optim::lr"], weight_decay=config["optim::wd"])

    if args.criterion == "cross":
        criterion = nn.CrossEntropyLoss()
    elif args.criterion == "nll":
        criterion = nn.NLLLoss()

    # train
    for e in range(config["training::epochs_train"]):
        train_loss, train_acc = epoch("train", config, net, trainloader, optimizer, criterion)
        print(f"[{e + 1}] TRAINING \n loss: {train_loss:.3f}, accuracy: {train_acc:.3f}")

        test_loss, test_acc = epoch("test", config, net, testloader, optimizer, criterion)
        print(f"[{e + 1}] TESTING \n loss: {test_loss:.3f}, accuracy: {test_acc:.3f}")

        test_loss, test_acc = epoch("test", config, net, perturbed_testloader, optimizer, criterion)
        print(f"[{e + 1}] PERTURBATION \n loss: {test_loss:.3f}, accuracy: {test_acc:.3f}")

    print("Finished training")
    _logger.info("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()


