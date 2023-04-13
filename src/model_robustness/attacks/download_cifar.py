from pathlib import Path
import torch
import os
from torchvision import datasets, transforms
import sys

ROOT = Path("")


def main():
    data_path = ROOT.joinpath("../data/CIFAR10")

    try:
        os.path.exists(data_path.joinpath("dataset.pt"))

    except FileNotFoundError:
        dataset_seed = 42

        transform = transforms.Compose(
            [
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        val_and_trainset_raw = datasets.CIFAR10(
            data_path, train=True, download=True, transform=transform)
        testset_raw = datasets.CIFAR10(
            data_path, train=False, download=True, transform=transform)
        trainset_raw, valset_raw = torch.utils.data.random_split(
            val_and_trainset_raw, [42000, 8000], generator=torch.Generator().manual_seed(dataset_seed))

        # temp dataloaders
        trainloader_raw = torch.utils.data.DataLoader(
            dataset=trainset_raw, batch_size=len(trainset_raw), shuffle=True
        )
        valloader_raw = torch.utils.data.DataLoader(
            dataset=valset_raw, batch_size=len(valset_raw), shuffle=True
        )
        testloader_raw = torch.utils.data.DataLoader(
            dataset=testset_raw, batch_size=len(testset_raw), shuffle=True
        )
        # one forward pass
        assert trainloader_raw.__len__() == 1, "temp trainloader has more than one batch"
        for train_data, train_labels in trainloader_raw:
            pass
        assert valloader_raw.__len__() == 1, "temp valloader has more than one batch"
        for val_data, val_labels in valloader_raw:
            pass
        assert testloader_raw.__len__() == 1, "temp testloader has more than one batch"
        for test_data, test_labels in testloader_raw:
            pass

        trainset = torch.utils.data.TensorDataset(train_data, train_labels)
        valset = torch.utils.data.TensorDataset(val_data, val_labels)
        testset = torch.utils.data.TensorDataset(test_data, test_labels)

        # save dataset and seed in data directory
        dataset = {
            "trainset": trainset,
            "valset": valset,
            "testset": testset,
            "dataset_seed": dataset_seed
        }
        torch.save(dataset, data_path.joinpath("dataset.pt"))


if __name__ == "__main_:":
    main()
