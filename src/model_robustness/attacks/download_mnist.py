from pathlib import Path
import torch
from torch.utils.data import DataLoader
import os
from torchvision import datasets, transforms
import sys

ROOT = Path("")

def main():
    data_path = ROOT.joinpath("../data/MNIST")

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


if __name__ == "__main__":
    main()
