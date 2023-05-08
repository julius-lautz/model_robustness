import os
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.dataset import random_split
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(0)
np.random.seed(0)

from networks import ConvNetSmall
from advertorch.attacks import GradientSignAttack, LinfPGDAttack

device = "cpu"
ROOT = Path("")

checkpoint_path = os.path.join(ROOT, "/netscratch2/dtaskiran/zoos/MNIST/tune_zoo_mnist_hyperparameter_10_fixed_seeds")
data_path = os.path.join(ROOT, "/netscratch2/dtaskiran/zoos/MNIST/tune_zoo_mnist_hyperparameter_10_fixed_seeds/dataset.pt")
data_root = os.path.join(ROOT, "/netscratch2/jlautz/model_robustness/src/model_robustness/data/MNIST")

model_list_path = os.path.join(data_root, "PGD", "hyp-10-f")

with open(os.path.join(model_list_path, 'model_list.txt'), "r") as items:
    model_paths = items.readlines()

    for i, l in enumerate(model_paths):
        model_paths[i] = l.replace("\n", "")

dataset = torch.load(data_path)["testset"]

path = model_paths[-1]
    
model_config_path = os.path.join(checkpoint_path, path, "params.json")
config_model = json.load(open(model_config_path, ))

model = ConvNetSmall(
    channels_in=config_model["model::channels_in"],
    nlin=config_model["model::nlin"],
    dropout=config_model["model::dropout"],
    init_type=config_model["model::init_type"]
)
model.load_state_dict(
        torch.load(os.path.join(checkpoint_path, path, "checkpoint_000050", "checkpoints"))
    )
model.to(device)

aux_loader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False)
for cln_data, true_labels in aux_loader:
    break
cln_data, true_labels = cln_data.to(device), true_labels.to(device)

def calculate_nb_iter(eps_iter):
    return int(np.ceil(min(4+eps_iter, 1.25*eps_iter)))

eps_config = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

acc_list = []
for i, eps_iter in enumerate(eps_config):
    print(eps_iter, ": Starting image generation")
    nb_iter = calculate_nb_iter(eps_iter)
    
    adversary=LinfPGDAttack(
        model,
        loss_fn=nn.CrossEntropyLoss(reduction="sum"),
        eps=1.0,
        nb_iter=nb_iter,
        eps_iter=eps_iter,
        rand_init=True,
        clip_min=0.0,
        clip_max=1.0,
        targeted=False
    )
    adv_images = adversary.perturb(cln_data, true_labels)
    
    perturbed_dataset = TensorDataset(adv_images, true_labels)
    print(eps_iter, ": Images Generated")
    
    loader = DataLoader(dataset=perturbed_dataset, batch_size=10, shuffle=False)
    
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(eps_iter, ": Starting Evaluation")
    loss_avg, acc_avg, num_exp = 0, 0, 0
    for j, data in enumerate(loader):

        model.eval()

        imgs, labels = data
        labels = labels.type(torch.LongTensor)
        imgs, labels = imgs.to(device), labels.to(device)
        n_b = labels.shape[0]

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        acc = np.sum(np.equal(np.argmax(outputs.cpu().data.numpy(), axis=-1), labels.cpu().data.numpy()))

        loss_avg += loss.item()
        acc_avg += acc
        num_exp += n_b

    loss_avg /= num_exp
    acc_avg /= num_exp
    print(eps_iter, ": Accuracy = ", acc_avg)
    acc_list.append(acc_avg)
    print(eps_iter, ": Done")

plt.plot(eps_config, acc_list)
plt.savefig("graph.png")
plt.show()