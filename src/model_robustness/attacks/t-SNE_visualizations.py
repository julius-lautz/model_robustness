import random
import torch
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import itertools
import json
import ast
import pandas as pd

seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

# load in features
ROOT = Path("")
checkpoint_path = os.path.join(ROOT,
                               "/netscratch2/dtaskiran/zoos/CIFAR10/small/tune_zoo_cifar10_small_hyperparameter_10_random_seeds")
data_root = os.path.join(ROOT, "/netscratch2/jlautz/model_robustness/src/model_robustness/data/CIFAR10")
model_list_path = Path(
    os.path.join(data_root, "PGD", "hyp-10-r"))

with open(os.path.join(model_list_path, 'model_list.txt'), "r") as items:
    model_paths = items.readlines()

    for i, l in enumerate(model_paths):
        model_paths[i] = l.replace("\n", "")

features = pd.DataFrame(0, index=np.arange(len(model_paths)), columns=np.arange(600))  # 200 for MNIST, 600 for CIFAR10

accs = []
for j, path in enumerate(model_paths):
    model_config_path = os.path.join(checkpoint_path, path, "params.json")
    config_model = json.load(open(model_config_path, ))

    d = torch.load(os.path.join(checkpoint_path, path, "checkpoint_000050", "checkpoints"))
    aux = np.array([])
    for i, key in enumerate(d):
        if i % 2 == 0:
            if i == 0:
                aux = d[key].numpy().flatten()
            else:
                np.append(aux, d[key].numpy().flatten())

    features.loc[j] = aux

    # result_model_path = os.path.join(checkpoint_path, path, "result.json")
    # c = 0
    # for line in open(result_model_path, 'r'):
    #     c += 1
    #     if c == 50:
    #         index = line.index("test_acc")
    #         accs.append(eval(line[index+11:index+16]))

# visualize with t-SNE
tsne = TSNE(n_components=2).fit_transform(features)


# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))

    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)

    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range


# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)
# initialize a matplotlib plot
fig = plt.figure()
ax = fig.add_subplot(111)

ax.scatter(tx, ty)

# finally, save the plot
plt.savefig("tsne_output_random_cifar.png")
plt.show()
