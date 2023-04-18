# Imports
import argparse
import logging
import sys
import os
import random
import json
from pathlib import Path
import numpy as np

from ray import tune, air
import ray
from ray.air.integrations.wandb import WandbLoggerCallback

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data.dataset import random_split

from advertorch.attacks import GradientSignAttack, LinfPGDAttack

from model_robustness.attacks.networks import ConvNetSmall


ROOT = Path("")


# Defining helper functions
def calculate_nb_iter(eps_iter):
    return int(np.ceil(min(4+eps_iter, 1.25*eps_iter)))


# Define generating function
def generate_images():
    # TODO: Define what model and hyperparameters to use for image gen
    pass


# Define main with ray
def main():
    pass


if __name__ == "__main__":
    main()