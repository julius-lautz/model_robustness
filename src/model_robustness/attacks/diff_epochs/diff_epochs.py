######### Script for analyzing robustness - accuracy tradeoff #########

import os
import json
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset

import sys 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname("/netscratch2/jlautz/model_robustness/src/model_robustness/attacks/diff_epochs"), '..')))

from attacks.networks import ResNet18

ROOT = Path("")
device = "cuda" if torch.cuda.is_available() else "cpu"



### Steps ###
# Read in zoo

# For all 50 models from model list
# - get normal results for epochs 10, 20, 30, 40, 50
# - evaluate on perturbed dataset for epochs 10, 20, 30, 40, 50
# - plot results (boxplot, lineplot) for all epochs
# - compute kendalls tau
# - save result from kendalls tau in dataframe
# - get plot for only epochs where kendalls tau i