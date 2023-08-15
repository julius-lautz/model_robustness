import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
np.random.seed(0)
ROOT = Path("")

# Define parameters
datasets = ["MNIST", "CIFAR10", "SVHN"]
attacks = ["PGD", "FGSM"]
setups = ["hyp-10-f", "hyp-10-r", "seed"]
eps = [.1, .2, .3, .4, .5]

# Define paths
result_root = ROOT.joinpath("/netscratch2/jlautz/model_robustness/src/model_robustness/plots")
zoo_root = ROOT.joinpath("/ds2/model_zoos/zoos_v2/")
df = pd.read_csv("/netscratch2/jlautz/model_robustness/src/model_robustness/data/all_results.csv")

def get_names_for_path(dataset, setup):

    if setup == "hyp-10-r":
        abbreviated = "hyp_rand"
    elif setup == "hyp-10-f":
        abbreviated = "hyp_fix"
    else:
        abbreviated = "seed"

    if dataset == "CIFAR10":
        p = f"large/analysis_data_{abbreviated}.pt"
    else:
        p = f"analysis_data_{abbreviated}.pt"
    
    return p


# Big Loop
for ds in datasets:
    for setup in setups:
        for attack in attacks:
        
            aux_df = df[(df.dataset == ds) & (df.attack == attack) & (df.setup == setup)]
            both_results = pd.DataFrame()

            for i, e in enumerate(eps):

                eps_df = aux_df[aux_df.eps == e].sort_values(by="name").reset_index(drop=True)

                both_results["normal"] = eps_df["old_acc"]
                both_results[f"eps_{e}"] = eps_df["new_acc"]

            sns.displot(data=both_results[["normal", "eps_0.1"]], kde=True)
            plt.savefig(
                os.path.join(result_root, f"accuracy_histplots/{ds}_{attack}_{setup}_histplot.png"),
                bbox_inches="tight"
            )
            plt.clf()

