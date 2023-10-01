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
df = pd.read_csv("/netscratch2/jlautz/model_robustness/src/model_robustness/data/sparsified/all_results_df.csv")
df = df[df.old_acc>=0.2]


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

            # sns.displot(data=both_results[["normal", "eps_0.1"]], kde=True)
            sns.boxplot(both_results, orient="h")
            
            plt.savefig(
                os.path.join(result_root, f"accuracy_boxplots/sparsified/{ds}_{setup}_{attack}_boxplot.png"),    # add setup here if needed
                bbox_inches="tight"
            )
            plt.clf()
            print(f"{ds} {setup} {attack} done.")

