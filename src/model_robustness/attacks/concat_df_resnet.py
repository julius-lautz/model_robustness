import os
from pathlib import Path
import pandas as pd

ROOT = Path("")

datasets = ["CIFAR10", "CIFAR100", "TinyImageNet"]
attacks = ["PGD", "FGSM"]
eps = [0.1, 0.2, 0.3, 0.4, 0.5]

path_root = os.path.join(ROOT, "/netscratch2/jlautz/model_robustness/src/model_robustness/data/resnet")

# Instantiate empty dataframe
df = pd.DataFrame(columns=["name", "dataset", "attack", "eps", "dropout", 
                            "init_type", "nlin", "lr", "momentum", "optimizer", "wd", "seed",
                            "old_loss", "old_acc", "new_loss", "new_acc"])

for ds in datasets:
    for attack in attacks:
        for e in eps:
            path = os.path.join(path_root, ds, attack, f"eps_{e}", "results", "results_df.csv")
            aux_df = pd.read_csv(path)
            df = pd.concat([df, aux_df])

df.to_csv(os.path.join(path_root, "all_results_resnet_df.csv"), index=False)
