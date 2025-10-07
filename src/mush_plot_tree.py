import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier, plot_tree

DATA_DIR = "data"
IMG_DIR  = "docs/decision-tree/exercicio/img"
os.makedirs(IMG_DIR, exist_ok=True)

# carregar dados
x_train = pd.read_csv(f"{DATA_DIR}/dataset-x-train.csv")
y_train = pd.read_csv(f"{DATA_DIR}/dataset-y-train.csv")["class"]


clf_viz = DecisionTreeClassifier(
    criterion="gini",       
    max_depth=4,            
    random_state=42
)
clf_viz.fit(x_train, y_train)

# plotar o topo da árvore
plt.figure(figsize=(18, 10))
plot_tree(
    clf_viz,                                 
    feature_names= x_train.columns,
    class_names=["edible(0)", "poisonous(1)"],
    filled=True, rounded=True, fontsize=8,
    max_depth=4                    
)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "tree_top_depth4.png"), dpi=200, bbox_inches="tight")
plt.show()
