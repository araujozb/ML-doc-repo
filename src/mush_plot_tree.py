import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier, plot_tree

DATA_DIR = "data"
IMG_DIR  = "docs/decision-tree/exercicio/img"
os.makedirs(IMG_DIR, exist_ok=True)

# carregar dados
X_train = pd.read_csv(f"{DATA_DIR}/dataset-x-train.csv")
y_train = pd.read_csv(f"{DATA_DIR}/dataset-y-train.csv")["class"]


clf_viz = DecisionTreeClassifier(
    criterion="gini",       # mesmo critério do baseline 
    max_depth=4,            # limite para caber na figura
    random_state=42
)
clf_viz.fit(X_train, y_train)

# plotar o topo da árvore
plt.figure(figsize=(22, 12))  
plot_tree(
    clf_viz,
    feature_names=X_train.columns.tolist(),
    class_names=["edible(0)", "poisonous(1)"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.tight_layout()
out_path = os.path.join(IMG_DIR, "tree_top.png")
plt.savefig(out_path, dpi=200)
plt.show()
print(f"Figura salva em: {out_path}")
