# knn_mushroom_pca_visual_pretty.py
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

RANDOM_STATE = 42
IMG_DIR = "docs/knn/exercicio/img/"
os.makedirs(IMG_DIR, exist_ok=True)

# --- dados + split ---
mush = fetch_openml(name="mushroom", version=1, as_frame=True)
df = mush.frame.copy()
X = df.drop(columns=["class"])
y = df["class"].map({"e": 0, "p": 1})
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)

# --- one-hot ---
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), X.columns)])
X_train_enc = pre.fit_transform(X_train)
X_test_enc  = pre.transform(X_test)

# --- PCA 2D (apenas para visual) ---
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_train_2d = pca.fit_transform(X_train_enc.toarray())
X_test_2d  = pca.transform(X_test_enc.toarray())

# --- params para deixar “reto/suave” ---
K = 5
P = 1                 # Manhattan (mais “reto” que p=2 euclidiano)
WEIGHTS = "distance"  # suaviza bordas
h = 0.08              # passo da malha (maior = aparência mais lisa)

knn_2d = KNeighborsClassifier(n_neighbors=K, p=P, weights=WEIGHTS)
knn_2d.fit(X_train_2d, y_train)

# --- malha e predição ---
x_min, x_max = X_train_2d[:,0].min()-1, X_train_2d[:,0].max()+1
y_min, y_max = X_train_2d[:,1].min()-1, X_train_2d[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# --- plot bonito ---
cmap_bg = ListedColormap(["#f2e8b3", "#cab8ff"])  # cores suaves de fundo
cmap_pts = ["#e3b505", "#5f3dc4"]                 # cores dos pontos

plt.figure(figsize=(9,7))
plt.pcolormesh(xx, yy, Z, shading="nearest", cmap=cmap_bg, alpha=0.6)
plt.scatter(X_train_2d[:,0], X_train_2d[:,1], c=y_train, s=28,
            edgecolor="k", linewidth=0.3, alpha=0.9, cmap=ListedColormap(cmap_pts))
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title(f"KNN Decision Boundary (PCA-2D, k={K}, p={P}, weights='{WEIGHTS}')")
# legenda simples
from matplotlib.lines import Line2D
legend_elems = [
    Line2D([0],[0], marker='o', color='w', label='edible (0)',
           markerfacecolor=cmap_pts[0], markeredgecolor='k', markersize=8),
    Line2D([0],[0], marker='o', color='w', label='poisonous (1)',
           markerfacecolor=cmap_pts[1], markeredgecolor='k', markersize=8),
]
plt.legend(handles=legend_elems, loc="best", frameon=True)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "decision_boundary_pca2d_pretty.png"), dpi=150)
plt.show()
