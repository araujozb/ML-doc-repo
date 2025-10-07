# knn_mushroom_pca_visual.py
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

RANDOM_STATE = 42
IMG_DIR = "docs/knn/exercicio/img/"
os.makedirs(IMG_DIR, exist_ok=True)

# 1) Dados e split (igual ao básico)
mush = fetch_openml(name="mushroom", version=1, as_frame=True)
df = mush.frame.copy()
X = df.drop(columns=["class"])
y = df["class"].map({"e": 0, "p": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)

# 2) One-Hot (mesmo encoder do básico)
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), X.columns)])
X_train_enc = pre.fit_transform(X_train)
X_test_enc  = pre.transform(X_test)

# 3) PCA para 2D (apenas para visualização)
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_train_2d = pca.fit_transform(X_train_enc.toarray())
X_test_2d  = pca.transform(X_test_enc.toarray())

# 4) KNN em 2D só para desenhar fronteira (não é o modelo “oficial”)
knn_2d = KNeighborsClassifier(n_neighbors=7)
knn_2d.fit(X_train_2d, y_train)

# 5) Malha e fronteira de decisão
h = 0.05
x_min, x_max = X_train_2d[:,0].min()-1, X_train_2d[:,0].max()+1
y_min, y_max = X_train_2d[:,1].min()-1, X_train_2d[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_train_2d[:,0], X_train_2d[:,1], c=y_train, s=20, edgecolor="k")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title("KNN Decision Boundary (PCA-2D, k=7)")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "decision_boundary_pca2d.png"), dpi=150)
