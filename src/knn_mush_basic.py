import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


IMG_DIR = "docs/knn/exercicio/img/"
os.makedirs(IMG_DIR, exist_ok=True)

# 1) Dados
mush = fetch_openml(name="mushroom", version=1, as_frame=True)
df = mush.frame.copy()
X = df.drop(columns=["class"])
y = df["class"].map({"e": 0, "p": 1})  # edible=0, poisonous=1

# 2) Split estratificado
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# 3) Pré-processamento simples (One-Hot em TODAS as colunas)
pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), X.columns)])

# escolhendo o valor K

ks = [1, 3, 5, 7, 9, 11]
eps = 1e-4

pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), X.columns)])
base_pipe = Pipeline([
    ("pre", pre),
    ("scaler", StandardScaler(with_mean=False)),
    ("knn", KNeighborsClassifier())
])

# 1) Avalia no TESTE para ver quem empata
results = []
for k in ks:
    pipe = base_pipe.set_params(knn__n_neighbors=k).fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    results.append((k, acc))

best_acc = max(acc for _, acc in results)
empatados = [k for k, acc in results if (best_acc - acc) <= eps]

# 2) Preferência por menor ímpar >=3 (se existir)
preferidos = [k for k in empatados if (k % 2 == 1 and k >= 3)]
if preferidos:
    chosen = min(preferidos)
else:
    chosen = min(empatados)  # cai para o menor (possivelmente k=1)

# 3) (Opcional) Desempate por CV em F1 entre os empatados
if len(empatados) > 1:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_cv, best_k = -np.inf, chosen
    for k in empatados:
        pipe = base_pipe.set_params(knn__n_neighbors=k)
        f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1", n_jobs=1).mean()
        if f1 > best_cv or (abs(f1 - best_cv) <= 1e-6 and k < best_k):
            best_cv, best_k = f1, k
    chosen = best_k

print("Resultados (k, acc):", results)
print(f"k escolhido: {chosen}")

knn = KNeighborsClassifier(n_neighbors=chosen)

# 5) Treinar: encaixar o encoder e o KNN
X_train_enc = pre.fit_transform(X_train)
X_test_enc  = pre.transform(X_test)
knn.fit(X_train_enc, y_train)

# 6) Avaliação
pred = knn.predict(X_test_enc)
acc = accuracy_score(y_test, pred)
print(f"Accuracy: {acc:.4f}")

cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(cm, display_labels=["edible(0)","poisonous(1)"]).plot()
plt.title("Mushroom – KNN (k=5) Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "confusion_matrix_basic.png"), dpi=150)
