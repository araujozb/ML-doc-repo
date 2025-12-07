import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

mush = fetch_openml("mushroom", version=1, as_frame=True)


df = mush.frame

# print(df.head())
# print(df.columns)

# 1. Carregamento do dataset

print("Primeiras linhas do dataset:")
print(df.head())
print("\nColunas:", df.columns.tolist())

y = df["class"]
X = df.drop("class", axis=1)

# 2. EDA simples


print("\nDistribuição da variável alvo:")
print(y.value_counts())

# Plot da distribuição da classe
y.value_counts().plot(kind="bar")
plt.title("Distribuição da classe (Mushroom)")
plt.xlabel("Classe (e = comestível, p = venenoso)")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.show()

# Como todas as features são categóricas, vamos só olhar algumas frequências
print("\nExemplo de contagem de valores de 'odor':")
print(df["odor"].value_counts())

# 3. Pré-processamento

X_encoded = pd.get_dummies(X, drop_first=True)
y_encoded = y.map({"e": 0, "p": 1})  # 0 = comestível, 1 = venenoso

print("\nShape após one-hot encoding:", X_encoded.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y_encoded,
    test_size=0.3,
    random_state=42,
    stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_full_scaled = scaler.fit_transform(X_encoded)  # K-Means

# 4. Modelo KNN 
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred_knn = knn.predict(X_test_scaled)

print("\n===== RESULTADOS KNN =====")
print("Acurácia:", accuracy_score(y_test, y_pred_knn))
print("\nRelatório de classificação (KNN):")
print(classification_report(y_test, y_pred_knn, target_names=["comestível", "venenoso"]))

cm_knn = confusion_matrix(y_test, y_pred_knn)
print("Matriz de confusão (KNN):")
print(cm_knn)

# Plot da matriz de confusão KNN
plt.figure(figsize=(4, 3))
plt.imshow(cm_knn, interpolation='nearest')
plt.title('Matriz de Confusão - KNN')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["comestível", "venenoso"], rotation=45)
plt.yticks(tick_marks, ["comestível", "venenoso"])
plt.xlabel('Predito')
plt.ylabel('Real')
plt.tight_layout()
plt.show()

# 5. Modelo K-Means (não supervisionado)

kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto")
clusters = kmeans.fit_predict(X_full_scaled)

mapping = {}
for c in np.unique(clusters):
    mask = clusters == c
    majority_class = y_encoded[mask].mode()[0]
    mapping[c] = majority_class

print("\nMapeamento cluster -> classe real (0=comestível, 1=venenoso):")
print(mapping)

# Transformar os clusters em previsões de classe
y_pred_kmeans = np.array([mapping[c] for c in clusters])
y_true_kmeans = y_encoded.values  # usando todo o dataset

print("\n===== RESULTADOS K-MEANS =====")
print("Acurácia (k-means vs rótulos verdadeiros):", accuracy_score(y_true_kmeans, y_pred_kmeans))
print("\nRelatório de classificação (K-Means):")
print(classification_report(y_true_kmeans, y_pred_kmeans, target_names=["comestível", "venenoso"]))

cm_kmeans = confusion_matrix(y_true_kmeans, y_pred_kmeans)
print("Matriz de confusão (K-Means):")
print(cm_kmeans)

# Plot da matriz de confusão K-Means
plt.figure(figsize=(4, 3))
plt.imshow(cm_kmeans, interpolation='nearest')
plt.title('Matriz de Confusão - K-Means')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ["comestível", "venenoso"], rotation=45)
plt.yticks(tick_marks, ["comestível", "venenoso"])
plt.xlabel('Predito (por cluster)')
plt.ylabel('Real')
plt.tight_layout()
plt.show()

# 6. Tabela comparando métricas

metrics = []

# Métricas KNN (no conjunto de teste)
metrics.append({
    "modelo": "KNN",
    "acuracia": accuracy_score(y_test, y_pred_knn),
    "precisao": precision_score(y_test, y_pred_knn),
    "recall": recall_score(y_test, y_pred_knn),
    "f1": f1_score(y_test, y_pred_knn),
})

# Métricas K-Means (no dataset completo)
metrics.append({
    "modelo": "K-Means",
    "acuracia": accuracy_score(y_true_kmeans, y_pred_kmeans),
    "precisao": precision_score(y_true_kmeans, y_pred_kmeans),
    "recall": recall_score(y_true_kmeans, y_pred_kmeans),
    "f1": f1_score(y_true_kmeans, y_pred_kmeans),
})

df_metrics = pd.DataFrame(metrics)
print("\n===== COMPARAÇÃO DE MÉTRICAS =====")
print(df_metrics)
