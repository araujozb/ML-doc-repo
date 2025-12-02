import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

IMG_DIR = "docs/kmeans/exercicio/img/"
os.makedirs(IMG_DIR, exist_ok=True)


df = pd.read_csv("src/Mall_Customers.csv")

df_pp = df.drop(columns=["CustomerID"])  
df_pp.head()

#encoding coluna gender
df_pp["Gender_num"] = df_pp["Gender"].map({"Male": 0, "Female": 1})
df_pp = df_pp.drop(columns=["Gender"])
df_pp.head()

features = ["Gender_num", 
            "Age", 
            "Annual Income (k$)", 
            "Spending Score (1-100)"]

X = df_pp[features].values

print(X[:5])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(X_scaled[:5])

#no clustering nao existe variavel alvo, então a divisão é feita apenas na matriz de features

X_train, X_test = train_test_split(
    X_scaled, test_size=0.2, random_state=42
)

print("Formato do conjunto de treino:", X_train.shape)
print("Formato do conjunto de teste:", X_test.shape)


# cotovelo
wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, wcss, marker='o')
plt.xticks(K_range)
plt.xlabel("Número de Clusters (K)")
plt.title("Método do Cotovelo para KMeans")
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
plt.savefig(os.path.join(IMG_DIR, "kmeans_elbow_method.png"))


k = 5

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
kmeans.fit(X_train)

#clusters do conjunto de treino
train_clusters = kmeans.predict(X_train)

# clusters do conjunto de teste
test_clusters = kmeans.predict(X_test)

print("Centroides (em espaço padronizado):")
print(kmeans.cluster_centers_)


# silhouette score
train_silhouette = silhouette_score(X_train, train_clusters)
test_silhouette = silhouette_score(X_test, test_clusters)

print(f"Silhouette Score (Treino): {train_silhouette:.4f}")
print(f"Silhouette Score (Teste): {test_silhouette:.4f}")

#testando com outros valores pq ficou uma caca
for k in range (2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train)
    labels_train = kmeans.predict(X_train)
    sil_train = silhouette_score(X_train, labels_train)
    labels_test = kmeans.predict(X_test)
    sil_test = silhouette_score(X_test, labels_test)
    print(f"K={k} -> Silhouette Score (Teste): {sil_test:.4f}")
    print(f"K={k} -> Silhouette Score (Treino): {sil_train:.4f}")


# rodando dnv
k_2 = 7  

kmeans = KMeans(
    n_clusters=k_2,
    n_init=10,
    random_state=42
)

kmeans.fit(X_train)

train_clusters = kmeans.predict(X_train)
test_clusters = kmeans.predict(X_test)

print("Centroides (k=7, espaço padronizado):")
print(kmeans.cluster_centers_)

#silhouette com k = 7
silhouette_train = silhouette_score(X_train, train_clusters)
silhouette_test = silhouette_score(X_test, test_clusters)

print("Silhouette Treino:", silhouette_train)
print("Silhouette Teste:", silhouette_test)



# PCA no dataset inteiro
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# labels com k=7
labels = kmeans.predict(X_scaled)

df_pca = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "cluster": labels
})

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_pca,
    x="PC1",
    y="PC2",
    hue="cluster",
    palette="tab10",
    s=70
)
plt.title("Clusters K-Means (k=7) em PCA 2D")
plt.legend(title="Cluster")
plt.show()
plt.savefig(os.path.join(IMG_DIR, "kmeans_clusters_pca.png"))

#Perfil Médio dos clusters
df_clusters = df_pp.copy()
df_clusters["cluster"] = labels

cluster_profile = df_clusters.groupby("cluster")[[
    "Gender_num",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]].mean()

cluster_profile["count"] = df_clusters["cluster"].value_counts().sort_index()

print(cluster_profile)

