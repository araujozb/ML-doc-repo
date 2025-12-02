from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from mall_preprocess.py import X_train

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