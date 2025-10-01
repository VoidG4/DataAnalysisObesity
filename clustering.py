import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

df = pd.read_csv('processed_df.csv')

features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sse = []
K = range(2, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K, sse, marker='o', color='cyan')
plt.xlabel('Αριθμός Κλάσεων (k)')
plt.ylabel('SSE (Sum of Squared Errors)')
plt.title('Elbow Method για KMeans')
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=6, random_state=0)
clusters_kmeans = kmeans.fit_predict(X_scaled)

df['Cluster_KMeans'] = clusters_kmeans

silhouette_kmeans = silhouette_score(X_scaled, clusters_kmeans)
print(f'Silhouette Score για KMeans: {silhouette_kmeans:.3f}')

dbscan = DBSCAN(eps=1.38, min_samples=5)
clusters_dbscan = dbscan.fit_predict(X_scaled)

df['Cluster_DBSCAN'] = clusters_dbscan

mask = clusters_dbscan != -1
if mask.sum() > 0:
    silhouette_dbscan = silhouette_score(X_scaled[mask], clusters_dbscan[mask])
    print(f'Silhouette Score για DBSCAN (χωρίς outliers): {silhouette_dbscan:.3f}')
else:
    print("Πάρα πολλοί outliers - δε μπορεί να υπολογιστεί Silhouette Score για DBSCAN.")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters_kmeans, palette='viridis')
plt.title('KMeans Clustering (PCA προβολή)')

plt.subplot(1, 2, 2)
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=clusters_dbscan, palette='Set2')
plt.title('DBSCAN Clustering (PCA προβολή)')

plt.show()