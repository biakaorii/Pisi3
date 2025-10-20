import pandas as pd
from sklearn.cluster import KMeans
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score

#Encontrar o caminho do dataset
caminho_atual = os.path.dirname(os.path.abspath(__file__))
caminho_dataset = os.path.join(caminho_atual, '..', 'dataset', 'dados.parquet')

#Carregar o dataset
df = pd.read_parquet(caminho_dataset)

#Criar feature derivada de abandonos e leram
df['taxa_abandono'] = df['abandonos'] / df['leram']
df['taxa_abandono'] = df['taxa_abandono'].replace([np.inf, -np.inf], np.nan).fillna(0)

#Selecionar as features
features = ['taxa_abandono','lendo', 'leram', 'rating', 'avaliacao']
X = df[features]

#Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Criar e ajustar o modelo K-Means
kmeans = KMeans(n_clusters=5, random_state=42, algorithm='elkan') 
df['cluster'] = kmeans.fit_predict(X_scaled)

#Avaliar os clusters
silhouette_avg = silhouette_score(X_scaled, df['cluster'])
davies_bouldin = davies_bouldin_score(X_scaled, df['cluster'])
    
print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Davies-Bouldin Score: {davies_bouldin:.4f}")

#Ver estatísticas por cluster
print("\nMedianas por cluster:")
print(df.groupby('cluster')[features].median())
X_scaled_df = pd.DataFrame(X_scaled, columns=features)


print("\nMedias por cluster:")
print(df.groupby('cluster')[features].mean())
X_scaled_df = pd.DataFrame(X_scaled, columns=features)

#Reduzir dimensionalidade para 2D com PCA para plotar
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

#Plotar clusters
cores = ['red', 'blue', 'green', 'yellow', 'purple']
nomes_clusters = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4']
plt.figure(figsize=(8,6))
for i in range(len(cores)):
    plt.scatter(
        X_pca[df['cluster'] == i, 0],
        X_pca[df['cluster'] == i, 1],
        s=50,
        c=cores[i],
        label=nomes_clusters[i],
        alpha=0.4
    )
plt.xlabel('PCA 1') 
plt.ylabel('PCA 2') 
plt.title('Clusters de livros') 
plt.legend(title='Clusters', loc='best', fontsize=10, frameon=False)
plt.show()

#Salvar o DataFrame com os clusters
caminho_saida = os.path.join(caminho_atual, '..', 'dataset', 'cluster_popularidade.parquet')
df.to_parquet(caminho_saida, index=False)