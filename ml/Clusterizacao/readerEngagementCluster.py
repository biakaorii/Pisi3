import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Encontrar o caminho do dataset
caminho_atual = os.path.dirname(os.path.abspath(__file__))
caminho_dataset = os.path.join(caminho_atual, '..', '..', 'dataset', 'dados.parquet')

# Carregar o dataset
df = pd.read_parquet(caminho_dataset)

# Criar features derivadas
df['taxa_releitura'] = df['relendo'] / (df['leram'] + 1)
df['taxa_releitura'] = df['taxa_releitura'].replace([np.inf, -np.inf], np.nan).fillna(0)

df['engajamento_resenhas'] = df['resenha'] / (df['avaliacao'] + 1)
df['engajamento_resenhas'] = df['engajamento_resenhas'].replace([np.inf, -np.inf], np.nan).fillna(0)

df['interesse_futuro'] = df['querem_ler'] / (df['leram'] + 1)
df['interesse_futuro'] = df['interesse_futuro'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Selecionar features
features = ['lendo', 'leram', 'querem_ler', 'relendo', 'resenha', 
            'taxa_releitura', 'engajamento_resenhas', 'interesse_futuro']
X = df[features]

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Criar e ajustar o modelo K-Means
kmeans = KMeans(n_clusters=5, random_state=42, algorithm='elkan', n_init=10) 
df['cluster'] = kmeans.fit_predict(X_scaled)

# Avaliar os clusters
silhouette_avg = silhouette_score(X_scaled, df['cluster'])
davies_bouldin = davies_bouldin_score(X_scaled, df['cluster'])
    
print("="*80)
print("MÉTRICAS DE AVALIAÇÃO")
print("="*80)
print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Davies-Bouldin Score: {davies_bouldin:.4f}\n")

# Ver estatísticas por cluster
print("="*80)
print("MEDIANAS POR CLUSTER")
print("="*80)
print(df.groupby('cluster')[features].median().round(2))
print()

print("="*80)
print("MÉDIAS POR CLUSTER")
print("="*80)
print(df.groupby('cluster')[features].mean().round(2))
print()

print("="*80)
print("CONTAGEM POR CLUSTER")
print("="*80)
print(df['cluster'].value_counts().sort_index())
print()

# Análise interpretativa dos clusters
print("="*80)
print("INTERPRETAÇÃO DOS CLUSTERS - ENGAJAMENTO DOS LEITORES")
print("="*80)
for cluster_id in range(5):
    cluster_data = df[df['cluster'] == cluster_id][features].median()
    lendo_med = cluster_data['lendo']
    leram_med = cluster_data['leram']
    querem_med = cluster_data['querem_ler']
    relendo_med = cluster_data['relendo']
    taxa_releit = cluster_data['taxa_releitura']
    eng_resenhas = cluster_data['engajamento_resenhas']
    interesse_fut = cluster_data['interesse_futuro']
    
    # Classificar o perfil de engajamento
    if taxa_releit > 0.05 and eng_resenhas > 0.10:
        tipo = "CULT FOLLOWING (Alta releitura + Alto engajamento em resenhas)"
    elif querem_med > leram_med * 1.5 and lendo_med > 50:
        tipo = "VIRAIS / TENDÊNCIA (Muitos lendo e querendo ler agora)"
    elif leram_med > 1000 and lendo_med < 100:
        tipo = "JÁ ESTABELECIDOS (Muitos já leram, poucos lendo agora)"
    elif querem_med > leram_med * 2:
        tipo = "EM ASCENSÃO (Mais querem ler do que já leram)"
    elif leram_med > 500 and eng_resenhas > 0.05:
        tipo = "COMUNIDADE ATIVA (Boa base + Alta discussão)"
    elif relendo_med > leram_med * 0.03:
        tipo = "OBRAS VICIANTES (Alta taxa de releitura)"
    else:
        tipo = "BAIXO ENGAJAMENTO (Pouca interação geral)"
    
    print(f"\nCluster {cluster_id}:")
    print(f"  Lendo: {lendo_med:.0f}")
    print(f"  Leram: {leram_med:.0f}")
    print(f"  Querem ler: {querem_med:.0f}")
    print(f"  Relendo: {relendo_med:.0f}")
    print(f"  Resenhas: {cluster_data['resenha']:.0f}")
    print(f"  Taxa releitura: {taxa_releit:.2%}")
    print(f"  Engajamento resenhas: {eng_resenhas:.2%}")
    print(f"  Interesse futuro: {interesse_fut:.2f}x")
    print(f"  Perfil: {tipo}")
print("\n")

# Reduzir dimensionalidade para 2D com PCA para plotar
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("="*80)
print("VARIÂNCIA EXPLICADA PELO PCA")
print("="*80)
print(f"PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"PC2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"Total: {sum(pca.explained_variance_ratio_):.2%}")
print("\n")

# Plotar clusters
cores = ['red', 'blue', 'green', 'orange', 'purple']
nomes_clusters = [f'Cluster {i}' for i in range(5)]

plt.figure(figsize=(12, 8))
for i in range(5):
    mask = df['cluster'] == i
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        s=50,
        c=cores[i],
        label=nomes_clusters[i],
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

# Plotar centroides
centroides_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroides_pca[:, 0], centroides_pca[:, 1], 
            c='black', marker='X', s=300, 
            edgecolors='white', linewidth=2,
            label='Centroides')

plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.1%} variância)')
plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.1%} variância)')
plt.title('Clusters de Engajamento dos Leitores', fontsize=14, fontweight='bold')
plt.legend(title='Clusters', loc='best', fontsize=10, frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Salvar o DataFrame com os clusters
caminho_saida = os.path.join(caminho_atual, '..', '..', 'dataset', 'cluster_engajamento.parquet')
df.to_parquet(caminho_saida, index=False)
print(f"Dataset com clusters salvo em: {caminho_saida}")