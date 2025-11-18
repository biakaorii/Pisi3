import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import AgglomerativeClustering
import warnings
warnings.filterwarnings('ignore')

# Encontrar o caminho do dataset
caminho_atual = os.path.dirname(os.path.abspath(__file__))
caminho_dataset = os.path.join(caminho_atual, '..', '..', 'dataset', 'dados.parquet')

# Carregar o dataset
df = pd.read_parquet(caminho_dataset)

# Criar features derivadas
df['taxa_abandono'] = df['abandonos'] / (df['leram'] + 1)
df['taxa_abandono'] = df['taxa_abandono'].replace([np.inf, -np.inf], np.nan).fillna(0)

df['dificuldade'] = df['paginas'] * (1 - df['rating'] / 5)
df['dificuldade'] = df['dificuldade'].replace([np.inf, -np.inf], np.nan).fillna(0)

df['taxa_conclusao'] = df['leram'] / (df['leram'] + df['abandonos'] + 1)
df['taxa_conclusao'] = df['taxa_conclusao'].replace([np.inf, -np.inf], np.nan).fillna(0)

df['densidade_conteudo'] = df['paginas'] / (df['ano'].max() - df['ano'] + 1)
df['densidade_conteudo'] = df['densidade_conteudo'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Selecionar features
features = ['paginas', 'rating', 'abandonos', 'leram', 'ano',
            'taxa_abandono', 'dificuldade', 'taxa_conclusao']
X = df[features]

# Remover valores nulos
X = X.dropna()
df = df.loc[X.index].copy()

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Testar diferentes números de clusters com Agglomerative
print("="*80)
print("AGGLOMERATIVE CLUSTERING (HIERÁRQUICO)")
print("="*80)
print("Testando diferentes números de clusters...\n")

resultados = []
for n in range(3, 9):
    clusterer = AgglomerativeClustering(n_clusters=n, linkage='ward')
    labels = clusterer.fit_predict(X_scaled)
    
    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    
    resultados.append({
        'n_clusters': n,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin
    })
    
    print(f"n_clusters={n}:")
    print(f"  Silhouette: {silhouette:.4f}")
    print(f"  Davies-Bouldin: {davies_bouldin:.4f}")

# Escolher melhor número de clusters
melhor = max(resultados, key=lambda x: x['silhouette'])
melhor_n = melhor['n_clusters']

print(f"\n{'='*80}")
print(f"MELHOR NÚMERO DE CLUSTERS: {melhor_n}")
print(f"{'='*80}\n")

# Criar e ajustar o modelo Agglomerative com melhor n
clusterer = AgglomerativeClustering(n_clusters=melhor_n, linkage='ward')
df['cluster'] = clusterer.fit_predict(X_scaled)

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
print("INTERPRETAÇÃO DOS CLUSTERS - COMPLEXIDADE DE LEITURA")
print("="*80)

clusters_unicos = sorted(df['cluster'].unique())

for cluster_id in clusters_unicos:
    cluster_data = df[df['cluster'] == cluster_id][features].median()
    paginas_med = cluster_data['paginas']
    rating_med = cluster_data['rating']
    abandonos_med = cluster_data['abandonos']
    leram_med = cluster_data['leram']
    ano_med = cluster_data['ano']
    taxa_aband = cluster_data['taxa_abandono']
    dificuldade_med = cluster_data['dificuldade']
    taxa_concl = cluster_data['taxa_conclusao']
    
    # Classificar o perfil de complexidade
    if paginas_med < 300 and taxa_aband < 0.1:
        tipo = "🏃 LEITURA RÁPIDA (Poucas páginas + Baixos abandonos)"
    elif paginas_med > 500 and taxa_aband > 0.15:
        tipo = "🧗 DESAFIADORES (Muitas páginas + Muitos abandonos)"
    elif paginas_med > 400 and taxa_aband < 0.1:
        tipo = "📖 ACESSÍVEIS (Muitas páginas + Baixos abandonos - Bem escritos)"
    elif taxa_aband > 0.2:
        tipo = "💪 PARA INICIADOS (Poucos conseguem terminar)"
    elif paginas_med < 250:
        tipo = "⚡ LEITURAS LEVES (Curtas e fáceis)"
    elif rating_med > 4.2 and taxa_aband < 0.12:
        tipo = "⭐ ENVOLVENTES (Boa avaliação + Baixos abandonos)"
    elif dificuldade_med > 100:
        tipo = "📚 COMPLEXOS (Alta dificuldade geral)"
    else:
        tipo = "📘 PADRÃO (Complexidade moderada)"
    
    print(f"\nCluster {cluster_id}:")
    print(f"  Páginas (mediana): {paginas_med:.0f}")
    print(f"  Rating: {rating_med:.2f}")
    print(f"  Abandonos: {abandonos_med:.0f}")
    print(f"  Leram: {leram_med:.0f}")
    print(f"  Ano: {ano_med:.0f}")
    print(f"  Taxa de abandono: {taxa_aband:.2%}")
    print(f"  Taxa de conclusão: {taxa_concl:.2%}")
    print(f"  Dificuldade: {dificuldade_med:.2f}")
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
cores = plt.cm.Spectral(np.linspace(0, 1, len(clusters_unicos)))

plt.figure(figsize=(12, 8))
for i, cluster_id in enumerate(clusters_unicos):
    mask = df['cluster'] == cluster_id
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        s=50,
        c=[cores[i]],
        label=f'Cluster {cluster_id}',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5
    )

plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.1%} variância)')
plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.1%} variância)')
plt.title('Clusters de Complexidade de Leitura (Agglomerative)', fontsize=14, fontweight='bold')
plt.legend(title='Clusters', loc='best', fontsize=10, frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Salvar o DataFrame com os clusters
caminho_saida = os.path.join(caminho_atual, '..', '..', 'dataset', 'cluster_complexidade_leitura.parquet')
df.to_parquet(caminho_saida, index=False)
print(f"Dataset com clusters salvo em: {caminho_saida}")
