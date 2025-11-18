import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Encontrar o caminho do dataset
caminho_atual = os.path.dirname(os.path.abspath(__file__))
caminho_dataset = os.path.join(caminho_atual, '..', '..', 'dataset', 'dados.parquet')

# Carregar o dataset
df = pd.read_parquet(caminho_dataset)

# Criar features derivadas
df['balanco_genero'] = np.abs(df['male'] - df['female']) / (df['male'] + df['female'] + 1)
df['balanco_genero'] = df['balanco_genero'].replace([np.inf, -np.inf], np.nan).fillna(0)

df['popularidade_total'] = df['male'] + df['female']

df['dominancia_masculina'] = df['male'] / (df['male'] + df['female'] + 1)
df['dominancia_masculina'] = df['dominancia_masculina'].replace([np.inf, -np.inf], np.nan).fillna(0)

df['dominancia_feminina'] = df['female'] / (df['male'] + df['female'] + 1)
df['dominancia_feminina'] = df['dominancia_feminina'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Selecionar features
features = ['male', 'female', 'rating', 'avaliacao', 'balanco_genero', 
            'popularidade_total', 'dominancia_masculina', 'dominancia_feminina']
X = df[features]

# Remover valores nulos
X = X.dropna()
df = df.loc[X.index].copy()

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encontrar eps ideal usando k-distance graph
print("="*80)
print("ENCONTRANDO EPS IDEAL")
print("="*80)

k = 20  # min_samples
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

# Ordenar distâncias
distances = np.sort(distances[:, k-1], axis=0)

# Sugerir eps (joelho da curva - aproximado)
eps_sugerido = np.percentile(distances, 90)
print(f"EPS sugerido (percentil 90): {eps_sugerido:.3f}")
print(f"EPS médio: {np.mean(distances):.3f}")
print(f"EPS mediano: {np.median(distances):.3f}\n")

# Testar diferentes valores de eps
eps_valores = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
resultados = []

print("="*80)
print("TESTANDO DIFERENTES VALORES DE EPS")
print("="*80)

for eps_teste in eps_valores:
    clusterer = DBSCAN(eps=eps_teste, min_samples=20)
    labels = clusterer.fit_predict(X_scaled)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    perc_noise = (n_noise / len(labels)) * 100
    
    print(f"\nEPS={eps_teste:.1f}:")
    print(f"  Clusters: {n_clusters}")
    print(f"  Ruído: {n_noise} ({perc_noise:.1f}%)")
    
    resultados.append({
        'eps': eps_teste,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'perc_noise': perc_noise
    })

# Escolher o melhor eps (entre 3-8 clusters e <30% ruído)
melhor_eps = None
for r in resultados:
    if 3 <= r['n_clusters'] <= 8 and r['perc_noise'] < 30:
        melhor_eps = r['eps']
        break

# Se não encontrou, usar o que tem mais clusters
if melhor_eps is None:
    melhor_eps = max(resultados, key=lambda x: x['n_clusters'] if x['perc_noise'] < 50 else 0)['eps']

print(f"\n{'='*80}")
print(f"MELHOR EPS ESCOLHIDO: {melhor_eps}")
print(f"{'='*80}\n")

# Criar e ajustar o modelo DBSCAN com melhor eps
clusterer = DBSCAN(eps=melhor_eps, min_samples=20)
df['cluster'] = clusterer.fit_predict(X_scaled)

# Contar clusters (excluindo ruído -1)
n_clusters = len(set(df['cluster'])) - (1 if -1 in df['cluster'].values else 0)
n_noise = list(df['cluster']).count(-1)

print("="*80)
print("INFORMAÇÕES DO DBSCAN")
print("="*80)
print(f"Número de clusters encontrados: {n_clusters}")
print(f"Número de pontos de ruído: {n_noise}")
print(f"Porcentagem de ruído: {(n_noise/len(df)*100):.2f}%\n")

# Avaliar os clusters (apenas pontos não-ruído)
if n_clusters >= 2:
    mask = df['cluster'] != -1
    X_filtered = X_scaled[mask]
    labels_filtered = df.loc[mask, 'cluster']
    
    silhouette_avg = silhouette_score(X_filtered, labels_filtered)
    davies_bouldin = davies_bouldin_score(X_filtered, labels_filtered)
    
    print("="*80)
    print("MÉTRICAS DE AVALIAÇÃO")
    print("="*80)
    print(f"Silhouette Score: {silhouette_avg:.4f}")
    print(f"Davies-Bouldin Score: {davies_bouldin:.4f}\n")
else:
    print("="*80)
    print("AVISO: Menos de 2 clusters encontrados")
    print("="*80)
    print("Não é possível calcular métricas de avaliação\n")

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
print("INTERPRETAÇÃO DOS CLUSTERS - DIVERSIDADE DE PÚBLICO")
print("="*80)

# Incluir análise do ruído se existir
clusters_unicos = sorted(df['cluster'].unique())

for cluster_id in clusters_unicos:
    cluster_data = df[df['cluster'] == cluster_id][features].median()
    male_med = cluster_data['male']
    female_med = cluster_data['female']
    rating_med = cluster_data['rating']
    avaliacao_med = cluster_data['avaliacao']
    balanco = cluster_data['balanco_genero']
    pop_total = cluster_data['popularidade_total']
    dom_masc = cluster_data['dominancia_masculina']
    dom_fem = cluster_data['dominancia_feminina']
    
    # Calcular percentuais
    total = male_med + female_med
    perc_male = (male_med / total * 100) if total > 0 else 0
    perc_female = (female_med / total * 100) if total > 0 else 0
    
    if cluster_id == -1:
        tipo = "RUÍDO (Pontos que não se encaixam em nenhum cluster)"
        print(f"\nCluster {cluster_id} (RUÍDO):")
    else:
        # Classificar o perfil de público
        if balanco < 0.2 and pop_total > 500:
            tipo = "UNIVERSAL (Balanceado entre gêneros + Alta popularidade)"
        elif dom_masc > 0.65 and pop_total > 200:
            tipo = "MASCULINO DOMINANTE (Público majoritariamente masculino)"
        elif dom_fem > 0.65 and pop_total > 200:
            tipo = "FEMININO DOMINANTE (Público majoritariamente feminino)"
        elif balanco > 0.5 and pop_total > 300:
            tipo = "NICHO ESPECÍFICO (Desbalanceado mas popular em um gênero)"
        elif pop_total < 100:
            tipo = "BAIXA POPULARIDADE (Pouco público geral)"
        elif balanco < 0.3:
            tipo = "PÚBLICO EQUILIBRADO (Boa diversidade de gênero)"
        else:
            tipo = "PADRÃO (Características mistas)"
        
        print(f"\nCluster {cluster_id}:")
    
    print(f"  Leitores masculinos: {male_med:.0f} ({perc_male:.1f}%)")
    print(f"  Leitoras femininas: {female_med:.0f} ({perc_female:.1f}%)")
    print(f"  Rating médio: {rating_med:.2f}")
    print(f"  Avaliações: {avaliacao_med:.0f}")
    print(f"  Balanço de gênero: {balanco:.2f} (0=equilibrado, 1=desbalanceado)")
    print(f"  Popularidade total: {pop_total:.0f}")
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
n_colors = len(clusters_unicos)
if -1 in clusters_unicos:
    # Cores para clusters + preto para ruído
    cores = plt.cm.Spectral(np.linspace(0, 1, max(n_colors - 1, 1)))
    cores = list(cores) if n_colors > 1 else []
    cores = cores + [[0, 0, 0, 1]]  # Adicionar preto para ruído
else:
    cores = plt.cm.Spectral(np.linspace(0, 1, n_colors))

plt.figure(figsize=(12, 8))
for i, cluster_id in enumerate(clusters_unicos):
    mask = df['cluster'] == cluster_id
    label = f'Ruído' if cluster_id == -1 else f'Cluster {cluster_id}'
    marker = 'x' if cluster_id == -1 else 'o'
    alpha = 0.3 if cluster_id == -1 else 0.6
    
    plt.scatter(
        X_pca[mask, 0],
        X_pca[mask, 1],
        s=50,
        c=[cores[i]],
        label=label,
        alpha=alpha,
        edgecolors='black',
        linewidth=0.5,
        marker=marker
    )

plt.xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.1%} variância)')
plt.ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.1%} variância)')
plt.title(f'Clusters de Diversidade de Público (DBSCAN eps={melhor_eps})', fontsize=14, fontweight='bold')
plt.legend(title='Clusters', loc='best', fontsize=10, frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Salvar o DataFrame com os clusters
caminho_saida = os.path.join(caminho_atual, '..', '..', 'dataset', 'cluster_diversidade_publico.parquet')
df.to_parquet(caminho_saida, index=False)
print(f"Dataset com clusters salvo em: {caminho_saida}")