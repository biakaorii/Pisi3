import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ==============================================================================

# Encontrar o caminho do dataset
caminho_atual = os.path.dirname(os.path.abspath(__file__))
caminho_dataset = os.path.join(caminho_atual, '..', '..', 'dataset', 'dados.parquet')

# Carregar o dataset
print(f"Carregando dataset de: {caminho_dataset}")
df = pd.read_parquet(caminho_dataset)

# Criar features de engajamento (mesma lógica do script anterior)
df['engajamento_total'] = df['leram'] + df['lendo'] + df['querem_ler']

df['taxa_abandono'] = df['abandonos'] / (df['leram'] + 1)
df['taxa_abandono'] = df['taxa_abandono'].replace([np.inf, -np.inf], np.nan).fillna(0)

df['taxa_conclusao'] = df['leram'] / (df['leram'] + df['abandonos'] + 1)
df['taxa_conclusao'] = df['taxa_conclusao'].replace([np.inf, -np.inf], np.nan).fillna(0)

df['popularidade'] = np.log1p(df['avaliacao'])

df['interesse_futuro'] = df['querem_ler'] / (df['engajamento_total'] + 1)
df['interesse_futuro'] = df['interesse_futuro'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Selecionar features
features = ['leram', 'lendo', 'querem_ler', 'abandonos', 'avaliacao', 
            'engajamento_total', 'taxa_abandono', 'taxa_conclusao', 
            'popularidade', 'interesse_futuro']

X = df[features]

# Remover valores nulos
X = X.dropna()
df = df.loc[X.index].copy()

# Padronizar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Dados preparados: {X_scaled.shape[0]} amostras, {X_scaled.shape[1]} features")

# ==============================================================================
# 2. ANÁLISE DE SILHUETA
# ==============================================================================

print("\n" + "="*80)
print("ANÁLISE DE SILHUETA (SILHOUETTE ANALYSIS)")
print("="*80)

# Intervalo de K para testar (focando nos mais prováveis para visualização detalhada)
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

best_k = 2
best_score = -1
results = []

for n_clusters in range_n_clusters:
    # Criar subplot com 1 linha e 2 colunas
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # O gráfico de silhueta varia de -1 a 1
    ax1.set_xlim([-0.1, 1])
    # O (n_clusters+1)*10 é para inserir espaço em branco entre os plots de silhueta
    ax1.set_ylim([0, len(X_scaled) + (n_clusters + 1) * 10])

    # Inicializar o clusterer
    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = clusterer.fit_predict(X_scaled)

    # O silhouette_score dá o valor médio para todas as amostras
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    results.append((n_clusters, silhouette_avg))
    
    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_k = n_clusters
        
    print(f"Para n_clusters = {n_clusters}, o score médio de silhueta é: {silhouette_avg:.4f}")

    # Calcular scores de silhueta para cada amostra
    sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Agregar scores de silhueta para amostras pertencentes ao cluster i e ordená-los
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Rotular os plots de silhueta com seus números de cluster no meio
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Calcular o novo y_lower para o próximo plot
        y_lower = y_upper + 10  # 10 para as 0 amostras

    ax1.set_title(f"Gráfico de Silhueta para {n_clusters} clusters")
    ax1.set_xlabel("Valores do coeficiente de silhueta")
    ax1.set_ylabel("Rótulo do Cluster")

    # A linha vertical para o score médio de silhueta de todos os valores
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Limpar os rótulos / ticks do eixo y
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2º Plot mostrando os clusters reais formados (usando as 2 primeiras features principais para visualização)
    # Nota: Isso é uma simplificação visual, já que temos muitas dimensões
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    
    # Usar PCA para reduzir para 2D apenas para visualização neste gráfico
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    ax2.scatter(X_pca[:, 0], X_pca[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Rotular os clusters
    centers = clusterer.cluster_centers_
    # Projetar centros também
    centers_pca = pca.transform(centers)
    
    ax2.scatter(centers_pca[:, 0], centers_pca[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers_pca):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title(f"Visualização dos dados clusterizados (PCA 2D) - K={n_clusters}")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")

    plt.suptitle(f"Análise de Silhueta para clusterização KMeans com n_clusters = {n_clusters}",
                 fontsize=14, fontweight='bold')

plt.show()

# ==============================================================================
# 3. RESULTADO FINAL E TREINAMENTO
# ==============================================================================

print("\n" + "="*80)
print("RESULTADO DA ANÁLISE")
print("="*80)

# Plotar resumo dos scores
ks = [r[0] for r in results]
scores = [r[1] for r in results]

plt.figure(figsize=(10, 6))
plt.plot(ks, scores, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Número de Clusters (K)')
plt.ylabel('Silhouette Score Médio')
plt.title('Comparação de Silhouette Scores')
plt.grid(True, alpha=0.3)
plt.axvline(x=best_k, color='r', linestyle='--', label=f'Melhor K={best_k}')
plt.legend()
plt.show()

print(f"Melhor número de clusters baseado na Silhueta: K = {best_k}")
print(f"Score máximo: {best_score:.4f}")

print(f"\nTreinando modelo final com K={best_k}...")
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster_silhouette'] = kmeans_final.fit_predict(X_scaled)

# Salvar
caminho_saida = os.path.join(caminho_atual, '..', '..', 'dataset', 'dados_clusterizados_silhouette.parquet')
df.to_parquet(caminho_saida)
print(f"Dataset com clusters salvo em: {caminho_saida}")