import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

# Encontrar o caminho do dataset
caminho_atual = os.path.dirname(os.path.abspath(__file__))
caminho_dataset = os.path.join(caminho_atual, '..', '..', 'dataset', 'dados.parquet')

# Carregar o dataset
df = pd.read_parquet(caminho_dataset)

# Verificar colunas disponíveis (para debug se necessário)
print("Colunas disponíveis:", df.columns.tolist())

# Criar features de engajamento
df['engajamento_total'] = df['leram'] + df['lendo'] + df['querem_ler']

df['taxa_abandono'] = df['abandonos'] / (df['leram'] + 1)
df['taxa_abandono'] = df['taxa_abandono'].replace([np.inf, -np.inf], np.nan).fillna(0)

df['taxa_conclusao'] = df['leram'] / (df['leram'] + df['abandonos'] + 1)
df['taxa_conclusao'] = df['taxa_conclusao'].replace([np.inf, -np.inf], np.nan).fillna(0)

df['popularidade'] = np.log1p(df['avaliacao'])  # Log para normalizar valores extremos

df['interesse_futuro'] = df['querem_ler'] / (df['engajamento_total'] + 1)
df['interesse_futuro'] = df['interesse_futuro'].replace([np.inf, -np.inf], np.nan).fillna(0)

# Selecionar features para clustering
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

print("="*80)
print("CLUSTERIZAÇÃO DE ENGAJAMENTO DE LEITORES - MÉTODO DO COTOVELO")
print("="*80)
print(f"Total de livros analisados: {len(df):,}")
print(f"Features utilizadas: {len(features)}")
print("\n")

# ========== MÉTODO DO COTOVELO ==========
print("="*80)
print("MÉTODO DO COTOVELO (ELBOW METHOD)")
print("="*80)
print("Calculando métricas para diferentes valores de K (2 a 30)...\n")

# Calcular métricas para K de 2 a 30
K_range = range(2, 31)
inertias = []
silhouette_scores = []
davies_bouldin_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
    kmeans.fit(X_scaled)
    
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    davies_bouldin_scores.append(davies_bouldin_score(X_scaled, kmeans.labels_))
    
    print(f"K={k:2d} | Inércia: {kmeans.inertia_:10.2f} | Silhouette: {silhouette_scores[-1]:.4f} | Davies-Bouldin: {davies_bouldin_scores[-1]:.4f}")

# Criar figura com 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Método do Cotovelo - Engajamento de Leitores (K=2 a 30)', fontsize=16, fontweight='bold')

# 1. Gráfico de Inércia (Elbow)
axes[0, 0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=6)
axes[0, 0].set_xlabel('Número de Clusters (K)', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Inércia (WCSS)', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Método do Cotovelo - Inércia', fontsize=13, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, linestyle='--')
axes[0, 0].set_xticks(range(2, 31, 2)) # Ajuste ticks para não poluir

# Calcular taxa de redução da inércia
inertia_diffs = np.diff(inertias)
inertia_diffs_pct = (inertia_diffs / inertias[:-1]) * 100

# 2. Taxa de Redução da Inércia
axes[0, 1].plot(list(K_range)[1:], np.abs(inertia_diffs_pct), 'ro-', linewidth=2, markersize=6)
axes[0, 1].set_xlabel('Número de Clusters (K)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Redução da Inércia (%)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Taxa de Redução da Inércia', fontsize=13, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, linestyle='--')
axes[0, 1].set_xticks(range(2, 31, 2))

# 3. Silhouette Score
axes[1, 0].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=6)
axes[1, 0].set_xlabel('Número de Clusters (K)', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Silhouette Score', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Silhouette Score por K (maior é melhor)', fontsize=13, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, linestyle='--')
axes[1, 0].set_xticks(range(2, 31, 2))
axes[1, 0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Boa separação (0.5)')
axes[1, 0].legend()

# 4. Davies-Bouldin Index
axes[1, 1].plot(K_range, davies_bouldin_scores, 'mo-', linewidth=2, markersize=6)
axes[1, 1].set_xlabel('Número de Clusters (K)', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Davies-Bouldin Index', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Davies-Bouldin Index por K (menor é melhor)', fontsize=13, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, linestyle='--')
axes[1, 1].set_xticks(range(2, 31, 2))

plt.tight_layout()
plt.show()

# Detectar cotovelo automaticamente
def find_elbow(inertias):
    """Encontra o ponto do cotovelo usando a segunda derivada"""
    n = len(inertias)
    x = np.arange(n)
    
    # Normalizar para [0, 1]
    inertias_norm = (inertias - np.min(inertias)) / (np.max(inertias) - np.min(inertias))
    
    # Calcular primeira derivada
    first_derivative = np.gradient(inertias_norm)
    
    # Calcular segunda derivada
    second_derivative = np.gradient(first_derivative)
    
    # O cotovelo é onde a segunda derivada é máxima
    elbow_idx = np.argmax(second_derivative)
    
    return elbow_idx

elbow_idx = find_elbow(inertias)
optimal_k_elbow = list(K_range)[elbow_idx]

print(f"\n{'='*80}")
print("ANÁLISE DO COTOVELO")
print("="*80)
print(f"Cotovelo detectado automaticamente em: K = {optimal_k_elbow}")
print(f"\nMelhor Silhouette Score: K = {list(K_range)[np.argmax(silhouette_scores)]} (score: {max(silhouette_scores):.4f})")
print(f"Melhor Davies-Bouldin: K = {list(K_range)[np.argmin(davies_bouldin_scores)]} (score: {min(davies_bouldin_scores):.4f})")

# Recomendação baseada em múltiplas métricas
print(f"\n{'='*80}")
print("RECOMENDAÇÃO BASEADA EM MÚLTIPLAS MÉTRICAS")
print("="*80)

k_recommendations = {
    'elbow': optimal_k_elbow,
    'silhouette': list(K_range)[np.argmax(silhouette_scores)],
    'davies_bouldin': list(K_range)[np.argmin(davies_bouldin_scores)]
}

from collections import Counter
k_votes = Counter(k_recommendations.values())
recommended_k = k_votes.most_common(1)[0][0]

print(f"✓ Método do Cotovelo sugere: K = {k_recommendations['elbow']}")
print(f"✓ Silhouette Score sugere: K = {k_recommendations['silhouette']}")
print(f"✓ Davies-Bouldin sugere: K = {k_recommendations['davies_bouldin']}")
print(f"\n🎯 RECOMENDAÇÃO FINAL: K = {recommended_k}")
print(f"   (Baseado na convergência das métricas)")

# Treinar modelo com K recomendado
print(f"\n{'='*80}")
print(f"TREINANDO MODELO FINAL COM K = {recommended_k}")
print("="*80)

kmeans_final = KMeans(n_clusters=recommended_k, random_state=42, n_init=10, max_iter=300)
df['cluster'] = kmeans_final.fit_predict(X_scaled)

# Avaliar clusters finais
silhouette_final = silhouette_score(X_scaled, df['cluster'])
davies_bouldin_final = davies_bouldin_score(X_scaled, df['cluster'])

print(f"Silhouette Score: {silhouette_final:.4f}")
print(f"Davies-Bouldin Score: {davies_bouldin_final:.4f}")
print(f"Inércia: {kmeans_final.inertia_:.2f}\n")

# Salvar resultados
caminho_saida = os.path.join(caminho_atual, '..', '..', 'dataset', 'cluster_engajamento_elbow.parquet')
df.to_parquet(caminho_saida, index=False)
print(f"✅ Dataset com clusters salvo em: {caminho_saida}")

print("\n" + "="*80)
print("ANÁLISE CONCLUÍDA COM SUCESSO! 🎉")
print("="*80)