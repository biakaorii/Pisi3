"""
Gera relatório completo de divisão treino/teste e balanceamento.
Saída: 
  - outputs/reports/train_test_summary.csv
  - outputs/reports/train_test_distribution.png
  - outputs/reports/class_balance_comparison.png
  - outputs/reports/feature_statistics.csv
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter

# Configurações
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'dataset' / 'dados.parquet'
OUT_DIR = BASE_DIR / 'outputs' / 'reports'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Configurações dos splits
SPLITS_CONFIG = {
    'Classificação Abandono': {
        'test_size': 0.20,
        'stratify': True,
        'target_col': 'sera_abandonado',  # Ajuste conforme seu código
        'threshold': 0.5  # taxa_abandono > 0.5 = será abandonado
    },
    'Classificação Rating': {
        'test_size': 0.30,
        'stratify': True,
        'target_col': 'categoria_rating',  # Ajuste conforme seu código
        'bins': [0, 3.5, 4.2, 5.0],
        'labels': ['Baixo', 'Médio', 'Alto']
    }
}


def carregar_dados(path: Path) -> pd.DataFrame:
    """Carrega dataset e adiciona features derivadas."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado em: {path}")
    
    df = pd.read_parquet(path)
    
    # Adicionar features derivadas
    if 'abandonos' in df.columns and 'leram' in df.columns:
        df['taxa_abandono'] = df['abandonos'] / df['leram']
        df['taxa_abandono'] = df['taxa_abandono'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if 'avaliacao' in df.columns and 'leram' in df.columns:
        df['taxa_avaliacao'] = df['avaliacao'] / df['leram']
        df['taxa_avaliacao'] = df['taxa_avaliacao'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df


def criar_target_abandono(df: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    """Cria target binário para classificação de abandono."""
    if 'taxa_abandono' not in df.columns:
        raise ValueError("Coluna 'taxa_abandono' não encontrada")
    return (df['taxa_abandono'] > threshold).astype(int)


def criar_target_rating(df: pd.DataFrame, bins: list, labels: list) -> pd.Series:
    """Cria target categórico para classificação de rating."""
    if 'rating' not in df.columns:
        raise ValueError("Coluna 'rating' não encontrada")
    return pd.cut(df['rating'], bins=bins, labels=labels, include_lowest=True)


def gerar_split_info(df: pd.DataFrame, config: dict, nome_modelo: str) -> dict:
    """Gera informações sobre o split treino/teste."""
    
    # Criar target conforme modelo
    if 'abandono' in nome_modelo.lower():
        y = criar_target_abandono(df, config['threshold'])
    else:
        y = criar_target_rating(df, config['bins'], config['labels'])
    
    # Remover NaN do target
    mask = ~y.isna()
    df_clean = df[mask].copy()
    y_clean = y[mask]
    
    # Fazer split
    stratify_param = y_clean if config['stratify'] else None
    X_train, X_test, y_train, y_test = train_test_split(
        df_clean,
        y_clean,
        test_size=config['test_size'],
        random_state=42,
        stratify=stratify_param
    )
    
    # Calcular distribuições
    train_dist = Counter(y_train)
    test_dist = Counter(y_test)
    total_dist = Counter(y_clean)
    
    return {
        'nome_modelo': nome_modelo,
        'total_amostras': len(df_clean),
        'treino_amostras': len(X_train),
        'teste_amostras': len(X_test),
        'treino_pct': f"{(len(X_train)/len(df_clean))*100:.1f}%",
        'teste_pct': f"{(len(X_test)/len(df_clean))*100:.1f}%",
        'estrategia': 'Stratified' if config['stratify'] else 'Random',
        'train_dist': dict(train_dist),
        'test_dist': dict(test_dist),
        'total_dist': dict(total_dist),
        'y_train': y_train,
        'y_test': y_test,
        'y_total': y_clean
    }


def criar_tabela_resumo(splits_info: list) -> pd.DataFrame:
    """Cria tabela resumo dos splits."""
    dados = []
    
    # Adicionar linha para clusterização
    dados.append({
        'Modelo': 'KMeans (Clusterização)',
        'Treino': '100%',
        'Teste': 'N/A',
        'Total Amostras': splits_info[0]['total_amostras'],
        'Estratégia': 'Não Supervisionado'
    })
    
    # Adicionar linhas para classificações
    for info in splits_info:
        dados.append({
            'Modelo': info['nome_modelo'],
            'Treino': info['treino_pct'],
            'Teste': info['teste_pct'],
            'Total Amostras': info['total_amostras'],
            'Estratégia': info['estrategia']
        })
    
    return pd.DataFrame(dados)


def plotar_distribuicao_splits(splits_info: list, saida: Path):
    """Plota distribuição de amostras entre treino/teste."""
    fig, axes = plt.subplots(1, len(splits_info), figsize=(14, 5))
    
    if len(splits_info) == 1:
        axes = [axes]
    
    for idx, info in enumerate(splits_info):
        ax = axes[idx]
        
        # Dados para o gráfico
        conjuntos = ['Treino', 'Teste']
        valores = [info['treino_amostras'], info['teste_amostras']]
        cores = ['#3498db', '#e74c3c']
        
        # Gráfico de barras
        bars = ax.bar(conjuntos, valores, color=cores, alpha=0.7, edgecolor='black')
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height):,}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_title(info['nome_modelo'], fontsize=12, fontweight='bold')
        ax.set_ylabel('Número de Amostras', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, max(valores) * 1.15)
    
    plt.suptitle('Distribuição Treino/Teste por Modelo', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(saida, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Salvo: {saida}")


def plotar_balanceamento_classes(splits_info: list, saida: Path):
    """Plota comparação de balanceamento de classes."""
    fig, axes = plt.subplots(len(splits_info), 3, figsize=(16, 5*len(splits_info)))
    
    if len(splits_info) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, info in enumerate(splits_info):
        # Subplot 1: Distribuição Original (Total)
        ax1 = axes[idx, 0]
        classes = list(info['total_dist'].keys())
        valores = list(info['total_dist'].values())
        
        bars = ax1.bar(range(len(classes)), valores, color='#95a5a6', alpha=0.7, edgecolor='black')
        ax1.set_xticks(range(len(classes)))
        ax1.set_xticklabels(classes)
        ax1.set_title('Distribuição Original', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Quantidade', fontsize=10)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, val in zip(bars, valores):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:,}', ha='center', va='bottom', fontsize=9)
        
        # Subplot 2: Conjunto de Treino
        ax2 = axes[idx, 1]
        train_classes = list(info['train_dist'].keys())
        train_valores = list(info['train_dist'].values())
        
        bars = ax2.bar(range(len(train_classes)), train_valores, color='#3498db', alpha=0.7, edgecolor='black')
        ax2.set_xticks(range(len(train_classes)))
        ax2.set_xticklabels(train_classes)
        ax2.set_title('Conjunto de Treino', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Quantidade', fontsize=10)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, val in zip(bars, train_valores):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:,}', ha='center', va='bottom', fontsize=9)
        
        # Subplot 3: Conjunto de Teste
        ax3 = axes[idx, 2]
        test_classes = list(info['test_dist'].keys())
        test_valores = list(info['test_dist'].values())
        
        bars = ax3.bar(range(len(test_classes)), test_valores, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax3.set_xticks(range(len(test_classes)))
        ax3.set_xticklabels(test_classes)
        ax3.set_title('Conjunto de Teste', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Quantidade', fontsize=10)
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        
        for bar, val in zip(bars, test_valores):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:,}', ha='center', va='bottom', fontsize=9)
        
        # Label do modelo no lado esquerdo
        fig.text(0.02, 0.5 - (idx * (1/len(splits_info))), info['nome_modelo'],
                va='center', rotation='vertical', fontsize=12, fontweight='bold')
    
    plt.suptitle('Análise de Balanceamento de Classes', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0.03, 0, 1, 0.99])
    plt.savefig(saida, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✓ Salvo: {saida}")


def calcular_estatisticas_features(df: pd.DataFrame, splits_info: list) -> pd.DataFrame:
    """Calcula estatísticas das features numéricas por conjunto."""
    features_numericas = ['lendo', 'leram', 'abandonos', 'rating', 'avaliacao', 
                         'taxa_abandono', 'taxa_avaliacao']
    
    # Filtrar apenas as que existem
    features_numericas = [f for f in features_numericas if f in df.columns]
    
    estatisticas = []
    
    for info in splits_info:
        # Reconstruir os índices
        y_total = info['y_total']
        indices_treino = y_total.index[:info['treino_amostras']]
        indices_teste = y_total.index[info['treino_amostras']:]
        
        for feature in features_numericas:
            dados_treino = df.loc[indices_treino, feature]
            dados_teste = df.loc[indices_teste, feature]
            
            estatisticas.append({
                'Modelo': info['nome_modelo'],
                'Feature': feature,
                'Conjunto': 'Treino',
                'Média': dados_treino.mean(),
                'Desvio Padrão': dados_treino.std(),
                'Mínimo': dados_treino.min(),
                'Máximo': dados_treino.max()
            })
            
            estatisticas.append({
                'Modelo': info['nome_modelo'],
                'Feature': feature,
                'Conjunto': 'Teste',
                'Média': dados_teste.mean(),
                'Desvio Padrão': dados_teste.std(),
                'Mínimo': dados_teste.min(),
                'Máximo': dados_teste.max()
            })
    
    return pd.DataFrame(estatisticas)


def main():
    print("=" * 60)
    print("RELATÓRIO DE TREINO/TESTE E BALANCEAMENTO")
    print("=" * 60)
    
    # Carregar dados
    print(f"\n📂 Lendo dataset: {DATA_PATH}")
    df = carregar_dados(DATA_PATH)
    print(f"✓ Dataset carregado: {len(df):,} amostras")
    
    # Gerar informações dos splits
    print("\n📊 Gerando informações dos splits...")
    splits_info = []
    
    for nome_modelo, config in SPLITS_CONFIG.items():
        try:
            info = gerar_split_info(df, config, nome_modelo)
            splits_info.append(info)
            print(f"✓ {nome_modelo}: {info['total_amostras']:,} amostras ({info['treino_pct']} treino / {info['teste_pct']} teste)")
        except Exception as e:
            print(f"✗ Erro ao processar {nome_modelo}: {e}")
    
    if not splits_info:
        print("✗ Nenhum split foi gerado com sucesso!")
        return
    
    # Criar tabela resumo
    print("\n📋 Criando tabela resumo...")
    tabela_resumo = criar_tabela_resumo(splits_info)
    saida_resumo = OUT_DIR / 'train_test_summary.csv'
    tabela_resumo.to_csv(saida_resumo, index=False)
    print(f"✓ Salvo: {saida_resumo}")
    print("\nTabela Resumo:")
    print(tabela_resumo.to_string(index=False))
    
    # Plotar distribuição dos splits
    print("\n📈 Gerando gráfico de distribuição...")
    saida_dist = OUT_DIR / 'train_test_distribution.png'
    plotar_distribuicao_splits(splits_info, saida_dist)
    
    # Plotar balanceamento de classes
    print("\n📊 Gerando gráfico de balanceamento...")
    saida_balance = OUT_DIR / 'class_balance_comparison.png'
    plotar_balanceamento_classes(splits_info, saida_balance)
    
    # Calcular estatísticas das features
    print("\n📐 Calculando estatísticas das features...")
    stats = calcular_estatisticas_features(df, splits_info)
    saida_stats = OUT_DIR / 'feature_statistics.csv'
    stats.to_csv(saida_stats, index=False)
    print(f"✓ Salvo: {saida_stats}")
    print("\n" + "=" * 60)
    print("✓ RELATÓRIO COMPLETO GERADO COM SUCESSO!")
    print("=" * 60)
    print(f"\nArquivos gerados em: {OUT_DIR}/")
    print("  • train_test_summary.csv")
    print("  • train_test_distribution.png")
    print("  • class_balance_comparison.png")
    print("  • feature_statistics.csv")

if __name__ == '__main__':
    main()