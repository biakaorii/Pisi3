"""
Gera um heatmap de correlação (Spearman) entre métricas principais
do dataset `dataset/dados.parquet`.

Saída: `outputs/eda/correlation_heatmap_spearman.png`
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'dataset' / 'dados.parquet'
OUT_DIR = BASE_DIR / 'outputs' / 'eda'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def carregar_dados(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado em: {path}")
    df = pd.read_parquet(path)
    return df


def adicionar_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'abandonos' in df.columns and 'leram' in df.columns and 'taxa_abandono' not in df.columns:
        df['taxa_abandono'] = df['abandonos'] / df['leram']
    if 'avaliacao' in df.columns and 'leram' in df.columns and 'taxa_avaliacao' not in df.columns:
        df['taxa_avaliacao'] = df['avaliacao'] / df['leram']
    # Limpeza básica
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def selecionar_numericas(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number]).copy()
    # Remove colunas completamente vazias ou constantes (sem variância)
    nunique = num.nunique(dropna=True)
    variaveis_validas = nunique[nunique > 1].index
    num = num[variaveis_validas]
    return num


def plotar_heatmap(corr: pd.DataFrame, saida: Path, titulo: str = 'Correlação (Spearman)') -> None:
    plt.figure(figsize=(12, 10))
    sns.set(style='whitegrid')
    # Máscara para triângulo superior
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8},
    )
    plt.title(titulo)
    plt.tight_layout()
    plt.savefig(saida, dpi=200)
    plt.close()
    print(f"Salvo: {saida}")


def main():
    print(f"Lendo dataset: {DATA_PATH}")
    df = carregar_dados(DATA_PATH)
    df = adicionar_derivadas(df)
    num = selecionar_numericas(df)

    if num.shape[1] < 2:
        raise RuntimeError("Dados numéricos insuficientes para correlação.")

    # Spearman captura relações monotônicas e é mais robusto a outliers
    corr = num.corr(method='spearman')
    saida = OUT_DIR / 'correlation_heatmap_spearman.png'
    plotar_heatmap(corr, saida)


if __name__ == '__main__':
    main()

