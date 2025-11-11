import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / 'dataset' / 'dados.parquet'
OUT_DIR = BASE_DIR / 'outputs' / 'eda_gender'
OUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set(style='whitegrid')


def carregar():
    print(f"Lendo dataset: {DATA_PATH}")
    df = pd.read_parquet(DATA_PATH)
    return df


def explode_generos(df):
    # Gênero pode estar em formato "A / B / C" - normalizamos e explodimos
    if 'genero' not in df.columns:
        df['genero'] = None
    genero_series = (
        df[['genero', 'male', 'female', 'leram']]
        .fillna({'genero': ''})
        .assign(genero=lambda d: d['genero'].str.split('/'))
        .explode('genero')
    )
    # Limpar espaços
    genero_series['genero'] = genero_series['genero'].astype(str).str.strip()
    genero_series = genero_series[genero_series['genero'] != '']
    return genero_series


def gerar_estatisticas_por_genero(df_gen):
    # Soma de leitores por gênero (male/female) e agregados
    agg = (
        df_gen.groupby('genero')
        .agg(
            male_sum=pd.NamedAgg(column='male', aggfunc='sum'),
            female_sum=pd.NamedAgg(column='female', aggfunc='sum'),
            leram_sum=pd.NamedAgg(column='leram', aggfunc='sum'),
            count=pd.NamedAgg(column='genero', aggfunc='count')
        )
        .sort_values('leram_sum', ascending=False)
    )
    agg['total_gender'] = agg['male_sum'] + agg['female_sum']
    agg['female_ratio'] = agg['female_sum'] / agg['total_gender'].replace(0, pd.NA)
    agg = agg.fillna(0)
    return agg


def plot_genre_counts(agg):
    top = agg.head(20).copy()
    plt.figure(figsize=(10, 8))
    top['leram_sum'].plot(kind='barh', color='steelblue')
    plt.gca().invert_yaxis()
    plt.title('Top 20 gêneros por número de leitores (leram)')
    plt.xlabel('Total de leitores (leram)')
    plt.tight_layout()
    p = OUT_DIR / 'genres_gender_counts.png'
    plt.savefig(p, dpi=200)
    print(f'Salvo: {p}')
    plt.close()


def plot_gender_preferences(agg):
    """Plota preferências de gênero literário por gênero do leitor (M/F)"""
    
    # Filtrar apenas gêneros com número significativo de leitores
    filtered = agg[agg['total_gender'] > 50].copy()
    
    #gêneros preferidos por mulheres
    plt.figure(figsize=(10, 8))
    female_df = filtered.nlargest(20, 'female_sum')[['female_sum']]
    female_df['female_sum'].plot(kind='barh', color='steelblue')
    plt.gca().invert_yaxis()
    plt.title('Top 20 Gêneros Literários Mais Lidos por Mulheres')
    plt.xlabel('Total de Leitoras')
    
    # Adicionar valores nas barras
    for i, v in enumerate(female_df['female_sum']):
        plt.text(v, i, f' {int(v):,}', va='center')
    
    plt.tight_layout()
    p = OUT_DIR / 'female_preferred_genres.png'
    plt.savefig(p, dpi=200)
    print(f'Salvo: {p}')
    plt.close()
    
    # Top 20 gêneros preferidos por homens
    plt.figure(figsize=(10, 8))
    male_df = filtered.nlargest(20, 'male_sum')[['male_sum']]
    male_df['male_sum'].plot(kind='barh', color='steelblue')
    plt.gca().invert_yaxis()
    plt.title('Top 20 Gêneros Literários Mais Lidos por Homens')
    plt.xlabel('Total de Leitores')
    
    # Adicionar valores nas barras
    for i, v in enumerate(male_df['male_sum']):
        plt.text(v, i, f' {int(v):,}', va='center')
    
    plt.tight_layout()
    p = OUT_DIR / 'male_preferred_genres.png'
    plt.savefig(p, dpi=200)
    print(f'Salvo: {p}')
    plt.close()

def main():
    df = carregar()
    df_gen = explode_generos(df)
    agg = gerar_estatisticas_por_genero(df_gen)
    plot_genre_counts(agg)
    plot_gender_preferences(agg)


if __name__ == '__main__':
    main()
