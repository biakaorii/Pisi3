import pickle
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import os

print("="*80)
print("ANÁLISE SHAP - XGBoost Rating Predictor")
print("="*80)

# Encontrar o caminho do dataset e modelo
caminho_atual = os.path.dirname(os.path.abspath(__file__))
caminho_dataset = os.path.join(caminho_atual, '..', '..', 'dataset', 'dados.parquet')
caminho_modelo = os.path.join(caminho_atual, 'xgb_rating_predictor.pkl')

# Criar diretório de saída
output_dir = os.path.join(caminho_atual, 'saida')
os.makedirs(output_dir, exist_ok=True)

# Carregar o dataset
print("\n[1/6] Carregando e preparando dados...")
df = pd.read_parquet(caminho_dataset)
df = df[df['avaliacao'] >= 25].copy()
df['popularidade'] = np.where(df['rating'] >= 4.0, 1, 0)

# Criar features de GeneroPrimario e SubGenero
def extrair_genero_primario(genero_str):
    if pd.isna(genero_str) or genero_str == 'Desconhecido':
        return 'Desconhecido'
    partes = str(genero_str).split('/')
    if len(partes) > 0:
        return partes[0].strip()
    return 'Desconhecido'

def extrair_subgenero(genero_str):
    if pd.isna(genero_str) or genero_str == 'Desconhecido':
        return 'Desconhecido'
    partes = str(genero_str).split('/')
    if len(partes) > 1:
        return partes[1].strip()
    return 'Desconhecido'

df['GeneroPrimario'] = df['genero'].apply(extrair_genero_primario)
df['SubGenero'] = df['genero'].apply(extrair_subgenero)

# Features
features = ['ano', 'paginas', 'querem_ler', 'autor', "editora", 'GeneroPrimario', 'SubGenero']
X = df[features]
y = df['popularidade']

# One-hot encoding
X = pd.get_dummies(X, columns=['autor', 'editora', 'GeneroPrimario', 'SubGenero'], 
                   drop_first=True, 
                   prefix=['autor', 'editora', 'genero_primario', 'subgenero'])

# Limpar nomes das colunas
X.columns = X.columns.str.replace('[', '_', regex=False).str.replace(']', '_', regex=False).str.replace('<', '_', regex=False).str.replace('>', '_', regex=False).str.replace('"', '', regex=False).str.replace(':', '_', regex=False).str.replace(',', '_', regex=False).str.replace('{', '_', regex=False).str.replace('}', '_', regex=False)
X = X.loc[:, ~X.columns.duplicated()]

print(f"   Dataset: {len(X)} amostras, {len(X.columns)} features")

# Carregar o modelo
print("\n[2/6] Carregando modelo treinado...")
with open(caminho_modelo, 'rb') as f:
    modelo = pickle.load(f)

print("   Modelo carregado com sucesso")

# Verificar se já existem SHAP values calculados
shap_cache_file = os.path.join(output_dir, 'shap_values_cache.pkl')

# Verificar se já existem SHAP values calculados
shap_cache_file = os.path.join(output_dir, 'shap_values_cache.pkl')
cache_loaded = False

if os.path.exists(shap_cache_file):
    try:
        print("\n[3/6] Carregando SHAP values do cache...")
        with open(shap_cache_file, 'rb') as f:
            cache_data = pickle.load(f)
            # Reconstruir shap.Explanation a partir dos dados salvos
            base_vals = cache_data['base_values']
            # Garantir que base_values tenha a forma correta (um valor por amostra)
            num_samples = cache_data['shap_values_array'].shape[0]
            if isinstance(base_vals, (int, float, np.number)):
                base_vals = np.full(num_samples, float(base_vals))
            elif isinstance(base_vals, np.ndarray):
                if base_vals.size == 1:
                    base_vals = np.full(num_samples, float(base_vals.item()))
                elif len(base_vals) != num_samples:
                    # Se o tamanho não bate, repetir o primeiro valor
                    base_vals = np.full(num_samples, float(base_vals[0]))
            
            shap_values = shap.Explanation(
                values=cache_data['shap_values_array'],
                base_values=base_vals,
                data=cache_data['data'],
                feature_names=cache_data['feature_names']
            )
            X_sample = cache_data['X_sample']
            expected_value = cache_data.get('expected_value', base_vals[0] if len(base_vals) > 0 else 0.5)
        print("   ✓ SHAP values carregados do cache!")
        cache_loaded = True
    except (EOFError, pickle.UnpicklingError, KeyError, TypeError) as e:
        print(f"   ⚠️ Cache corrompido ({e}), recalculando...")
        os.remove(shap_cache_file)
        cache_loaded = False
        
if not cache_loaded:
    # Criar amostra para SHAP (para performance)
    print("\n[3/6] Criando SHAP explainer...")
    X_sample = shap.sample(X, 100, random_state=42)

    # Usar KernelExplainer ao invés de TreeExplainer para evitar bug do XGBoost
    # KernelExplainer funciona com qualquer modelo através de predições
    background = shap.sample(X, 50, random_state=42)
    explainer = shap.KernelExplainer(modelo.predict_proba, background)
    shap_values_array = explainer.shap_values(X_sample)

    # Converter para formato Explanation do SHAP moderno
    # Para classificação binária, usamos os valores da classe positiva (índice 1)
    shap_vals = shap_values_array[1] if isinstance(shap_values_array, list) else shap_values_array[:, :, 1]
    
    # Criar base_values - deve ser um array com um valor para cada amostra
    exp_val = explainer.expected_value
    if isinstance(exp_val, (list, np.ndarray)):
        base_val_scalar = float(exp_val[1]) if len(exp_val) > 1 else float(exp_val[0])
    else:
        base_val_scalar = float(exp_val)
    
    base_vals_array = np.full(len(X_sample), base_val_scalar)
    
    shap_values = shap.Explanation(
        values=shap_vals,
        base_values=base_vals_array,
        data=X_sample.values,
        feature_names=X_sample.columns.tolist()
    )
    print("   SHAP values calculados")
    
    # Salvar no cache imediatamente (apenas dados serializáveis)
    print("   💾 Salvando SHAP values no cache...")
    cache_data = {
        'shap_values_array': np.array(shap_values.values),
        'base_values': base_val_scalar,  # Salvar apenas o valor escalar
        'data': np.array(shap_values.data),
        'feature_names': list(shap_values.feature_names),
        'X_sample': X_sample,
        'expected_value': base_val_scalar
    }
    with open(shap_cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    print("   ✓ Cache salvo com sucesso!")

# ========== EXPLICAÇÕES GLOBAIS ==========
print("\n[4/6] Gerando explicações globais...")

# 1. Feature Importance (bar plot)
print("   - Feature importance (top 20)...")
plt.figure(figsize=(14, 10))
shap.plots.bar(shap_values, max_display=20, show=False)
plt.title('SHAP Feature Importance - Top 20 Features', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('mean(|SHAP value|)', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_feature_importance.png'), dpi=400, bbox_inches='tight')
plt.close()

# 2. Beeswarm Plot (distribuição dos SHAP values)
print("   - Beeswarm plot (distribuição)...")
plt.figure(figsize=(14, 12))
shap.plots.beeswarm(shap_values, max_display=20, show=False)
plt.title('SHAP Beeswarm Plot - Distribuição dos Impactos', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('SHAP value (impact on model output)', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_beeswarm_global.png'), dpi=400, bbox_inches='tight')
plt.close()

# 3. Summary Plot (alternativo)
print("   - Summary plot...")
plt.figure(figsize=(14, 12))
shap.summary_plot(shap_values, X_sample, show=False, max_display=20, plot_type='dot')
plt.title('SHAP Summary Plot - Visão Geral', fontsize=18, fontweight='bold', pad=15)
plt.xlabel('SHAP value (impact on model output)', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_summary_global.png'), dpi=400, bbox_inches='tight')
plt.close()

# ========== EXPLICAÇÕES POR CLASSE ==========
print("\n[5/6] Gerando explicações por classe...")

# Separar predições por classe
predictions = modelo.predict(X_sample)
popular_mask = predictions == 1
impopular_mask = predictions == 0

print(f"   - Classe Popular: {popular_mask.sum()} amostras")
print(f"   - Classe Impopular: {impopular_mask.sum()} amostras")

# Feature importance por classe
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))

# Classe Impopular
if impopular_mask.sum() > 0:
    mean_shap_impopular = np.abs(shap_values.values[impopular_mask]).mean(0)
    top_idx = np.argsort(mean_shap_impopular)[-20:][::-1]
    ax1.barh(range(20), mean_shap_impopular[top_idx], color='#FF6B6B', alpha=0.8)
    ax1.set_yticks(range(20))
    ax1.set_yticklabels([X_sample.columns[i][:40] for i in top_idx], fontsize=11)
    ax1.set_xlabel('Mean |SHAP value|', fontsize=14, fontweight='bold')
    ax1.set_title('Features Mais Importantes - IMPOPULAR (rating < 4.0)', fontsize=15, fontweight='bold', pad=15)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

# Classe Popular
if popular_mask.sum() > 0:
    mean_shap_popular = np.abs(shap_values.values[popular_mask]).mean(0)
    top_idx = np.argsort(mean_shap_popular)[-20:][::-1]
    ax2.barh(range(20), mean_shap_popular[top_idx], color='#4ECDC4', alpha=0.8)
    ax2.set_yticks(range(20))
    ax2.set_yticklabels([X_sample.columns[i][:40] for i in top_idx], fontsize=11)
    ax2.set_xlabel('Mean |SHAP value|', fontsize=14, fontweight='bold')
    ax2.set_title('Features Mais Importantes - POPULAR (rating >= 4.0)', fontsize=15, fontweight='bold', pad=15)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_multiclass_importance.png'), dpi=400, bbox_inches='tight')
plt.close()

# ========== EXPLICAÇÕES LOCAIS ==========
print("\n[6/6] Gerando explicações locais (force plots)...")

# Selecionar exemplos representativos de cada classe
exemplos = []

# Exemplo Popular
if popular_mask.sum() > 0:
    idx_popular = np.where(popular_mask)[0][0]
    exemplos.append(('popular', idx_popular))

# Exemplo Impopular
if impopular_mask.sum() > 0:
    idx_impopular = np.where(impopular_mask)[0][0]
    exemplos.append(('impopular', idx_impopular))

# Gerar force plots
for classe, idx in exemplos:
    print(f"   - Force plot para exemplo {classe}...")
    
    # Waterfall plot
    plt.figure(figsize=(16, 10))
    shap.plots.waterfall(shap_values[idx], max_display=15, show=False)
    plt.title(f'Explicação Local - Livro {classe.upper()}', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('SHAP value', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'shap_local_{classe}_waterfall.png'), dpi=400, bbox_inches='tight')
    plt.close()
    
    # Force plot interativo (HTML)
    base_val = shap_values.base_values if np.isscalar(shap_values.base_values) else shap_values.base_values[idx]
    force_plot = shap.force_plot(
        base_val, 
        shap_values.values[idx], 
        X_sample.iloc[idx],
        matplotlib=False,
        show=False
    )
    shap.save_html(os.path.join(output_dir, f'shap_local_{classe}_force.html'), force_plot)

# Gerar múltiplos exemplos em HTML
print("   - Force plot com múltiplos exemplos...")
num_examples = min(100, len(X_sample))
# Se base_values for escalar, repetir para cada exemplo
if np.isscalar(shap_values.base_values):
    base_vals = np.full(num_examples, shap_values.base_values)
else:
    base_vals = shap_values.base_values[:num_examples]

force_plot_multi = shap.force_plot(
    base_vals,
    shap_values.values[:num_examples],
    X_sample.iloc[:num_examples],
    matplotlib=False,
    show=False
)
shap.save_html(os.path.join(output_dir, 'shap_local_multi_examples.html'), force_plot_multi)

# ========== SALVAR DADOS SHAP ==========
print("\n[EXTRA] Salvando dados SHAP para uso posterior...")

# Salvar valores SHAP e explainer
# Usar expected_value do cache se disponível, senão do explainer
if cache_loaded:
    base_val_to_save = expected_value
else:
    exp_val = explainer.expected_value
    if isinstance(exp_val, (list, np.ndarray)):
        base_val_to_save = float(exp_val[1]) if len(exp_val) > 1 else float(exp_val[0])
    else:
        base_val_to_save = float(exp_val)

shap_data = {
    'shap_values': shap_values.values,
    'base_value': base_val_to_save,
    'data': X_sample,
    'feature_names': X_sample.columns.tolist()
}

with open(os.path.join(output_dir, 'shap_data.pkl'), 'wb') as f:
    pickle.dump(shap_data, f)

print("\n" + "="*80)
print("ANÁLISE SHAP CONCLUÍDA COM SUCESSO!")
print("="*80)
print(f"\nTodos os arquivos salvos em: {output_dir}\n")
print("Arquivos gerados:")
print("\n📊 EXPLICAÇÕES GLOBAIS:")
print("   ✓ shap_feature_importance.png - Ranking das features mais importantes")
print("   ✓ shap_beeswarm_global.png - Distribuição dos impactos (beeswarm)")
print("   ✓ shap_summary_global.png - Visão geral (summary plot)")
print("   ✓ shap_multiclass_importance.png - Importância por classe (Popular/Impopular)")
print("\n🔍 EXPLICAÇÕES LOCAIS:")
print("   ✓ shap_local_popular_waterfall.png - Exemplo waterfall (Popular)")
print("   ✓ shap_local_impopular_waterfall.png - Exemplo waterfall (Impopular)")
print("   ✓ shap_local_popular_force.html - Force plot interativo (Popular)")
print("   ✓ shap_local_impopular_force.html - Force plot interativo (Impopular)")
print("   ✓ shap_local_multi_examples.html - Force plots múltiplos exemplos")
print("\n💾 DADOS:")
print("   ✓ shap_data.pkl - Valores SHAP salvos para uso posterior")
print("\n" + "="*80)
