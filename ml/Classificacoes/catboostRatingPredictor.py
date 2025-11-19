from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

#Encontrar o caminho do dataset
caminho_atual = os.path.dirname(os.path.abspath(__file__))
caminho_dataset = os.path.join(caminho_atual, '..', '..', 'dataset', 'dados.parquet')

#Carregar o dataset
df = pd.read_parquet(caminho_dataset)

#Filtragem de no minimo 25 avaliacoes para predicao
df = df[df['avaliacao'] >= 25].copy()

#Criar a coluna de popularidade, 1 para Popular e 0 para Impopular
df['popularidade'] = np.where(df['rating'] >= 4.0, 1, 0)

#Criar features de GeneroPrimario e SubGenero a partir da coluna genero
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

#Features e variavel alvo
features = ['ano', 'paginas', 'querem_ler', 'autor', "editora", 'GeneroPrimario', 'SubGenero']
X = df[features]
y = df['popularidade']

#One-hot encoding para as colunas categoricas
X = pd.get_dummies(X, columns=['autor', 'editora', 'GeneroPrimario', 'SubGenero'], 
                   drop_first=True, 
                   prefix=['autor', 'editora', 'genero_primario', 'subgenero'])

# Limpar nomes das colunas para CatBoost/compatibilidade (remover caracteres especiais)
X.columns = X.columns.str.replace('[', '_', regex=False).str.replace(']', '_', regex=False).str.replace('"', '', regex=False).str.replace(':', '_', regex=False).str.replace(',', '_', regex=False).str.replace('{', '_', regex=False).str.replace('}', '_', regex=False)

# Remover colunas duplicadas que podem ter sido criadas pela limpeza de nomes
X = X.loc[:, ~X.columns.duplicated()]

# Dividir os dados em conjunto de treino e teste balanceando as classes
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Calcular pesos de classe para passar ao CatBoost
classes = np.unique(y_treino)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_treino)
class_weights = class_weights.tolist()

# Primeiro treinar o modelo base para obter feature importance
catboost_model = CatBoostClassifier(
    subsample = 1,
    random_strength=0.5,
    min_child_samples= 1,
    learning_rate=0.2,
    l2_leaf_reg= 10,
    iterations=1000,
    depth=8,
    colsample_bylevel=0.6,
    border_count=64,
    bagging_temperature=0.5,
    class_weights=class_weights,
    verbose=0
)
catboost_model.fit(X_treino, y_treino)

# Feature importance do modelo treinado
importance_values = catboost_model.get_feature_importance()
feature_importance = pd.DataFrame({
    'feature': X_treino.columns,
    'importance': importance_values
}).sort_values('importance', ascending=False)

# Agora calibrar o modelo já treinado
modelo = CalibratedClassifierCV(estimator=catboost_model, method='sigmoid', cv='prefit')
modelo.fit(X_treino, y_treino)

# Predições
y_pred = modelo.predict(X_teste)
y_pred_treino = modelo.predict(X_treino)

print("="*80)
print("FEATURE IMPORTANCE GERAL")
print("="*80)
print(feature_importance.head(30).to_string(index=False))
print("\n")

# ========== FEATURE IMPORTANCE POR CLASSE ==========

print("="*80)
print("FEATURE IMPORTANCE POR CLASSE")
print("="*80)

for classe in [0, 1]:
    # Criar um classificador binário para cada classe
    y_binary = (y_treino == classe).astype(int)
    dt_model = DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')
    dt_model.fit(X_treino, y_binary)
    
    # Feature importance para esta classe
    class_importance = pd.DataFrame({
        'feature': X_treino.columns,
        'importance': dt_model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    classe_nome = 'Impopular (rating < 4.0)' if classe == 0 else 'Popular (rating >= 4.0)'
    print(f"\nClasse {classe} - {classe_nome}:")
    print("-"*80)
    print(class_importance.to_string(index=False))
    print("\n")

# ========== CLASSIFICATION REPORT - TREINO ==========
print("="*80)
print("CLASSIFICATION REPORT - TREINO")
print("="*80)
print(classification_report(y_treino, y_pred_treino, 
                          target_names=['Impopular', 'Popular']))

# ========== MATRIZ DE CONFUSÃO - TREINO ==========
cm_treino = confusion_matrix(y_treino, y_pred_treino)

print("="*80)
print("MATRIZ DE CONFUSÃO - TREINO")
print("="*80)
print("\n                 Predito")
print("              Impopular  Popular")
print(f"Real Impopular    {cm_treino[0,0]:5d}     {cm_treino[0,1]:5d}")
print(f"     Popular      {cm_treino[1,0]:5d}     {cm_treino[1,1]:5d}")
print("\n")

# Visualização da matriz de confusão - TREINO
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_treino, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Impopular', 'Popular'],
            yticklabels=['Impopular', 'Popular'],
            cbar_kws={'label': 'Contagem'})
plt.title('Matriz de Confusão - TREINO\nCatBoost', fontsize=14, fontweight='bold')
plt.ylabel('Real', fontsize=12)
plt.xlabel('Predito', fontsize=12)
plt.tight_layout()
plt.show()

# ========== MATRIZ DE CONFUSÃO - TESTE ==========
cm = confusion_matrix(y_teste, y_pred)

print("="*80)
print("MATRIZ DE CONFUSÃO - TESTE")
print("="*80)
print("\n                 Predito")
print("              Impopular  Popular")
print(f"Real Impopular    {cm[0,0]:5d}     {cm[0,1]:5d}")
print(f"     Popular      {cm[1,0]:5d}     {cm[1,1]:5d}")
print("\n")

# Visualização da matriz de confusão - TESTE
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Impopular', 'Popular'],
            yticklabels=['Impopular', 'Popular'],
            cbar_kws={'label': 'Contagem'})
plt.title('Matriz de Confusão - TESTE\nCatBoost', fontsize=14, fontweight='bold')
plt.ylabel('Real', fontsize=12)
plt.xlabel('Predito', fontsize=12)
plt.tight_layout()
plt.show()

# ========== CLASSIFICATION REPORT - TESTE ==========
print("="*80)
print("CLASSIFICATION REPORT - TESTE")
print("="*80)
print(classification_report(y_teste, y_pred, 
                          target_names=['Impopular', 'Popular']))
