import os
import json
import joblib
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# Encontrar o caminho do dataset
caminho_atual = os.path.dirname(os.path.abspath(__file__))
caminho_dataset = os.path.join(caminho_atual, '..', '..', 'dataset', 'dados.parquet')

# Carregar o dataset
df = pd.read_parquet(caminho_dataset)

# Filtragem de no minimo 25 avaliacoes para predicao
df = df[df['avaliacao'] >= 25].copy()

# Criar a coluna de popularidade, 1 para Popular e 0 para Impopular
df['popularidade'] = np.where(df['rating'] >= 4.0, 1, 0)

# Features e variavel alvo
features = ['ano', 'paginas', 'querem_ler', 'autor', 'editora']
X = df[features]
y = df['popularidade']

# One-hot encoding para as colunas categóricas
X = pd.get_dummies(X, columns=['autor', 'editora'], drop_first=True)

# Limpar nomes das colunas para CatBoost (remover caracteres especiais)
X.columns = X.columns.str.replace('[', '_', regex=False).str.replace(']', '_', regex=False).str.replace('<', '_', regex=False).str.replace('>', '_', regex=False).str.replace('"', '', regex=False).str.replace(':', '_', regex=False).str.replace(',', '_', regex=False).str.replace('{', '_', regex=False).str.replace('}', '_', regex=False)

# Dividir treino/teste stratificado
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Calcular class_weights para balanceamento
classes = np.unique(y_treino)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_treino)
class_weights = class_weights.tolist()

print("="*80)
print("INICIANDO RANDOMIZED SEARCH PARA CatBoost")
print("="*80)

# Espaço de busca otimizado para CatBoost (valores razoáveis para generalização)
param_dist = {
    'iterations': [200, 500, 1000],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'depth': [4, 6, 8, 10],
    'l2_leaf_reg': [1, 3, 5, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bylevel': [0.6, 0.8, 1.0],
    'random_strength': [0.5, 1, 2],
    'bagging_temperature': [0.1, 0.2, 0.5],
    'min_child_samples': [1, 5, 10],
    'border_count': [32, 64, 128]
}

# Configurar o RandomizedSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base_catboost = CatBoostClassifier(
    class_weights=class_weights,
    random_seed=42,
    verbose=0,
    has_time=False
)
rs = RandomizedSearchCV(
    estimator=base_catboost,
    param_distributions=param_dist,
    n_iter=25,
    scoring='f1_macro',
    cv=cv,
    verbose=2,
    n_jobs=-1,
    random_state=42,
    return_train_score=True,
)

# Rodar busca
rs.fit(X_treino, y_treino)

best = rs.best_estimator_
best_params = rs.best_params_
best_score = rs.best_score_

print("\n" + "="*80)
print("MELHORES PARAMETROS ENCONTRADOS")
print("="*80)
print(json.dumps(best_params, indent=2))
print(f"Best CV score (f1_macro): {best_score:.4f}")

# Avaliar no conjunto de teste
y_pred = best.predict(X_teste)

print("\n" + "="*80)
print("CLASSIFICATION REPORT - TESTE")
print("="*80)
print(classification_report(y_teste, y_pred, target_names=['Impopular', 'Popular']))

cm = confusion_matrix(y_teste, y_pred)
print("="*80)
print("MATRIZ DE CONFUSÃO - TESTE")
print("="*80)
print("\n                 Predito")
print("              Impopular  Popular")
print(f"Real Impopular    {cm[0,0]:5d}     {cm[0,1]:5d}")
print(f"     Popular      {cm[1,0]:5d}     {cm[1,1]:5d}")
print("\n")

print("Busca concluída.")
