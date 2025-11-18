import os
import json
import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Encontrar o caminho do dataset
caminho_atual = os.path.dirname(os.path.abspath(__file__))
caminho_dataset = os.path.join(caminho_atual, '..', '..', 'dataset', 'dados.parquet')

# Carregar o dataset
df = pd.read_parquet(caminho_dataset)

# Filtragem de no minimo 25 avaliacoes para predicao
df = df[df['avaliacao'] >= 25].copy()

# Criar a coluna de popularidade, 1 para Popular e 0 para Impopular
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

# Features e variavel alvo
features = ['ano', 'paginas', 'querem_ler', 'autor', 'editora', 'GeneroPrimario', 'SubGenero']
X = df[features]
y = df['popularidade']

# One-hot encoding para as colunas categóricas
X = pd.get_dummies(X, columns=['autor', 'editora', 'GeneroPrimario', 'SubGenero'], 
                   drop_first=True, 
                   prefix=['autor', 'editora', 'genero_primario', 'subgenero'])

# Sanitizar nomes das colunas (remover caracteres especiais para LightGBM)
X.columns = X.columns.str.replace('[^a-zA-Z0-9_]', '_', regex=True)

# Remover colunas duplicadas que podem ter sido criadas pela limpeza de nomes
X = X.loc[:, ~X.columns.duplicated()]

# Dividir treino/teste stratificado
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Calcular scale_pos_weight para balanceamento
scale_pos_weight = (y_treino == 0).sum() / (y_treino == 1).sum()

print("="*80)
print("INICIANDO RANDOMIZED SEARCH PARA LightGBM")
print("="*80)

# Espaço de busca (valores razoáveis para generalização)
param_dist = {
    'n_estimators': [100, 300, 500],  # Reduzido de 4 para 3 valores
    'max_depth': [6, 10, 12],  # Reduzido de 6 para 3 valores
    'learning_rate': [0.05, 0.1, 0.2],  # Reduzido de 5 para 3 valores
    'num_leaves': [50, 100, 150],  # Reduzido de 5 para 3 valores
    'min_child_samples': [10, 20],  # Reduzido de 4 para 2 valores
    'subsample': [0.8, 1.0],  # Reduzido de 3 para 2 valores
    'colsample_bytree': [0.8, 1.0],  # Reduzido de 3 para 2 valores
    'reg_alpha': [0.1, 0.5],  # Reduzido de 4 para 2 valores
    'reg_lambda': [1, 1.5]  # Reduzido de 4 para 2 valores
}

# Configurar o RandomizedSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base_lgb = LGBMClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbosity=-1,
    n_jobs=-1
)
rs = RandomizedSearchCV(
    estimator=base_lgb,
    param_distributions=param_dist,
    n_iter=15,  # Reduzido de 30 para 15 iterações
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
