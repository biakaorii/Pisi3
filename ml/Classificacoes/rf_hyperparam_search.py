import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

# Dividir treino/teste stratificado
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("="*80)
print("INICIANDO RANDOMIZED SEARCH PARA RandomForest")
print("="*80)

# Espaço de busca (valores razoáveis para generalização)
param_dist = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [6, 8, 12, 20, None],
    'min_samples_leaf': [1, 3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2', 0.3, 0.5, None],
    'bootstrap': [True, False]
}

# Configurar o RandomizedSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base_rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rs = RandomizedSearchCV(
    estimator=base_rf,
    param_distributions=param_dist,
    n_iter=20,
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
