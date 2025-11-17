import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from scipy.stats import uniform, loguniform

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

# Dividir treino/teste stratificado
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalização dos dados (FUNDAMENTAL para SVM)
scaler = StandardScaler()
X_treino_scaled = scaler.fit_transform(X_treino)
X_teste_scaled = scaler.transform(X_teste)

print("="*80)
print("INICIANDO RANDOMIZED SEARCH PARA SVM com kernel RBF")
print("="*80)

# Espaço de busca (valores razoáveis para SVM com RBF)
param_dist = {
    'C': loguniform(0.01, 100),           # Regularização (log-uniform entre 0.01 e 100)
    'gamma': ['scale', 'auto'] + list(loguniform(0.001, 10).rvs(10)),  # Parâmetro do kernel RBF
    'kernel': ['rbf'],                     # Fixo em RBF
    'shrinking': [True, False],           # Usar heurística de shrinking
    'tol': [1e-3, 1e-4, 1e-5],           # Tolerância para critério de parada
}

# Configurar o RandomizedSearchCV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
base_svm = SVC(
    class_weight='balanced',
    random_state=42,
    max_iter=3000  # Limite de iterações para evitar treinos muito longos
)

rs = RandomizedSearchCV(
    estimator=base_svm,
    param_distributions=param_dist,
    n_iter=25,  # Número de combinações a testar
    scoring='f1_macro',
    cv=cv,
    verbose=2,
    n_jobs=-1,
    random_state=42,
    return_train_score=True,
)

# Rodar busca
print("\nIniciando busca... (pode demorar alguns minutos)")
rs.fit(X_treino_scaled, y_treino)

best = rs.best_estimator_
best_params = rs.best_params_
best_score = rs.best_score_

print("\n" + "="*80)
print("MELHORES PARAMETROS ENCONTRADOS")
print("="*80)
print(json.dumps(best_params, indent=2, default=str))
print(f"Best CV score (f1_macro): {best_score:.4f}")

# Avaliar no conjunto de teste
y_pred = best.predict(X_teste_scaled)

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
