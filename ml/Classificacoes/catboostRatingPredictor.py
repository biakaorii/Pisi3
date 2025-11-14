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
import os

#Encontrar o caminho do dataset
caminho_atual = os.path.dirname(os.path.abspath(__file__))
caminho_dataset = os.path.join(caminho_atual, '..', 'dataset', 'dados.parquet')

#Carregar o dataset
df = pd.read_parquet(caminho_dataset)

#Filtragem de no minimo 25 avaliacoes para predicao
df = df[df['avaliacao'] >= 25].copy()

#Criar a coluna de popularidade, 1 para Popular e 0 para Impopular
df['popularidade'] = np.where(df['rating'] >= 4.0, 1, 0)


#Features e variavel alvo
features = ['ano', 'paginas', 'querem_ler', 'autor', "editora"]
X = df[features]
y = df['popularidade']

#One-hot encoding para a coluna 'autor' e 'editora'
X = pd.get_dummies(X, columns=['autor', 'editora'], drop_first=True)

# Limpar nomes das colunas para CatBoost/compatibilidade (remover caracteres especiais)
X.columns = X.columns.str.replace('[', '_', regex=False).str.replace(']', '_', regex=False).str.replace('"', '', regex=False).str.replace(':', '_', regex=False).str.replace(',', '_', regex=False).str.replace('{', '_', regex=False).str.replace('}', '_', regex=False)

# Dividir os dados em conjunto de treino e teste balanceando as classes
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Calcular pesos de classe para passar ao CatBoost
classes = np.unique(y_treino)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_treino)
class_weights = class_weights.tolist()

# ========== TREINAMENTO OTIMIZADO COM EARLY STOPPING ==========
# Parâmetros ajustados para melhor generalização
cb_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'random_strength': 1,
    'bagging_temperature': 0.2,
    'border_count': 32,
    'random_seed': 42,
    'verbose': 100,
    'class_weights': class_weights,
}

# Treino preliminar com early stopping usando o conjunto de teste como validação
cat_model = CatBoostClassifier(**cb_params)
try:
    cat_model.fit(X_treino, y_treino, eval_set=(X_teste, y_teste), use_best_model=True, early_stopping_rounds=50)
except TypeError:
    # fallback se a versão da lib não aceitar alguns argumentos
    cat_model.fit(X_treino, y_treino, eval_set=(X_teste, y_teste), early_stopping_rounds=50)

# Obter iterações ótimas encontradas pelo early stopping
try:
    best_iter = int(cat_model.get_best_iteration())
except Exception:
    try:
        best_iter = int(cat_model.get_best_iteration())
    except Exception:
        # fallback razoável
        best_iter = min(200, cb_params['iterations'])

# Use o modelo com o número de iterações escolhido para o modelo calibrado
base_model = CatBoostClassifier(
    iterations=best_iter,
    learning_rate=cb_params['learning_rate'],
    depth=cb_params['depth'],
    l2_leaf_reg=cb_params['l2_leaf_reg'],
    random_seed=cb_params['random_seed'],
    verbose=False,
    class_weights=class_weights,
)
modelo = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=5)
modelo.fit(X_treino, y_treino)

# Predições
y_pred = modelo.predict(X_teste)
y_pred_treino = modelo.predict(X_treino)

# Feature importance (CatBoost API) obtida do modelo treinado com early stopping
importance_values = cat_model.get_feature_importance()
feature_importance = pd.DataFrame({
    'feature': X_treino.columns,
    'importance': importance_values
}).sort_values('importance', ascending=False)

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

# ========== CLASSIFICATION REPORT - TESTE ==========
print("="*80)
print("CLASSIFICATION REPORT - TESTE")
print("="*80)
print(classification_report(y_teste, y_pred, 
                          target_names=['Impopular', 'Popular']))
