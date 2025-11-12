from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    balanced_accuracy_score,
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
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

#One-hot encoding para a coluna 'autor'
X = pd.get_dummies(X, columns=['autor', 'editora'], drop_first=True)

# Dividir os dados em conjunto de treino e teste balanceando as classes
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Treinamento com calibração de probabilidade (Platt/sigmoid)
base_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
modelo = CalibratedClassifierCV(estimator=base_model, method='sigmoid', cv=5)
modelo.fit(X_treino, y_treino)

# Predições
y_pred = modelo.predict(X_teste)

# ========== FEATURE IMPORTANCE GERAL ==========
# Treinar modelo sem calibração para obter feature importance
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_treino, y_treino)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_treino.columns,
    'importance': rf_model.feature_importances_
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

# ========== MATRIZ DE CONFUSÃO ==========
cm = confusion_matrix(y_teste, y_pred)

print("="*80)
print("MATRIZ DE CONFUSÃO")
print("="*80)
print("\n                 Predito")
print("              Impopular  Popular")
print(f"Real Impopular    {cm[0,0]:5d}     {cm[0,1]:5d}")
print(f"     Popular      {cm[1,0]:5d}     {cm[1,1]:5d}")
print("\n")

# ========== CLASSIFICATION REPORT ==========
print("="*80)
print("CLASSIFICATION REPORT")
print("="*80)
print(classification_report(y_teste, y_pred, 
                          target_names=['Impopular', 'Popular']))



