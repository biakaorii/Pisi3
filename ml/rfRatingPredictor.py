from  sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    balanced_accuracy_score,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    average_precision_score,
)
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

#Encontrar o caminho do dataset
caminho_atual = os.path.dirname(os.path.abspath(__file__))
caminho_dataset = os.path.join(caminho_atual, '..', 'dataset', 'dados.parquet')

#Carregar o dataset
df = pd.read_parquet(caminho_dataset)

#Filtragem de no minimo 25 avaliacoes para predicao
df = df[df['avaliacao'] >= 25].copy()

#Criar a coluna de popularidade, 1 para Popular e 0 para Impopular
df['popularidade'] = np.where(df['rating'] >= 4.0, 1, 0)

#Criando uma nova feature derivada de abandono
df['taxa_abandono'] = df['abandonos'] / df['leram']
df['taxa_avaliacao'] = df['avaliacao'] / df['leram']
# Tratar divisões por zero/valores inválidos apenas para a nova feature
df['taxa_avaliacao'] = df['taxa_avaliacao'].replace([np.inf, -np.inf], np.nan).fillna(0)

#Features e variavel alvo
features = ['ano', 'paginas', 'leram', 'querem_ler', 'autor', 'taxa_abandono', 'taxa_avaliacao', 'relendo', 'avaliacao', 'resenha']
X = df[features]
y = df['popularidade']

#One-hot encoding para a coluna 'autor'
X = pd.get_dummies(X, columns=['autor'], drop_first=True)

# Dividir os dados em conjunto de treino e teste balanceando as classes
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

# Treinamento com calibração de probabilidade (Platt/sigmoid)
base_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
modelo = CalibratedClassifierCV(base_estimator=base_model, method='sigmoid', cv=5)
modelo.fit(X_treino, y_treino)

#Predicoes e metricas de avaliacao
y_pred = modelo.predict(X_teste)
y_proba = modelo.predict_proba(X_teste)[:, 1]

print("Classification Report:")
print(classification_report(y_teste, y_pred, digits=4))

cm = confusion_matrix(y_teste, y_pred)
print("Confusion Matrix:\n", cm)

bal_acc = balanced_accuracy_score(y_teste, y_pred)
print(f"Balanced Accuracy: {bal_acc:.4f}")

auc = roc_auc_score(y_teste, y_proba)
ap = average_precision_score(y_teste, y_proba)
print(f"ROC-AUC: {auc:.4f}")
print(f"Average Precision (PR-AUC): {ap:.4f}")

# Plotar Matriz de Confusão
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusão")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.tight_layout()
plt.show()

# Curva ROC
RocCurveDisplay.from_estimator(modelo, X_teste, y_teste)
plt.title(f"Curva ROC (AUC = {auc:.3f})")
plt.show()

# Curva Precisão-Recall
PrecisionRecallDisplay.from_predictions(y_teste, y_proba)
plt.title(f"Curva Precisão-Recall (AP = {ap:.3f})")
plt.show()

# Curva de Calibração (reliability)
CalibrationDisplay.from_estimator(modelo, X_teste, y_teste, n_bins=10, strategy='uniform')
plt.title("Curva de Calibração (10 bins)")
plt.show()
