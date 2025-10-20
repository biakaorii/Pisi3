from  sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, balanced_accuracy_score
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

#Criando uma nova feature derivada de abandono
df['taxa_abandono'] = df['abandonos'] / df['leram']

#Features e variavel alvo
features = ['ano', 'paginas', 'leram', 'querem_ler', 'autor', 'taxa_abandono', 'relendo', 'avaliacao', 'resenha']
X = df[features]
y = df['popularidade']

#One-hot encoding para a coluna 'autor'
X = pd.get_dummies(X, columns=['autor'], drop_first=True)

# Dividir os dados em conjunto de treino e teste balanceando as classes
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = 0.2, random_state=42, stratify=y)

#Treinamento do modelo RandomForest
modelo = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
modelo.fit(X_treino, y_treino)

#Predicoes e metricas de avaliacao
y_pred = modelo.predict(X_teste)
y_proba = None
try:
	y_proba = modelo.predict_proba(X_teste)[:, 1]
except Exception:
	pass

print("Classification Report:")
print(classification_report(y_teste, y_pred, digits=4))

cm = confusion_matrix(y_teste, y_pred)
print("Confusion Matrix:\n", cm)

bal_acc = balanced_accuracy_score(y_teste, y_pred)
print(f"Balanced Accuracy: {bal_acc:.4f}")

auc = roc_auc_score(y_teste, y_proba)
print(f"ROC-AUC: {auc:.4f}")
