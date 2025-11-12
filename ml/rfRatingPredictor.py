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

# Importâncias das features (sem plot) — tenta extrair do RF dentro do calibrador
def _get_feature_importances_from_model(m, n_features):
    import numpy as _np
    # Caso 1: o próprio modelo exponha (não comum em CalibratedClassifierCV)
    if hasattr(m, "feature_importances_"):
        return _np.asarray(getattr(m, "feature_importances_"))
    # Caso 2: estimator_/base_estimator_ dentro do calibrador
    for attr in ("estimator_", "base_estimator_"):
        base = getattr(m, attr, None)
        if base is not None and hasattr(base, "feature_importances_"):
            return _np.asarray(base.feature_importances_)
    # Caso 3: média das importâncias dos estimadores calibrados por fold
    imps = []
    for cc in getattr(m, "calibrated_classifiers_", []) or []:
        cand = (
            getattr(cc, "estimator", None)
            or getattr(cc, "base_estimator", None)
            or getattr(cc, "estimator_", None)
            or getattr(cc, "base_estimator_", None)
        )
        if cand is not None and hasattr(cand, "feature_importances_"):
            arr = _np.asarray(getattr(cand, "feature_importances_"))
            if arr.shape[0] == n_features:
                imps.append(arr)
    if imps:
        return _np.mean(_np.vstack(imps), axis=0)
    return None

importances = _get_feature_importances_from_model(modelo, X_treino.shape[1])
if importances is not None and importances.shape[0] == X_treino.shape[1]:
    fi_series = pd.Series(importances, index=X_treino.columns).sort_values(ascending=False)
    print("\nTop 20 Feature Importances (RandomForest):")
    print(fi_series.head(20).to_string())
else:
    print("\n[aviso] Não foi possível obter feature_importances_ diretamente do modelo calibrado.")

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
