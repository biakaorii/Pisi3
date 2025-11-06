import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# 📁 Configuração de caminhos (seguindo padrão do popularityCluster.py)
caminho_atual = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(caminho_atual, '..', 'dataset', 'dados.parquet')
MODELS_PATH = os.path.join(caminho_atual, '..', 'models')
OUTPUTS_PATH = os.path.join(caminho_atual, '..', 'outputs', 'ml_abandono')

# Criar diretórios se não existirem
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(OUTPUTS_PATH, exist_ok=True)

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class AbandonoLivrosClassifier:
    """
    🤖 Classificador de probabilidade de abandono de livros
    
    Prediz se um livro tende a ter alto ou baixo abandono baseado em:
    - Taxa de abandono (abandonos/leram)
    - Número de páginas
    - Rating e avaliações
    - Ano de publicação
    - Gênero
    - Descrição (TF-IDF)
    """
    
    def __init__(self, dataset_path=DATASET_PATH):
        """
        Inicializa o classificador
        
        Args:
            dataset_path: caminho para o arquivo parquet
        """
        self.dataset_path = dataset_path
        self.scaler = StandardScaler()
        self.tfidf = TfidfVectorizer(max_features=50, stop_words='english', min_df=2)
        self.label_encoders = {}
        self.modelo = None
        self.feature_names = None  # Armazena nomes das features de treino
        
        # Verifica se arquivo existe
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"❌ Arquivo não encontrado: {self.dataset_path}\n"
                f"   📁 Certifique-se de que o arquivo dados.parquet existe em dataset/"
            )
    
    def carregar_dados(self):
        """Carrega e prepara os dados iniciais"""
        print(f"📂 Carregando dados de: {self.dataset_path}")
        df = pd.read_parquet(self.dataset_path)
        
        # Mostra informações básicas
        print(f"\n📊 Dataset Info:")
        print(f"   • Total de livros: {len(df):,}")
        print(f"   • Colunas: {', '.join(df.columns.tolist())}")
        
        # Cria taxa de abandono se não existir
        if 'taxa_abandono' not in df.columns and 'abandonos' in df.columns and 'leram' in df.columns:
            df['taxa_abandono'] = df['abandonos'] / df['leram']
            df['taxa_abandono'] = df['taxa_abandono'].replace([np.inf, -np.inf], np.nan).fillna(0)
            print(f"   ✅ Taxa de abandono calculada")
        
        # Remove valores nulos das colunas essenciais
        colunas_necessarias = ['paginas', 'rating', 'ano']
        if 'genero' in df.columns:
            colunas_necessarias.append('genero')
        
        antes = len(df)
        df = df.dropna(subset=colunas_necessarias)
        
        # Preenche descrição vazia
        if 'descricao' in df.columns:
            df['descricao'] = df['descricao'].fillna('')
        else:
            df['descricao'] = ''
        
        if len(df) < antes:
            print(f"   ⚠️  Removidos {antes - len(df):,} registros com valores nulos")
        
        return df
    
    def criar_target(self, df, threshold='median', usar_coluna_abandonos=True):
        """
        Cria a variável target (alto/baixo abandono)
        
        Args:
            df: DataFrame com os dados
            threshold: 'median', 'mean' ou valor numérico
            usar_coluna_abandonos: usar coluna abandonos existente ou taxa_abandono
        """
        print("\n🎯 Criando variável target...")
        
        if usar_coluna_abandonos and 'taxa_abandono' in df.columns:
            # Usa taxa de abandono real
            coluna_ref = 'taxa_abandono'
            print(f"   • Usando coluna: {coluna_ref}")
        elif 'abandonos' in df.columns:
            coluna_ref = 'abandonos'
            print(f"   • Usando coluna: {coluna_ref}")
        else:
            # Simula abandonos baseado em heurísticas
            print("   ⚠️  Simulando abandonos baseado em features...")
            df['abandonos_simulado'] = (
                (df['paginas'] > 400).astype(int) * 0.3 +
                (df['rating'] < 3.5).astype(int) * 0.4 +
                np.random.random(len(df)) * 0.3
            )
            coluna_ref = 'abandonos_simulado'
        
        # Define threshold
        if threshold == 'median':
            limite = df[coluna_ref].median()
        elif threshold == 'mean':
            limite = df[coluna_ref].mean()
        else:
            limite = threshold
        
        # Cria target binário
        df['alto_abandono'] = (df[coluna_ref] > limite).astype(int)
        
        # Mostra distribuição
        dist = df['alto_abandono'].value_counts()
        print(f"\n   📊 Distribuição da Target:")
        print(f"      • Baixo Abandono (0): {dist.get(0, 0):,} ({dist.get(0, 0)/len(df)*100:.1f}%)")
        print(f"      • Alto Abandono (1):  {dist.get(1, 0):,} ({dist.get(1, 0)/len(df)*100:.1f}%)")
        print(f"      • Limiar: {limite:.4f}")
        
        return df
    
    def preparar_features(self, df, treino=True):
        """
        Prepara as features para o modelo
        
        Args:
            df: DataFrame com os dados
            treino: Se True, fit nos encoders. Se False, apenas transform
        """
        df_features = df.copy()
        
        # 1. Features numéricas básicas sempre disponíveis
        features_numericas = ['paginas', 'rating', 'ano']
        
        # 2. Features adicionais se disponíveis
        colunas_opcionais = ['avaliacao', 'lendo', 'leram', 'abandonos', 'taxa_abandono']
        for col in colunas_opcionais:
            if col in df.columns:
                features_numericas.append(col)
            elif not treino and col in self.feature_names:
                # Se a feature foi usada no treino mas não está disponível, preenche com 0
                df_features[col] = 0
                features_numericas.append(col)
        
        # 3. Encoding de gênero (se disponível)
        if 'genero' in df.columns:
            if treino:
                self.label_encoders['genero'] = LabelEncoder()
                df_features['genero_encoded'] = self.label_encoders['genero'].fit_transform(
                    df_features['genero'].fillna('Unknown')
                )
            else:
                # Handle unknown categories
                genero_values = df_features['genero'].fillna('Unknown')
                known_categories = set(self.label_encoders['genero'].classes_)
                genero_values = genero_values.apply(lambda x: x if x in known_categories else 'Unknown')
                df_features['genero_encoded'] = self.label_encoders['genero'].transform(genero_values)
            
            features_numericas.append('genero_encoded')
        elif not treino and 'genero_encoded' in self.feature_names:
            df_features['genero_encoded'] = 0
            features_numericas.append('genero_encoded')
        
        # 4. Features engenheiradas
        df_features['paginas_rating'] = df_features['paginas'] * df_features['rating']
        df_features['anos_desde_publicacao'] = 2025 - df_features['ano']
        df_features['livro_longo'] = (df_features['paginas'] > 400).astype(int)
        df_features['rating_baixo'] = (df_features['rating'] < 3.5).astype(int)
        df_features['rating_alto'] = (df_features['rating'] >= 4.0).astype(int)
        df_features['livro_recente'] = (df_features['ano'] >= 2015).astype(int)
        
        features_engenheiradas = [
            'paginas_rating', 'anos_desde_publicacao', 
            'livro_longo', 'rating_baixo', 'rating_alto', 'livro_recente'
        ]
        
        # 5. TF-IDF da descrição (se disponível)
        if 'descricao' in df_features.columns and df_features['descricao'].notna().sum() > 10:
            try:
                if treino:
                    tfidf_matrix = self.tfidf.fit_transform(df_features['descricao'].fillna(''))
                else:
                    tfidf_matrix = self.tfidf.transform(df_features['descricao'].fillna(''))
                
                tfidf_df = pd.DataFrame(
                    tfidf_matrix.toarray(),
                    columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
                )
            except Exception as e:
                print(f"   ⚠️  Erro ao processar TF-IDF: {e}")
                tfidf_df = pd.DataFrame()
        else:
            # Se não tem descrição, cria colunas vazias com base no treino
            if not treino and self.feature_names is not None:
                tfidf_cols = [col for col in self.feature_names if col.startswith('tfidf_')]
                tfidf_df = pd.DataFrame(0, index=df_features.index, columns=tfidf_cols)
            else:
                tfidf_df = pd.DataFrame()
        
        # 6. Combina todas as features
        features_finais = features_numericas + features_engenheiradas
        
        X = df_features[features_finais].copy()
        
        if not tfidf_df.empty:
            X = pd.concat([
                X.reset_index(drop=True),
                tfidf_df.reset_index(drop=True)
            ], axis=1)
        
        # 7. Remove features com NaN
        X = X.fillna(0)
        
        # 8. Se estiver em modo de predição, garante que tem todas as features do treino
        if not treino and self.feature_names is not None:
            # Adiciona colunas faltantes com zeros
            for col in self.feature_names:
                if col not in X.columns:
                    X[col] = 0
            
            # Garante a mesma ordem das colunas do treino
            X = X[self.feature_names]
        
        # 9. Normalização
        if treino:
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = X.columns.tolist()  # Salva os nomes das features
        else:
            X_scaled = self.scaler.transform(X)
        
        if treino:
            print(f"   ✅ {X.shape[1]} features preparadas")
        
        return pd.DataFrame(X_scaled, columns=X.columns)
    
    def treinar_modelos(self, X_train, y_train, X_test, y_test):
        """Treina e compara diferentes modelos"""
        modelos = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, 
                random_state=42, 
                n_jobs=-1,
                max_depth=10
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100, 
                random_state=42,
                max_depth=5,
                learning_rate=0.1
            )
        }
        
        resultados = {}
        
        print("\n" + "="*70)
        print("🤖 TREINANDO E AVALIANDO MODELOS")
        print("="*70)
        
        for nome, modelo in modelos.items():
            print(f"\n📈 {nome}...")
            
            # Treina
            modelo.fit(X_train, y_train)
            
            # Predições
            y_pred = modelo.predict(X_test)
            y_proba = modelo.predict_proba(X_test)[:, 1]
            
            # Métricas
            roc_auc = roc_auc_score(y_test, y_proba)
            
            # Cross-validation
            print("   🔄 Executando cross-validation...")
            cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
            
            resultados[nome] = {
                'modelo': modelo,
                'roc_auc': roc_auc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_proba': y_proba
            }
            
            print(f"   • ROC-AUC Test:  {roc_auc:.4f}")
            print(f"   • CV Score:      {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # Seleciona melhor modelo
        melhor_nome = max(resultados, key=lambda x: resultados[x]['roc_auc'])
        self.modelo = resultados[melhor_nome]['modelo']
        
        print(f"\n✅ Melhor modelo selecionado: {melhor_nome}")
        print("="*70)
        
        return resultados, melhor_nome
    
    def avaliar_modelo(self, X_test, y_test, y_pred, y_proba):
        """Avalia o modelo com métricas detalhadas"""
        print("\n" + "="*70)
        print("📊 RELATÓRIO DE CLASSIFICAÇÃO")
        print("="*70 + "\n")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Baixo Abandono', 'Alto Abandono']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Baixo', 'Alto'],
                   yticklabels=['Baixo', 'Alto'])
        plt.title('Matriz de Confusão - Predição de Abandono', fontsize=14, fontweight='bold')
        plt.ylabel('Classe Real')
        plt.xlabel('Classe Predita')
        plt.tight_layout()
        
        cm_path = os.path.join(OUTPUTS_PATH, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"💾 Matriz de confusão: {cm_path}")
        plt.close()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falsos Positivos (FPR)')
        plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
        plt.title('Curva ROC - Predição de Abandono', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        roc_path = os.path.join(OUTPUTS_PATH, 'roc_curve.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        print(f"💾 Curva ROC: {roc_path}")
        plt.close()
    
    def analisar_importancia_features(self, X_train):
        """Analisa importância das features (para modelos tree-based)"""
        if hasattr(self.modelo, 'feature_importances_'):
            importancias = pd.DataFrame({
                'feature': X_train.columns,
                'importance': self.modelo.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            print("\n" + "="*70)
            print("🎯 TOP 20 FEATURES MAIS IMPORTANTES")
            print("="*70 + "\n")
            
            # Formata nomes das features
            importancias['feature_nome'] = importancias['feature'].apply(
                lambda x: x.replace('_', ' ').title() if not x.startswith('tfidf') 
                else f"Palavra-chave {x.split('_')[1]}"
            )
            
            print(importancias[['feature_nome', 'importance']].to_string(index=False))
            
            # Visualização
            plt.figure(figsize=(10, 8))
            top15 = importancias.head(15)
            sns.barplot(data=top15, x='importance', y='feature_nome', palette='viridis')
            plt.title('Top 15 Features Mais Importantes', fontsize=14, fontweight='bold')
            plt.xlabel('Importância Relativa')
            plt.ylabel('Feature')
            plt.tight_layout()
            
            fi_path = os.path.join(OUTPUTS_PATH, 'feature_importance.png')
            plt.savefig(fi_path, dpi=300, bbox_inches='tight')
            print(f"\n💾 Feature importance: {fi_path}")
            plt.close()
        else:
            print("\n⚠️  Modelo não suporta feature_importances_")
    
    def gerar_insights(self, df):
        """Gera insights sobre padrões de abandono"""
        print("\n" + "="*70)
        print("💡 INSIGHTS DE ABANDONO")
        print("="*70 + "\n")
        
        # Por número de páginas
        df['faixa_paginas'] = pd.cut(
            df['paginas'], 
            bins=[0, 200, 400, 600, 1000], 
            labels=['Curto (<200)', 'Médio (200-400)', 'Longo (400-600)', 'Muito Longo (>600)']
        )
        abandono_por_paginas = df.groupby('faixa_paginas')['alto_abandono'].mean()
        print("📖 Taxa de Abandono por Tamanho:")
        for idx, val in abandono_por_paginas.items():
            print(f"   • {idx:20s}: {val:.1%}")
        
        # Por gênero (se disponível)
        if 'genero' in df.columns:
            abandono_por_genero = df.groupby('genero')['alto_abandono'].mean().sort_values(ascending=False).head(5)
            print(f"\n🎭 Top 5 Gêneros com Maior Abandono:")
            for idx, val in abandono_por_genero.items():
                print(f"   • {idx:30s}: {val:.1%}")
        
        # Por rating
        df['faixa_rating'] = pd.cut(
            df['rating'], 
            bins=[0, 3, 3.5, 4, 5], 
            labels=['Baixo (<3)', 'Médio (3-3.5)', 'Bom (3.5-4)', 'Excelente (>4)']
        )
        abandono_por_rating = df.groupby('faixa_rating')['alto_abandono'].mean()
        print(f"\n⭐ Taxa de Abandono por Rating:")
        for idx, val in abandono_por_rating.items():
            print(f"   • {idx:20s}: {val:.1%}")
        
        # Estatísticas gerais
        print(f"\n📊 Estatísticas Comparativas:")
        print(f"   • Páginas médias (Alto Abandono):   {df[df['alto_abandono']==1]['paginas'].mean():.0f}")
        print(f"   • Páginas médias (Baixo Abandono):  {df[df['alto_abandono']==0]['paginas'].mean():.0f}")
        print(f"   • Rating médio (Alto Abandono):     {df[df['alto_abandono']==1]['rating'].mean():.2f}")
        print(f"   • Rating médio (Baixo Abandono):    {df[df['alto_abandono']==0]['rating'].mean():.2f}")
        
        # Taxa de abandono real (se disponível)
        if 'taxa_abandono' in df.columns:
            print(f"   • Taxa real (Alto Abandono):        {df[df['alto_abandono']==1]['taxa_abandono'].mean():.2%}")
            print(f"   • Taxa real (Baixo Abandono):       {df[df['alto_abandono']==0]['taxa_abandono'].mean():.2%}")
    
    def salvar_modelo(self, nome='modelo_abandono.pkl'):
        """Salva o modelo treinado"""
        model_path = os.path.join(MODELS_PATH, nome)
        joblib.dump({
            'modelo': self.modelo,
            'scaler': self.scaler,
            'tfidf': self.tfidf,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }, model_path)
        print(f"\n💾 Modelo salvo: {model_path}")
    
    def carregar_modelo(self, nome='modelo_abandono.pkl'):
        """Carrega um modelo previamente treinado"""
        model_path = os.path.join(MODELS_PATH, nome)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
        
        componentes = joblib.load(model_path)
        self.modelo = componentes['modelo']
        self.scaler = componentes['scaler']
        self.tfidf = componentes['tfidf']
        self.label_encoders = componentes['label_encoders']
        self.feature_names = componentes.get('feature_names', None)
        
        print(f"✅ Modelo carregado: {model_path}")
    
    def prever_abandono(self, livro_data):
        """
        Prevê probabilidade de abandono para um novo livro
        
        Args:
            livro_data: dict com features do livro
        
        Returns:
            dict com probabilidade e classificação
        """
        if self.modelo is None:
            raise ValueError("Modelo não treinado. Execute treinar_modelos() primeiro.")
        
        df_novo = pd.DataFrame([livro_data])
        X_novo = self.preparar_features(df_novo, treino=False)
        
        probabilidade = self.modelo.predict_proba(X_novo)[0, 1]
        predicao = self.modelo.predict(X_novo)[0]
        
        return {
            'probabilidade_abandono': probabilidade,
            'classificacao': 'Alto Abandono' if predicao == 1 else 'Baixo Abandono',
            'confianca': max(self.modelo.predict_proba(X_novo)[0])
        }


def main():
    """🚀 Função principal para executar o pipeline completo"""
    
    print("="*70)
    print("📚 SISTEMA DE PREDIÇÃO DE ABANDONO DE LIVROS")
    print("="*70)
    
    try:
        # Inicializa o classificador
        clf = AbandonoLivrosClassifier()
        
        # Carrega dados
        df = clf.carregar_dados()
        
        # Cria target
        df = clf.criar_target(df, threshold='median', usar_coluna_abandonos=True)
        
        # Prepara features
        print("\n🔧 Preparando features...")
        X = clf.preparar_features(df, treino=True)
        y = df['alto_abandono']
        
        # Split treino/teste
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   • Conjunto de treino: {len(X_train):,} livros")
        print(f"   • Conjunto de teste:  {len(X_test):,} livros")
        
        # Treina modelos
        resultados, melhor_modelo = clf.treinar_modelos(X_train, y_train, X_test, y_test)
        
        # Avalia melhor modelo
        clf.avaliar_modelo(
            X_test, y_test, 
            resultados[melhor_modelo]['y_pred'],
            resultados[melhor_modelo]['y_proba']
        )
        
        # Importância das features
        clf.analisar_importancia_features(X_train)
        
        # Gera insights
        clf.gerar_insights(df)
        
        # Salva modelo
        clf.salvar_modelo()
        
        # Exemplo de predição (usando apenas features básicas)
        print("\n" + "="*70)
        print("🔮 EXEMPLO DE PREDIÇÃO PARA NOVO LIVRO")
        print("="*70 + "\n")
        
        # Pega um exemplo real do dataset para teste
        livro_exemplo = df.sample(1).iloc[0]
        
        # Cria dict apenas com features básicas que sempre estarão disponíveis
        livro_teste = {
            'paginas': int(livro_exemplo['paginas']),
            'rating': float(livro_exemplo['rating']),
            'ano': int(livro_exemplo['ano']),
        }
        
        # Adiciona features opcionais se disponíveis
        if 'genero' in df.columns:
            livro_teste['genero'] = livro_exemplo['genero']
        if 'descricao' in df.columns:
            livro_teste['descricao'] = livro_exemplo['descricao']
        
        resultado = clf.prever_abandono(livro_teste)
        
        print(f"📘 Livro de Teste:")
        for key, val in livro_teste.items():
            if key != 'descricao':
                print(f"   • {key.title():15s}: {val}")
        
        print(f"\n🎯 Resultado da Predição:")
        print(f"   • Probabilidade de Abandono: {resultado['probabilidade_abandono']:.1%}")
        print(f"   • Classificação:             {resultado['classificacao']}")
        print(f"   • Confiança do Modelo:       {resultado['confianca']:.1%}")
        
        if 'alto_abandono' in livro_exemplo:
            real = 'Alto Abandono' if livro_exemplo['alto_abandono'] == 1 else 'Baixo Abandono'
            print(f"   • Classe Real:               {real}")
            acertou = resultado['classificacao'] == real
            print(f"   • Status:                    {'✅ Acertou' if acertou else '❌ Errou'}")
        
        print("\n" + "="*70)
        print("✅ PROCESSO CONCLUÍDO COM SUCESSO!")
        print("="*70)
        print(f"\n📁 Arquivos gerados:")
        print(f"   • Gráficos: {OUTPUTS_PATH}")
        print(f"   • Modelo:   {MODELS_PATH}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Erro: {e}")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()