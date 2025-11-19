import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from dash import Dash, html, dcc, Input, Output, callback
import dash_bootstrap_components as dbc
import dash
import os
from pathlib import Path
import base64
from io import BytesIO
from PIL import Image


pio.templates.default = "plotly"

import os
caminho_atual = os.path.dirname(os.path.abspath(__file__))
caminho_dataset = os.path.join(caminho_atual, '..', 'dataset', 'dados.parquet')
df = pd.read_parquet(caminho_dataset)


def explode_generos(df, coluna_genero, col_valor):
    df_copy = df.copy()
    df_copy[coluna_genero] = df_copy[coluna_genero].str.split(' / ')
    df_exploded = df_copy.explode(coluna_genero)
    df_exploded[coluna_genero] = df_exploded[coluna_genero].str.strip()
    df_agrupado = df_exploded.groupby(coluna_genero)[col_valor].sum().sort_values(ascending=False)
    return df_agrupado


df_generos_exploded = df.copy()
_sep_pattern = r'\s*[\\/;|·•,]\s*'
df_generos_exploded['genero'] = df_generos_exploded['genero'].astype(str).str.split(_sep_pattern, regex=True)
df_generos_exploded = df_generos_exploded.explode('genero')
df_generos_exploded['genero'] = df_generos_exploded['genero'].astype(str).str.strip()
df_generos_exploded['genero'] = df_generos_exploded['genero'].str.replace(r'(?i)edição em capa dura', '', regex=True)
df_generos_exploded['genero'] = df_generos_exploded['genero'].str.replace(r'(?i)capa dura', '', regex=True)
df_generos_exploded['genero'] = df_generos_exploded['genero'].str.replace(r'\s{2,}', ' ', regex=True).str.strip(' -–—•·,;:').str.strip()


_genero_regex = r'^[A-Za-zÀ-ÖØ-öø-ÿ\s\-]{2,40}$'
_stop_generos = {s.lower() for s in [
    'EDIÇÃO EM CAPA DURA','CAPA DURA','Capa Dura','Edição em Capa Dura','Edição em capa dura',
    'Será que','Sera que'
]}
df_generos_exploded = df_generos_exploded[
    df_generos_exploded['genero'].notna() &
    (df_generos_exploded['genero'] != '') &
    df_generos_exploded['genero'].str.match(_genero_regex, na=False) &
    (~df_generos_exploded['genero'].str.lower().isin(_stop_generos))
]


_counts_genero = df_generos_exploded['genero'].value_counts()
_valid_generos = _counts_genero[_counts_genero >= 3].index
df_generos_exploded = df_generos_exploded[df_generos_exploded['genero'].isin(_valid_generos)]



app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = "Dashboard de Livros"

# Adicionando estilos CSS customizados para fundo acinzentado claro
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background: #f5f5f5 !important;
                background-attachment: fixed;
                margin: 0;
                padding: 0;
                min-height: 100vh;
            }
            .container-fluid {
                background: transparent !important;
            }
            .text-primary {
                color: #2d5016 !important;
            }
            /* Header com cor preta */
            h1 {
                color: #000000 !important;
                font-weight: 700;
            }
            /* Barra de navegação moderna */
            .navigation-bar {
                background: white;
                border-radius: 20px;
                padding: 12px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
                margin: 20px 0;
                position: relative;
                z-index: 100;
            }
            .nav-btn {
                color: #666 !important;
                background: transparent !important;
                border: none !important;
                font-weight: 600;
                border-radius: 12px;
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                margin: 0 4px;
                padding: 12px 20px;
                position: relative;
                overflow: hidden;
                font-size: 0.95rem;
                letter-spacing: 0.3px;
            }
            .nav-btn:hover {
                color: #333 !important;
                background: rgba(0, 0, 0, 0.04) !important;
                transform: translateY(-1px);
            }
            .nav-btn::before {
                content: '';
                position: absolute;
                bottom: 0;
                left: 50%;
                width: 0;
                height: 3px;
                background: #4caf50;
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                transform: translateX(-50%);
                border-radius: 4px;
            }
            .nav-btn-active {
                color: #4caf50 !important;
                background: rgba(76, 175, 80, 0.08) !important;
            }
            .nav-btn-active::before {
                width: 80%;
            }
            /* KPI Cards com cores */
            .card {
                background: linear-gradient(135deg, #ffffff 0%, #f1f8e9 100%) !important;
                border: none !important;
                border-radius: 15px !important;
                border-left: 4px solid transparent !important;
                background-clip: padding-box !important;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08) !important;
                transition: all 0.3s ease !important;
            }
            .card:nth-child(1) { border-left-color: #4caf50 !important; }
            .card:nth-child(2) { border-left-color: #66bb6a !important; }
            .card:nth-child(3) { border-left-color: #81c784 !important; }
            .card:nth-child(4) { border-left-color: #2e7d32 !important; }
            .card:nth-child(5) { border-left-color: #388e3c !important; }
            .card:hover {
                transform: translateY(-5px) !important;
                box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15) !important;
            }
            .card i {
                font-size: 1.2rem;
                opacity: 0.8;
                width: 24px;
                text-align: center;
            }
            .card:hover i {
                opacity: 1;
                transform: scale(1.1);
                transition: all 0.3s ease;
            }
            /* Section dividers com gradiente */
            .section-divider {
                border: none;
                height: 3px;
                background: linear-gradient(90deg, #4caf50 0%, #388e3c 100%);
                margin: 30px 0 20px 0;
                border-radius: 2px;
            }
            /* Gráficos com bordas coloridas */
            .graph-container {
                transition: all 0.4s ease;
                border-radius: 15px;
                padding: 10px;
                margin: 8px;
                background: rgba(255, 255, 255, 0.9);
                border: 2px solid transparent;
                backdrop-filter: blur(10px);
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
                overflow: hidden;
            }
            .graph-container:hover {
                transform: scale(1.005);
                box-shadow: 0 15px 40px rgba(76, 175, 80, 0.2);
                background: rgba(255, 255, 255, 0.95);
                z-index: 20;
                position: relative;
            }
            .graph-container::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                border-radius: 15px;
                padding: 2px;
                background: linear-gradient(45deg, #4caf50, #388e3c);
                -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
                mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
                -webkit-mask-composite: xor;
                mask-composite: exclude;
                opacity: 0;
                transition: opacity 0.4s ease;
            }
            .graph-container:hover::before {
                opacity: 1;
            }
            .graph-row:hover .graph-container:not(:hover) {
                transform: scale(0.98);
                opacity: 0.8;
            }
            .plotly-graph-div {
                border-radius: 10px;
            }
            /* Labels dos filtros com cor */
            label {
                color: #2e7d32 !important;
                font-weight: 500 !important;
                margin-bottom: 8px !important;
            }
            /* Dropdowns com estilo */
            .Select-control {
                border-radius: 8px !important;
                border: 1px solid #c8e6c9 !important;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
            }
            /* Footer com gradiente sutil */
            .text-muted {
                background: linear-gradient(45deg, #66bb6a, #4caf50);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
            }
            /* RangeSlider personalizado verde */
            .rc-slider {
                background-color: #e8f5e8 !important;
            }
            .rc-slider-track {
                background-color: #4caf50 !important;
            }
            .rc-slider-handle {
                border-color: #2e7d32 !important;
                background-color: #4caf50 !important;
            }
            .rc-slider-handle:hover {
                border-color: #1b5e20 !important;
            }
            .rc-slider-handle:focus {
                border-color: #1b5e20 !important;
                box-shadow: 0 0 0 5px rgba(76, 175, 80, 0.2) !important;
            }
            .rc-slider-dot-active {
                border-color: #4caf50 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''



def calcular_kpis():
    """Calcula os KPIs principais"""
    return {
        'total_livros': len(df),
        'total_autores': df['autor'].nunique(),
        'total_editoras': df['editora'].nunique(),
        'media_rating': df['rating'].mean(),
        'total_resenhas': df['resenha'].sum(),
        'livro_mais_lido': df.loc[df['leram'].idxmax(), 'titulo'],
        'livro_mais_abandonado': df.loc[df['abandonos'].idxmax(), 'titulo'],
    }

def criar_card_kpi(titulo, valor, icone=""):
    """Cria um card de KPI"""
    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.I(className=f"fa {icone} me-2"),
                html.Span(titulo)
            ], className="text-muted d-flex align-items-center"),
            html.H3(valor, className="text-primary fw-bold mt-2")
        ]),
        className="shadow-sm mb-3"
    )



app.layout = dbc.Container([

    # Store para a seção ativa
    dcc.Store(id='active-section', data='popularidade'),
    
    # Font Awesome CSS
    html.Link(
        rel="stylesheet",
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"
    ),

    dbc.Row([
        dbc.Col([
            html.H1([
                html.I(className="fas fa-book me-2"),
                "Dashboard de Análise de Livros"
            ], className="text-center mt-4 mb-4"),
            html.Hr()
        ])
    ]),
    
    # Barra de navegação das seções
    dbc.Row([
        dbc.Col([
            dbc.Nav([
                dbc.NavItem(dbc.Button("Popularidade", id="nav-popularidade", className="nav-btn nav-btn-active", n_clicks=0)),
                dbc.NavItem(dbc.Button("Avaliação", id="nav-avaliacao", className="nav-btn", n_clicks=0)),
                dbc.NavItem(dbc.Button("Gêneros", id="nav-generos", className="nav-btn", n_clicks=0)),
                dbc.NavItem(dbc.Button("Idiomas", id="nav-idiomas", className="nav-btn", n_clicks=0)),
                dbc.NavItem(dbc.Button("Leitura", id="nav-leitura", className="nav-btn", n_clicks=0)),
                dbc.NavItem(dbc.Button("Público", id="nav-publico", className="nav-btn", n_clicks=0)),
                dbc.NavItem(dbc.Button("SHAP Analysis", id="nav-shap", className="nav-btn", n_clicks=0)),
            ], pills=True, justified=True, className="mb-4 navigation-bar")
        ])
    ]),
    

    # KPIs - sempre visíveis
    dbc.Row([
        dbc.Col(html.Div(id='kpis-container'), width=12)
    ], className="mb-4"),
    
    # Filtros - sempre visíveis
    dbc.Row([
        dbc.Col([
            html.Label("Filtrar por Gênero:"),
            dcc.Dropdown(
                id='filtro-genero',
                options=[{'label': 'Todos', 'value': 'todos'}] + 
                        [{'label': g, 'value': g} for g in sorted(df_generos_exploded['genero'].unique())],
                value='todos',
                clearable=False
            )
        ], md=3),
        dbc.Col([
            html.Label("Filtrar por Idioma:"),
            dcc.Dropdown(
                id='filtro-idioma',
                options=[{'label': 'Todos', 'value': 'todos'}] + 
                        [{'label': i, 'value': i} for i in sorted(df['idioma'].dropna().unique())],
                value='todos',
                clearable=False
            )
        ], md=3),
        dbc.Col([
            html.Label("Filtrar por Editora:"),
            dcc.Dropdown(
                id='filtro-editora',
                options=[{'label': 'Todos', 'value': 'todos'}] + 
                        [{'label': e, 'value': e} for e in sorted(df['editora'].dropna().unique())],
                value='todos',
                clearable=False
            )
        ], md=3),
        dbc.Col([
            html.Label("Filtrar por Ano:"),
            dcc.RangeSlider(
                id='filtro-ano',
                min=int(df['ano'].min()),
                max=int(df['ano'].max()),
                value=[int(df['ano'].min()), int(df['ano'].max())],
                marks={int(df['ano'].min()): str(int(df['ano'].min())), 
                       int(df['ano'].max()): str(int(df['ano'].max()))},
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], md=3)
    ], className="mb-4"),
    
    # Container para conteúdo dinâmico das seções
    html.Div(id='dynamic-content'),
    

    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("Dashboard desenvolvido para análise de dataset de livros", 
                   className="text-center text-muted mb-4")
        ])
    ])
    
], fluid=True)


# Callback para controlar a seção ativa
@callback(
    [Output('active-section', 'data'),
     Output('nav-popularidade', 'className'),
     Output('nav-avaliacao', 'className'),
     Output('nav-generos', 'className'),
     Output('nav-idiomas', 'className'),
     Output('nav-leitura', 'className'),
     Output('nav-publico', 'className'),
     Output('nav-shap', 'className')],
    [Input('nav-popularidade', 'n_clicks'),
     Input('nav-avaliacao', 'n_clicks'),
     Input('nav-generos', 'n_clicks'),
     Input('nav-idiomas', 'n_clicks'),
     Input('nav-leitura', 'n_clicks'),
     Input('nav-publico', 'n_clicks'),
     Input('nav-shap', 'n_clicks')],
    prevent_initial_call=False
)
def update_active_section(pop_clicks, av_clicks, gen_clicks, idi_clicks, leit_clicks, pub_clicks, shap_clicks):
    ctx = dash.callback_context
    if not ctx.triggered:
        return 'popularidade', 'nav-btn nav-btn-active', 'nav-btn', 'nav-btn', 'nav-btn', 'nav-btn', 'nav-btn', 'nav-btn'
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    classes = ['nav-btn'] * 7
    section_map = {
        'nav-popularidade': (0, 'popularidade'),
        'nav-avaliacao': (1, 'avaliacao'),
        'nav-generos': (2, 'generos'),
        'nav-idiomas': (3, 'idiomas'),
        'nav-leitura': (4, 'leitura'),
        'nav-publico': (5, 'publico'),
        'nav-shap': (6, 'shap')
    }
    
    if button_id in section_map:
        idx, section = section_map[button_id]
        classes[idx] = 'nav-btn nav-btn-active'
        return section, *classes
    
    return 'popularidade', 'nav-btn nav-btn-active', 'nav-btn', 'nav-btn', 'nav-btn', 'nav-btn', 'nav-btn', 'nav-btn'


# Callback para atualizar conteúdo dinâmico
@callback(
    Output('dynamic-content', 'children'),
    Input('active-section', 'data')
)
def update_dynamic_content(active_section):
    if active_section == 'popularidade':
        return html.Div([
            html.Hr(className="section-divider"),
            dbc.Row([
                dbc.Col([
                    html.H4("Análise de Popularidade", className="mt-3 mb-3"),
                ])
            ]),
            dbc.Row([
                dbc.Col([html.Div([dcc.Graph(id='top-livros-lidos')], className="graph-container")], md=4),
                dbc.Col([html.Div([dcc.Graph(id='top-autores')], className="graph-container")], md=4),
                dbc.Col([html.Div([dcc.Graph(id='top-editoras')], className="graph-container")], md=4),
            ], className="mb-4 graph-row"),
            dbc.Row([
                dbc.Col([html.Div([dcc.Graph(id='livros-por-ano')], className="graph-container")], md=12),
            ], className="mb-4 graph-row")
        ])
    
    elif active_section == 'avaliacao':
        return html.Div([
            html.Hr(className="section-divider"),
            dbc.Row([
                dbc.Col([
                    html.H4("Análise de Avaliação", className="mt-3 mb-3"),
                ])
            ]),
            dbc.Row([
                dbc.Col([html.Div([dcc.Graph(id='dist-rating')], className="graph-container")], md=6),
                dbc.Col([html.Div([dcc.Graph(id='scatter-rating-avaliacoes')], className="graph-container")], md=6),
            ], className="mb-4 graph-row"),
            dbc.Row([
                dbc.Col([html.Div([dcc.Graph(id='top-melhores-livros')], className="graph-container")], md=6),
                dbc.Col([html.Div([dcc.Graph(id='top-piores-livros')], className="graph-container")], md=6),
            ], className="mb-4 graph-row")
        ])
    
    elif active_section == 'generos':
        return html.Div([
            html.Hr(className="section-divider"),
            dbc.Row([
                dbc.Col([
                    html.H4("Análise por Gênero Literário", className="mt-3 mb-3"),
                ])
            ]),
            dbc.Row([
                dbc.Col([html.Div([dcc.Graph(id='dist-generos')], className="graph-container")], md=6),
                dbc.Col([html.Div([dcc.Graph(id='rating-por-genero')], className="graph-container")], md=6),
            ], className="mb-4 graph-row"),
            dbc.Row([
                dbc.Col([html.Div([dcc.Graph(id='generos-mais-lidos')], className="graph-container")], md=12),
            ], className="mb-4 graph-row")
        ])
    
    elif active_section == 'idiomas':
        return html.Div([
            html.Hr(className="section-divider"),
            dbc.Row([
                dbc.Col([
                    html.H4("Análise por Idioma", className="mt-3 mb-3"),
                ])
            ]),
            dbc.Row([
                dbc.Col([html.Div([dcc.Graph(id='dist-idiomas')], className="graph-container")], md=6),
                dbc.Col([html.Div([dcc.Graph(id='rating-por-idioma')], className="graph-container")], md=6),
            ], className="mb-4 graph-row")
        ])
    
    elif active_section == 'leitura':
        return html.Div([
            html.Hr(className="section-divider"),
            dbc.Row([
                dbc.Col([
                    html.H4("Comportamento de Leitura", className="mt-3 mb-3"),
                ])
            ]),
            dbc.Row([
                dbc.Col([html.Div([dcc.Graph(id='metricas-leitura')], className="graph-container")], md=6),
                dbc.Col([html.Div([dcc.Graph(id='taxa-abandono-genero')], className="graph-container")], md=6),
            ], className="mb-4 graph-row")
        ])
    
    elif active_section == 'publico':
        return html.Div([
            html.Hr(className="section-divider"),
            dbc.Row([
                dbc.Col([
                    html.H4("Análise por Gênero do Público", className="mt-3 mb-3"),
                ])
            ]),
            dbc.Row([
                dbc.Col([html.Div([dcc.Graph(id='leitores-genero')], className="graph-container")], md=6),
                dbc.Col([html.Div([dcc.Graph(id='generos-por-publico')], className="graph-container")], md=6),
            ], className="mb-4 graph-row")
        ])
    
    elif active_section == 'shap':
        # Caminhos para os arquivos do SHAP
        caminho_ml = os.path.join(caminho_atual, '..', 'ml', 'ModeloEscolhido', 'saida')
        
        # DEBUG: Imprimir informações de diagnóstico
        print("\n" + "="*80)
        print("DEBUG - CARREGAMENTO SHAP")
        print("="*80)
        print(f"Caminho base: {os.path.abspath(caminho_ml)}")
        print(f"Caminho existe? {os.path.exists(caminho_ml)}")
        
        if os.path.exists(caminho_ml):
            print(f"\nArquivos na pasta:")
            for arq in sorted(os.listdir(caminho_ml)):
                print(f"  - {arq}")
        
        # Arquivos HTML
        shap_html_files = {
            'popular': os.path.join(caminho_ml, 'shap_local_popular_force.html'),
            'impopular': os.path.join(caminho_ml, 'shap_local_impopular_force.html'),
            'multi': os.path.join(caminho_ml, 'shap_local_multi_examples.html')
        }
        
        # Arquivos PNG - NOMES CORRETOS BASEADOS NO DEBUG
        shap_png_files = {
            'feature_importance': os.path.join(caminho_ml, 'shap_feature_importance.png'),
            'summary': os.path.join(caminho_ml, 'shap_summary_global.png'),
            'beeswarm': os.path.join(caminho_ml, 'shap_beeswarm_global.png'),
            'multiclass': os.path.join(caminho_ml, 'shap_multiclass_importance.png'),
            'waterfall_popular': os.path.join(caminho_ml, 'shap_local_popular_waterfall.png'),
            'waterfall_impopular': os.path.join(caminho_ml, 'shap_local_impopular_waterfall.png'),
        }
        
        # Função para converter imagem PNG em base64
        def image_to_base64(filepath):
            try:
                if not os.path.exists(filepath):
                    print(f"  ✗ NÃO EXISTE: {os.path.basename(filepath)}")
                    return None
                
                with open(filepath, 'rb') as f:
                    data = f.read()
                    if len(data) == 0:
                        print(f"  ✗ ARQUIVO VAZIO: {os.path.basename(filepath)}")
                        return None
                    
                    encoded = base64.b64encode(data).decode('utf-8')
                    print(f"  ✓ CONVERTIDO: {os.path.basename(filepath)} ({len(encoded)} chars)")
                    return f"data:image/png;base64,{encoded}"
            except Exception as e:
                print(f"  ✗ ERRO ao converter {os.path.basename(filepath)}: {e}")
                return None
        
        # Ler conteúdo dos arquivos HTML
        print(f"\nCarregando arquivos HTML:")
        shap_html_contents = {}
        for key, filepath in shap_html_files.items():
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        shap_html_contents[key] = f.read()
                    print(f"  ✓ HTML {key}: OK")
                except Exception as e:
                    print(f"  ✗ Erro HTML {key}: {e}")
                    shap_html_contents[key] = f"<p>Erro ao carregar: {str(e)}</p>"
            else:
                print(f"  ✗ HTML {key}: NÃO ENCONTRADO")
        
        # Converter imagens PNG para base64
        print(f"\nConvertendo imagens PNG:")
        shap_images = {}
        for key, filepath in shap_png_files.items():
            result = image_to_base64(filepath)
            if result:
                shap_images[key] = result
        
        print(f"\nResumo:")
        print(f"  HTMLs carregados: {len(shap_html_contents)}")
        print(f"  Imagens carregadas: {len(shap_images)}")
        print("="*80 + "\n")
        
        # Verificar se tem algum conteúdo
        if not shap_html_contents and not shap_images:
            return html.Div([
                html.Hr(className="section-divider"),
                dbc.Row([
                    dbc.Col([
                        html.H4("SHAP Analysis", className="mt-3 mb-3"),
                        dbc.Alert([
                            html.I(className="fas fa-exclamation-triangle me-2"),
                            f"Arquivos SHAP não encontrados em: {os.path.abspath(caminho_ml)}"
                        ], color="warning")
                    ])
                ])
            ])
        
        # Criar lista de componentes
        components = []
        
        # Header
        components.extend([
            html.Hr(className="section-divider"),
            dbc.Row([
                dbc.Col([
                    html.H4("SHAP Analysis - Explicabilidade do Modelo XGBoost", className="mt-3 mb-3"),
                    html.P([
                        html.I(className="fas fa-info-circle me-2"),
                        "SHAP (SHapley Additive exPlanations) mostra como cada feature contribui para as predições do modelo."
                    ], className="text-muted")
                ])
            ])
        ])
        
        # ========== VISÃO GERAL ==========
        components.append(html.H5([
            html.I(className="fas fa-chart-bar me-2"),
            "Visão Geral - Importância Global das Features"
        ], className="mt-4 mb-3"))
        
        # Feature Importance
        if shap_images.get('feature_importance'):
            components.append(dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Feature Importance - Top Features", className="mb-3"),
                        html.Img(
                            src=shap_images['feature_importance'],
                            style={'width': '100%', 'max-width': '1000px'}
                        )
                    ], className="graph-container text-center")
                ], md=12)
            ], className="mb-4"))
        
        # Multiclass Importance
        if shap_images.get('multiclass'):
            components.append(dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Multiclass Feature Importance", className="mb-3"),
                        html.Img(
                            src=shap_images['multiclass'],
                            style={'width': '100%', 'max-width': '1000px'}
                        )
                    ], className="graph-container text-center")
                ], md=12)
            ], className="mb-4"))
        
        # Summary + Beeswarm lado a lado
        if shap_images.get('summary') or shap_images.get('beeswarm'):
            components.append(dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6("Summary Plot Global", className="mb-3"),
                        html.Img(
                            src=shap_images['summary'],
                            style={'width': '100%', 'max-width': '700px'}
                        ) if shap_images.get('summary') else html.P("Imagem não encontrada", className="text-muted")
                    ], className="graph-container text-center")
                ], md=6),
                dbc.Col([
                    html.Div([
                        html.H6("Beeswarm Plot Global", className="mb-3"),
                        html.Img(
                            src=shap_images['beeswarm'],
                            style={'width': '100%', 'max-width': '700px'}
                        ) if shap_images.get('beeswarm') else html.P("Imagem não encontrada", className="text-muted")
                    ], className="graph-container text-center")
                ], md=6)
            ], className="mb-4"))
        
        # ========== WATERFALL PLOTS ==========
        components.append(html.H5([
            html.I(className="fas fa-layer-group me-2"),
            "Waterfall Plots - Exemplos Individuais"
        ], className="mt-4 mb-3"))
        
        if shap_images.get('waterfall_popular') or shap_images.get('waterfall_impopular'):
            components.append(dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6([html.I(className="fas fa-star me-2"), "Livro POPULAR"], className="mb-3"),
                        html.Img(
                            src=shap_images['waterfall_popular'],
                            style={'width': '100%', 'max-width': '700px'}
                        ) if shap_images.get('waterfall_popular') else html.P("Imagem não encontrada", className="text-muted")
                    ], className="graph-container text-center")
                ], md=6),
                dbc.Col([
                    html.Div([
                        html.H6([html.I(className="fas fa-star-half-alt me-2"), "Livro IMPOPULAR"], className="mb-3"),
                        html.Img(
                            src=shap_images['waterfall_impopular'],
                            style={'width': '100%', 'max-width': '700px'}
                        ) if shap_images.get('waterfall_impopular') else html.P("Imagem não encontrada", className="text-muted")
                    ], className="graph-container text-center")
                ], md=6)
            ], className="mb-4"))
        
        # ========== FORCE PLOTS INTERATIVOS ==========
        components.append(html.H5([
            html.I(className="fas fa-bezier-curve me-2"),
            "Force Plots Interativos"
        ], className="mt-4 mb-3"))
        
        if 'popular' in shap_html_contents:
            components.append(dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6([html.I(className="fas fa-star me-2"), "Force Plot: Livro Popular"], className="mb-3"),
                        html.Iframe(
                            srcDoc=shap_html_contents['popular'],
                            style={'width': '100%', 'height': '400px', 'border': 'none'}
                        )
                    ], className="graph-container")
                ], md=12)
            ], className="mb-4"))
        
        if 'impopular' in shap_html_contents:
            components.append(dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6([html.I(className="fas fa-star-half-alt me-2"), "Force Plot: Livro Impopular"], className="mb-3"),
                        html.Iframe(
                            srcDoc=shap_html_contents['impopular'],
                            style={'width': '100%', 'height': '400px', 'border': 'none'}
                        )
                    ], className="graph-container")
                ], md=12)
            ], className="mb-4"))
        
        if 'multi' in shap_html_contents:
            components.append(dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H6([html.I(className="fas fa-layer-group me-2"), "Force Plot: Múltiplos Exemplos"], className="mb-3"),
                        html.Iframe(
                            srcDoc=shap_html_contents['multi'],
                            style={'width': '100%', 'height': '600px', 'border': 'none'}
                        )
                    ], className="graph-container")
                ], md=12)
            ], className="mb-4"))
        
        # ========== GUIA DE INTERPRETAÇÃO ==========
        components.append(dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5([html.I(className="fas fa-lightbulb me-2"), "Guia de Interpretação"]),
                        html.Hr(),
                        
                        html.H6("📊 Feature Importance:"),
                        html.Ul([
                            html.Li("Mostra quais features têm maior impacto médio nas predições"),
                            html.Li("Valores maiores = maior importância para o modelo")
                        ]),
                        
                        html.H6("🎯 Summary & Beeswarm Plot:", className="mt-3"),
                        html.Ul([
                            html.Li("Vermelho = valor alto da feature, Azul = valor baixo"),
                            html.Li("Eixo X: impacto no rating (positivo aumenta, negativo diminui)"),
                            html.Li("Cada ponto representa um livro diferente")
                        ]),
                        
                        html.H6("🌊 Waterfall Plots:", className="mt-3"),
                        html.Ul([
                            html.Li("Decomposição da predição para um livro específico"),
                            html.Li("Cada barra mostra o impacto de uma feature"),
                            html.Li("Começa do valor base (E[f(X)]) até a predição final (f(x))")
                        ]),
                        
                        html.H6("⚡ Force Plots Interativos:", className="mt-3"),
                        html.Ul([
                            html.Li("Versão interativa com hover para ver valores exatos"),
                            html.Li("Features em vermelho aumentam a predição"),
                            html.Li("Features em azul diminuem a predição"),
                            html.Li("Tamanho da barra = magnitude do impacto")
                        ])
                    ])
                ], className="shadow-sm")
            ], md=12)
        ], className="mb-4"))
        
        return html.Div(components)
    
    return html.Div()


@callback(
    Output('kpis-container', 'children'),
    Input('filtro-genero', 'value'),
    Input('filtro-idioma', 'value'),
    Input('filtro-editora', 'value'),
    Input('filtro-ano', 'value')
)
def atualizar_kpis(genero, idioma, editora, ano_range):
    df_filtrado = filtrar_dados(genero, idioma, editora, ano_range)
    kpis = {
        'total_livros': len(df_filtrado),
        'total_autores': df_filtrado['autor'].nunique(),
        'total_editoras': df_filtrado['editora'].nunique(),
        'media_rating': df_filtrado['rating'].mean(),
        'total_resenhas': df_filtrado['resenha'].sum(),
    }
    
    return dbc.Row([
        dbc.Col(criar_card_kpi("Total de Livros", f"{kpis['total_livros']:,}", "fas fa-book"), md=2),
        dbc.Col(criar_card_kpi("Total de Autores", f"{kpis['total_autores']:,}", "fas fa-users"), md=2),
        dbc.Col(criar_card_kpi("Total de Editoras", f"{kpis['total_editoras']:,}", "fas fa-building"), md=2),
        dbc.Col(criar_card_kpi("Rating Médio", f"{kpis['media_rating']:.2f}", "fas fa-star"), md=3),
        dbc.Col(criar_card_kpi("Total de Resenhas", f"{kpis['total_resenhas']:,}", "fas fa-comments"), md=3),
    ])

def filtrar_dados(genero, idioma, editora, ano_range):
    """Filtra o dataset baseado nos filtros selecionados"""
    df_filtrado = df.copy()
    
    if genero != 'todos':
        df_filtrado = df_filtrado[df_filtrado['genero'].str.contains(genero, na=False)]
    
    if idioma != 'todos':
        df_filtrado = df_filtrado[df_filtrado['idioma'] == idioma]
    
    if editora != 'todos':
        df_filtrado = df_filtrado[df_filtrado['editora'] == editora]
    
    df_filtrado = df_filtrado[(df_filtrado['ano'] >= ano_range[0]) & 
                              (df_filtrado['ano'] <= ano_range[1])]
    
    return df_filtrado


@callback(
    Output('top-livros-lidos', 'figure'),
    Input('filtro-genero', 'value'),
    Input('filtro-idioma', 'value'),
    Input('filtro-editora', 'value'),
    Input('filtro-ano', 'value')
)
def atualizar_top_livros(genero, idioma, editora, ano_range):
    df_filtrado = filtrar_dados(genero, idioma, editora, ano_range)
    top_10 = df_filtrado.nlargest(10, 'leram')[['titulo', 'leram']]
    
    fig = px.bar(top_10, x='leram', y='titulo', orientation='h',
                 title='Top 10 Livros Mais Lidos',
                 labels={'leram': 'Número de Leituras', 'titulo': 'Livro'},
                 color='leram', color_continuous_scale='Greens')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
    return fig


@callback(
    Output('top-autores', 'figure'),
    Input('filtro-genero', 'value'),
    Input('filtro-idioma', 'value'),
    Input('filtro-editora', 'value'),
    Input('filtro-ano', 'value')
)
def atualizar_top_autores(genero, idioma, editora, ano_range):
    df_filtrado = filtrar_dados(genero, idioma, editora, ano_range)
    top_autores = df_filtrado.groupby('autor')['leram'].sum().nlargest(10).reset_index()
    
    fig = px.bar(top_autores, x='leram', y='autor', orientation='h',
                 title='Top 10 Autores Mais Lidos',
                 labels={'leram': 'Total de Leituras', 'autor': 'Autor'},
                 color='leram', color_continuous_scale='Greens')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
    return fig


@callback(
    Output('top-editoras', 'figure'),
    Input('filtro-genero', 'value'),
    Input('filtro-idioma', 'value'),
    Input('filtro-editora', 'value'),
    Input('filtro-ano', 'value')
)
def atualizar_top_editoras(genero, idioma, editora, ano_range):
    df_filtrado = filtrar_dados(genero, idioma, editora, ano_range)
    
    top_editoras = df_filtrado.groupby('editora')['titulo'].nunique().nlargest(10).reset_index()
    top_editoras = top_editoras.rename(columns={'titulo': 'publicacoes'})

    fig = px.bar(top_editoras, x='publicacoes', y='editora', orientation='h',
                 title='Top 10 Editoras com Mais Publicações',
                 labels={'publicacoes': 'Quantidade de Títulos', 'editora': 'Editora'},
                 color='publicacoes', color_continuous_scale='Greens')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
    return fig


@callback(
    Output('livros-por-ano', 'figure'),
    Input('filtro-genero', 'value'),
    Input('filtro-idioma', 'value'),
    Input('filtro-editora', 'value'),
    Input('filtro-ano', 'value')
)
def atualizar_livros_ano(genero, idioma, editora, ano_range):
    df_filtrado = filtrar_dados(genero, idioma, editora, ano_range)
    livros_ano = df_filtrado.groupby('ano').size().reset_index(name='quantidade')
    
    fig = px.line(livros_ano, x='ano', y='quantidade',
                  title='Distribuição de Livros por Ano de Publicação',
                  labels={'ano': 'Ano', 'quantidade': 'Número de Livros'},
                  markers=True)
    fig.update_traces(line_color='#2ecc71')
    return fig


@callback(
    Output('dist-rating', 'figure'),
    Input('filtro-genero', 'value'),
    Input('filtro-idioma', 'value'),
    Input('filtro-editora', 'value'),
    Input('filtro-ano', 'value')
)
def atualizar_dist_rating(genero, idioma, editora, ano_range):
    df_filtrado = filtrar_dados(genero, idioma, editora, ano_range)
    
    fig = px.histogram(df_filtrado, x='rating', nbins=30,
                       title='Distribuição de Ratings',
                       labels={'rating': 'Rating', 'count': 'Frequência'})
    fig.update_traces(marker_color='#2ecc71')
    return fig


@callback(
    Output('scatter-rating-avaliacoes', 'figure'),
    Input('filtro-genero', 'value'),
    Input('filtro-idioma', 'value'),
    Input('filtro-editora', 'value'),
    Input('filtro-ano', 'value')
)
def atualizar_scatter_rating(genero, idioma, editora, ano_range):
    df_filtrado = filtrar_dados(genero, idioma, editora, ano_range)
    
    fig = px.scatter(df_filtrado, x='avaliacao', y='rating',
                     title='Rating vs Número de Avaliações',
                     labels={'avaliacao': 'Número de Avaliações', 'rating': 'Rating'},
                     opacity=0.6, hover_data=['titulo'])
    fig.update_traces(marker=dict(color='#27ae60'))
    return fig


@callback(
    Output('top-melhores-livros', 'figure'),
    Input('filtro-genero', 'value'),
    Input('filtro-idioma', 'value'),
    Input('filtro-editora', 'value'),
    Input('filtro-ano', 'value')
)
def atualizar_melhores_livros(genero, idioma, editora, ano_range):
    df_filtrado = filtrar_dados(genero, idioma, editora, ano_range)
    
    df_filtrado_min = df_filtrado[df_filtrado['avaliacao'] >= 10]
    top_10 = df_filtrado_min.nlargest(10, 'rating')[['titulo', 'rating', 'avaliacao']]
    
    fig = px.bar(top_10, x='rating', y='titulo', orientation='h',
                 title='Top 10 Livros com Melhor Rating (min. 10 avaliações)',
                 labels={'rating': 'Rating', 'titulo': 'Livro'},
                 color='rating', color_continuous_scale='YlGn',
                 hover_data=['avaliacao'])
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, showlegend=False)
    return fig


@callback(
    Output('top-piores-livros', 'figure'),
    Input('filtro-genero', 'value'),
    Input('filtro-idioma', 'value'),
    Input('filtro-editora', 'value'),
    Input('filtro-ano', 'value')
)
def atualizar_piores_livros(genero, idioma, editora, ano_range):
    df_filtrado = filtrar_dados(genero, idioma, editora, ano_range)
    
    df_filtrado_min = df_filtrado[df_filtrado['avaliacao'] >= 10]
    bottom_10 = df_filtrado_min.nsmallest(10, 'rating')[['titulo', 'rating', 'avaliacao']]
    
    fig = px.bar(bottom_10, x='rating', y='titulo', orientation='h',
                 title='Top 10 Livros com Menores Avaliações',
                 labels={'rating': 'Rating', 'titulo': 'Título'},
                 color='rating', color_continuous_scale='Greens',
                 hover_data=['avaliacao'])
    fig.update_layout(yaxis={'categoryorder': 'total descending'}, showlegend=False)
    return fig


@callback(
    Output('dist-generos', 'figure'),
    Input('filtro-idioma', 'value'),
    Input('filtro-ano', 'value')
)
def atualizar_dist_generos(idioma, ano_range):
    df_filtrado = df_generos_exploded.copy()
    
    if idioma != 'todos':
        df_filtrado = df_filtrado[df_filtrado['idioma'] == idioma]
    
    df_filtrado = df_filtrado[(df_filtrado['ano'] >= ano_range[0]) & 
                              (df_filtrado['ano'] <= ano_range[1])]
    
    top_20_generos = df_filtrado['genero'].value_counts().head(20)

    fig = px.bar(
        x=top_20_generos.values,
        y=top_20_generos.index,
        orientation='h',
        title='Top 20 Gêneros Literários (Contagem)',
        labels={'x': 'Quantidade', 'y': 'Gênero'},
        color=top_20_generos.values,
        color_continuous_scale='Greens'
    )
    fig.update_layout(showlegend=False, height=600)
    fig.update_yaxes(autorange='reversed')
    return fig


@callback(
    Output('rating-por-genero', 'figure'),
    Input('filtro-idioma', 'value'),
    Input('filtro-ano', 'value')
)
def atualizar_rating_genero(idioma, ano_range):
    df_filtrado = df_generos_exploded.copy()
    if idioma != 'todos':
        df_filtrado = df_filtrado[df_filtrado['idioma'] == idioma]
    df_filtrado = df_filtrado[(df_filtrado['ano'] >= ano_range[0]) & (df_filtrado['ano'] <= ano_range[1])]

    rating_genero = (
        df_filtrado.groupby('genero')
        .agg(media_rating=('rating', 'mean'), n=('titulo', 'count'))
        .reset_index()
    )
    
    rating_genero = rating_genero[rating_genero['n'] >= 5]
    rating_genero = rating_genero.sort_values('media_rating', ascending=False).head(15)

    fig = px.bar(
        rating_genero.sort_values('media_rating'),
        x='media_rating', y='genero', orientation='h',
        title='Top 15 Gêneros com Melhor Rating Médio',
        labels={'media_rating': 'Rating Médio', 'genero': 'Gênero'},
        color='media_rating', color_continuous_scale='YlGn'
    )
    fig.update_layout(showlegend=False, height=520)
    return fig


@callback(
    Output('generos-mais-lidos', 'figure'),
    Input('filtro-idioma', 'value'),
    Input('filtro-ano', 'value')
)
def atualizar_generos_lidos(idioma, ano_range):
    df_filtrado = df_generos_exploded.copy()
    
    if idioma != 'todos':
        df_filtrado = df_filtrado[df_filtrado['idioma'] == idioma]
    
    df_filtrado = df_filtrado[(df_filtrado['ano'] >= ano_range[0]) & 
                              (df_filtrado['ano'] <= ano_range[1])]
    
    generos_lidos = df_filtrado.groupby('genero').agg({
        'leram': 'sum',
        'abandonos': 'sum',
        'querem_ler': 'sum'
    }).nlargest(15, 'leram').reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Leram', x=generos_lidos['genero'], y=generos_lidos['leram'], marker_color='#2e7d32'))
    fig.add_trace(go.Bar(name='Abandonos', x=generos_lidos['genero'], y=generos_lidos['abandonos'], marker_color='#81c784'))
    fig.add_trace(go.Bar(name='Querem Ler', x=generos_lidos['genero'], y=generos_lidos['querem_ler'], marker_color='#4caf50'))
    
    fig.update_layout(
        title='Top 15 Gêneros: Comportamento de Leitura',
        xaxis_title='Gênero',
        yaxis_title='Quantidade',
        barmode='group'
    )
    return fig


@callback(
    Output('dist-idiomas', 'figure'),
    Input('filtro-genero', 'value'),
    Input('filtro-editora', 'value'),
    Input('filtro-ano', 'value')
)
def atualizar_dist_idiomas(genero, editora, ano_range):
    df_filtrado = filtrar_dados(genero, 'todos', editora, ano_range)
    idiomas = df_filtrado['idioma'].value_counts()

    fig = px.bar(
        x=idiomas.index,
        y=idiomas.values,
        title='Distribuição de Livros por Idioma (Contagem)',
        labels={'x': 'Idioma', 'y': 'Quantidade'},
        color=idiomas.values,
        color_continuous_scale='Greens'
    )
    fig.update_layout(showlegend=False)
    return fig


@callback(
    Output('rating-por-idioma', 'figure'),
    Input('filtro-genero', 'value'),
    Input('filtro-editora', 'value'),
    Input('filtro-ano', 'value')
)
def atualizar_rating_idioma(genero, editora, ano_range):
    df_filtrado = filtrar_dados(genero, 'todos', editora, ano_range)
    rating_idioma = df_filtrado.groupby('idioma')['rating'].mean().sort_values(ascending=False).reset_index()
    
    fig = px.bar(rating_idioma, x='idioma', y='rating',
                 title='Rating Médio por Idioma',
                 labels={'idioma': 'Idioma', 'rating': 'Rating Médio'},
                 color='rating', color_continuous_scale='YlGn')
    return fig


@callback(
    Output('metricas-leitura', 'figure'),
    Input('filtro-genero', 'value'),
    Input('filtro-idioma', 'value'),
    Input('filtro-editora', 'value'),
    Input('filtro-ano', 'value')
)
def atualizar_metricas_leitura(genero, idioma, editora, ano_range):
    df_filtrado = filtrar_dados(genero, idioma, editora, ano_range)
    
    metricas = {
        'Leram': df_filtrado['leram'].sum(),
        'Lendo': df_filtrado['lendo'].sum(),
        'Querem Ler': df_filtrado['querem_ler'].sum(),
        'Abandonos': df_filtrado['abandonos'].sum(),
        'Relendo': df_filtrado['relendo'].sum()
    }
    
   
    cores = {
        'Leram': '#2e7d32',
        'Lendo': '#388e3c',
        'Querem Ler': '#4caf50',
        'Abandonos': '#81c784',
        'Relendo': '#66bb6a'
    }
    fig = go.Figure()
    for nome, valor in metricas.items():
        fig.add_trace(go.Bar(name=nome, x=[nome], y=[valor], marker_color=cores.get(nome, '#2ecc71')))
    fig.update_layout(title='Métricas Gerais de Comportamento de Leitura', xaxis_title='Métrica', yaxis_title='Quantidade', showlegend=False)
    return fig


@callback(
    Output('taxa-abandono-genero', 'figure'),
    Input('filtro-idioma', 'value'),
    Input('filtro-ano', 'value')
)
def atualizar_taxa_abandono(idioma, ano_range):
    df_filtrado = df_generos_exploded.copy()
    if idioma != 'todos':
        df_filtrado = df_filtrado[df_filtrado['idioma'] == idioma]
    df_filtrado = df_filtrado[(df_filtrado['ano'] >= ano_range[0]) & (df_filtrado['ano'] <= ano_range[1])]

    abandono_genero = df_filtrado.groupby('genero').agg(
        abandonos=('abandonos', 'sum'),
        leram=('leram', 'sum'),
        lendo=('lendo', 'sum'),
        querem_ler=('querem_ler', 'sum')
    ).reset_index()
    
    abandono_genero['engajamento'] = abandono_genero['leram'] + abandono_genero['abandonos']
    abandono_genero = abandono_genero[abandono_genero['engajamento'] >= 20]
    
    denom = (abandono_genero['abandonos'] + abandono_genero['leram']).replace(0, pd.NA)
    abandono_genero['taxa_abandono'] = ((abandono_genero['abandonos'] / denom) * 100).clip(upper=100).fillna(0)
    top_15 = abandono_genero.sort_values('taxa_abandono', ascending=False).head(15)

    fig = px.bar(
        top_15.sort_values('taxa_abandono'),
        x='taxa_abandono', y='genero', orientation='h',
        title='Top 15 Gêneros com Maior Taxa de Abandono (%)',
        labels={'genero': 'Gênero', 'taxa_abandono': 'Taxa de Abandono (%)'},
        color='taxa_abandono', color_continuous_scale='Reds'
    )
    fig.update_layout(showlegend=False, height=520)
    fig.update_xaxes(ticksuffix='%', rangemode='tozero')
    return fig


@callback(
    Output('leitores-genero', 'figure'),
    Input('filtro-genero', 'value'),
    Input('filtro-idioma', 'value'),
    Input('filtro-editora', 'value'),
    Input('filtro-ano', 'value')
)
def atualizar_leitores_genero(genero, idioma, editora, ano_range):
    df_filtrado = filtrar_dados(genero, idioma, editora, ano_range)
    
    total_male = df_filtrado['male'].sum()
    total_female = df_filtrado['female'].sum()
    
    fig = px.pie(values=[total_male, total_female], names=['Masculino', 'Feminino'],
                 title='Distribuição de Leitores por Gênero',
                 color_discrete_sequence=["#90ce74", "#067923"])
    return fig


@callback(
    Output('generos-por-publico', 'figure'),
    Input('filtro-idioma', 'value'),
    Input('filtro-ano', 'value')
)
def atualizar_generos_publico(idioma, ano_range):
    df_filtrado = df_generos_exploded.copy()
    
    if idioma != 'todos':
        df_filtrado = df_filtrado[df_filtrado['idioma'] == idioma]
    
    df_filtrado = df_filtrado[(df_filtrado['ano'] >= ano_range[0]) & 
                              (df_filtrado['ano'] <= ano_range[1])]
    
    generos_publico = df_filtrado.groupby('genero').agg({
        'male': 'sum',
        'female': 'sum'
    })
    generos_publico['total'] = generos_publico['male'] + generos_publico['female']
    top_15 = generos_publico.nlargest(15, 'total').reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(name='Masculino', x=top_15['genero'], y=top_15['male'], marker_color='#90ce74'))
    fig.add_trace(go.Bar(name='Feminino', x=top_15['genero'], y=top_15['female'], marker_color='#067923'))
    
    fig.update_layout(
        title='Top 15 Gêneros: Leitores por Gênero do Público',
        xaxis_title='Gênero Literário',
        yaxis_title='Quantidade de Leitores',
        barmode='stack'
    )
    fig.update_xaxes(tickangle=45)
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8051)

import os
import base64

caminho_ml = os.path.join(os.path.dirname(__file__), '..', 'ml', 'ModeloEscolhido', 'saida')

print("="*80)
print("DIAGNÓSTICO - ARQUIVOS SHAP")
print("="*80)
print(f"\nCaminho procurado: {os.path.abspath(caminho_ml)}")
print(f"Caminho existe? {os.path.exists(caminho_ml)}")

if os.path.exists(caminho_ml):
    print(f"\nArquivos encontrados na pasta:")
    arquivos = os.listdir(caminho_ml)
    for arq in sorted(arquivos):
        filepath = os.path.join(caminho_ml, arq)
        tamanho = os.path.getsize(filepath) / 1024  # KB
        print(f"  ✓ {arq} ({tamanho:.1f} KB)")
    
    print(f"\n{'='*80}")
    print("TESTANDO CONVERSÃO BASE64")
    print("="*80)
    
    # Testar conversão de uma imagem
    teste_img = os.path.join(caminho_ml, 'shap_feature_importance.png')
    if os.path.exists(teste_img):
        try:
            with open(teste_img, 'rb') as f:
                encoded = base64.b64encode(f.read()).decode()
            print(f"\n✓ Conversão OK: {teste_img}")
            print(f"  Tamanho base64: {len(encoded)} caracteres")
            print(f"  Primeiros 100 chars: {encoded[:100]}...")
        except Exception as e:
            print(f"\n✗ ERRO na conversão: {e}")
    else:
        print(f"\n✗ Arquivo não encontrado: {teste_img}")
else:
    print("\n✗ PASTA NÃO EXISTE!")
