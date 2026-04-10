# Bank Marketing MLOps API

## Sobre o Projeto
Projeto de estudo em MLOps para construção de um modelo de Machine Learning. O sistema treina um classificador para prever a adesão de clientes a depósitos a prazo, utilizando a base de dados Bank Marketing do UCI Machine Learning Repository (ID: 222).

## Capacidades do Projeto
- **Treinamento**: Pipeline que extrai dados via API UCI, pre-processa categorias e exporta um artefato de inferência serializado.
- **Web API**: Modelo acessível usando FastAPI.
- **Segurança e Validação**: Entradas estritamente controladas com Pydantic V2, implementando limites de variáveis numéricas e recusa de parâmetros adicionais (extra="forbid").
- **Testes Automatizados**: Casos de testes (Pytest) validando tanto instâncias isoladas (schemas) quanto simulações completas da carga do servidor.
- **Conteinerização**: Contêiner configurado via Docker.

## Arquitetura do Sistema
A arquitetura adota a separação de responsabilidades.

- **`app/main.py`**: Ponto de entrada do projeto FastAPI. Gerencia o ciclo de vida e a injeção do modelo, além de dispor as rotas `/predict`, `/health` e `/info`.
- **`app/model.py`**: Lógica de inferência isolada. Implementa o padrão Singleton para manter o `.joblib` mapeado restritamente à inicialização original reduzindo Overhead, ligando de forma direta inferências Pydantic ao output numérico.
- **`app/schemas.py`**: Contrato Pydantic contendo os Input/Output aceitáveis do classificador da rede para proteção e recusa de carga morta em endpoints.
- **`scripts/train_model.py`**: Automação independente com o único propósito de construir os blocos preditivos matemáticos do `ColumnTransformer` limitados a conversão e gerência de dados nativamente processados no `RandomForest`. Exporta seu final de forma hermética.
- **`tests/test_api.py`** e **`tests/test_schemas.py`**: Validações assegurando as fronteiras da camada API usando HTTP Status codes, atestando o funcionamento da base mesmo com instabilidades externas simuladas no JSON do front-end.

## Setup e Instalação

### 1. Dependências do Sistema
O ambiente foi construído focado em Python 3.9+. O arquivo `requirements.txt` mapeia os essenciais:
- scikit-learn, pandas, joblib
- ucimlrepo
- fastapi, uvicorn, pydantic
- pytest, pytest-asyncio, httpx

### 2. Configurando o Ambiente
Abra o repositório principal no terminal e inicie a separação global montando uma maquina virtual:

```bash
# Construção limpa
python -m venv .venv

# Ativação via Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
# Ou Ativação via Windows (CMD Padrão)
.\.venv\Scripts\activate.bat

# Injeção de dependências
pip install -r requirements.txt
```

### 3. O Dataset Definido
Utilizei dados consumidos direto na base internacional do *UCI Machine Learning* correspondendo referencial 222 (Bank Marketing), focado em tabular contatos diretos (target `y`) determinando o fechamento do acordo financeiro de usuários de perfil diversificado.

### 4. Executando o Projeto

**Treinamento**
Inicia o processo de extração dos dados (UCI), pré-processamento e treinamento do modelo de Machine Learning. Ao finalizar, o artefato treinado será salvo automaticamente como `models/bank_marketing_model.joblib` para ser consumido pela API:
```bash
python scripts/train_model.py
```

**API de Inferência**
Com o modelo treinado salvo e pronto, inicie o servidor da API REST. O Uvicorn hospedará a aplicação localmente (por padrão, na porta 8000):
```bash
uvicorn app.main:app --reload
```
Acesse a interface web interativa da API: `http://localhost:8000/docs`.


## Índice de Arquivos
```text
mlops_final/
 ├── app/                    
 │    ├── config.py          # Constantes estritas (Hyperparametros, Ids e Versionamento API).
 │    ├── main.py            # Servidor FastAPI.
 │    ├── model.py           # Abstração de previsões ML via Handler e Singleton.
 │    └── schemas.py         # Arquitetura de payload e restrições de formatação DTO do Pydantic.
 ├── models/                 # Diretório da hospedagem dos builds (.joblib).
 ├── scripts/
 │    └── train_model.py     # Fluxo centralizado de conversão estatística, pipeline e gerência extrativa.
 ├── tests/                  
 │    ├── test_api.py        # Cobertura asgard-teste ASGITransport aferindo 422 Unprocessable Error x 200 Valid.
 │    └── test_schemas.py    # Cobertura restritiva na validade das fronteiras numéricas do Python de inserções falhas.
 ├── doc.md                  # Relatório analítico argumentando decisões técnicas dos engenheiros para as squads.
 ├── Dockerfile              # Docker portando a API crua de base slim otimizando as importações limpas à nuvem.
 ├── requirements.txt      
 └── README.md
```