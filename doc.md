# Documentação do Projeto: MLOps Bank Marketing API

## 1. Arquitetura da Solução
A arquitetura de software aplicada no projeto adere à separação de responsabilidades (SoC), partindo de uma divisão limpa entre Model-Building e Serving:
- **Camada de Treinamento (`scripts/train_model.py`)**: Script modular desconectado da aplicação principal. Realiza desde o acesso remoto (UCI ML Repo) a limpeza e salvamento final. Fica encarregado apenas nas etapas da esteira de CI/CD para gerar ou reter o artefato do modelo preditivo sem intervir no sistema em execução.
- **Camada de Serving (Pasta `app/`)**: Serviço online altamente otimizado construído sobre a framework FastAPI assíncrona. Puxa imediatamente o modelo treinado da memória do repositório no momento da invocação/startup do servidor (Padrão de gerência com `lifespan`) com técnica Singleton no seu handler para isolar desperdício da Memória RAM por duplicações em escala.

## 2. Decisões Técnicas
- **Scikit-Learn Pipeline**: Para evitar vazamentos de dados do seu treinamento para predição (Data Leakage) e poupar processamento manual nas requisições da web, encapsulamos um `ColumnTransformer` rigoroso (com `OneHotEncoder` em dados categóricos e `StandardScaler` em vetores numéricos) num contêiner sequencial `Pipeline` ligado ao próprio algoritmo preditor. Os dados submetidos pelo usuário são purgados e ajustados sozinhos.
- **RandomForestClassifier**: Foi o algoritmo de baseline escolhido por garantir performance robusta de imediato diante do dataset do "Bank Marketing", que lida com alta complexidade e heterogeneidade de informações, sem sofrer grandes estragos com prováveis outliers, exigindo menos refinação (tuning) agressiva da infraestrutura.

## 3. Validação de Dados
- **Uso Estrito e Intrusivo do Pydantic v2**: Foi utilizada uma malha extremamente controladora atuando como a Primeira Linha de Defesa. 
- Mapeou-se rigidamente constantes matemáticas como `Field(..., ge=18)` forçando a aplicação de regra de negócio simples mas essenciais sobre flutuantes em limites reais operantes.
- Encaixe cirúrgico e textual com o método `Literal[]`: não há adivinhação. A API não deixa nada atravessar a sua interface que não seja categoricamente mapeável nas strings exatas validadas pelo `OneHotEncoder`.
- Por fim a inserção do `model_config = ConfigDict(extra="forbid")` para garantir que lixo, ruídos operacionais e payloads abusivos corrompam o fluxo que será repassado para o DataFrame das predições.

## 4. Estratégia de Testes
O conjunto é garantido usando frameworks limpas e assertivas da comunidade, dividindo a validação da camada de software:
- **Testes Unitários**: Pontuado no arquivo `test_schemas.py`, ele mira de forma local o núcleo validativo do projeto antes dele tocar no `FastAPI`, verificando em isolamento reações de pânico e sucesso ao receber dados fora de padrão/escala ou envenenados na classe abstrata.
- **Testes de Integração**: Exposto no `test_api.py`, recria localmente no código uma camada transacional de Web API (`ASGITransport`) agindo idêntica a requests no Uvicorn/Rede, sem precisar iniciar de fato uma porta. Traz verificação sólida das diretrizes simulando payloads valiosos ponta a ponta sobre a integração da rota (`/predict`) lidando contra erros `HTTP 503` e `HTTP 422`.

## 5. Fluxo de Dados Integral
1. O Cliente realiza chamada `POST /predict`.
2. O FastAPI envia os metadados e o conteúdo `JSON` no Pydantic no Schema base `BankMarketingInput`. 
3. *Caso Falho:* A API encerra com `422 Unprocessable Entity`.
4. *Caso Certo:* O fluxo segue ao Singleton `ModelHandler()`.
5. Um utilitário de transição transforma seu tipado rígido em um `pandas.DataFrame`.
6. A árvore passa e limpa o DataFrame em seu pipeline `ColumnTransformer`.
7. O classificador Random Forest cospe a resolução final preditiva.
8. A resposta literal é exposta à porta do cliente obedecendo estritamente ao contrato final do construtor de saída com `HTTP 200`.
