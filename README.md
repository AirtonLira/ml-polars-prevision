# Projeto de Previsão de Preços de Casas

Este projeto utiliza o algoritmo `GradientBoostingRegressor` para prever os preços de casas com base em um conjunto de dados fornecido. Abaixo está uma explicação detalhada de cada etapa do processo.

## Carregamento dos Dados
```
train_df = pd.read_csv('train.csv') 
test_df = pd.read_csv('test.csv')
```

- **Descrição**: Carrega os dados de treinamento e teste a partir de arquivos CSV.
- **Motivo**: Precisamos dos dados para treinar o modelo (dados de treinamento) e para fazer previsões (dados de teste).

## Identificação e Tratamento de Colunas Numéricas
``numeric_cols = train_df.select_dtypes(include=['number']).columns.drop('SalePrice')``
- **Descrição**: Identifica as colunas numéricas no conjunto de dados de treinamento, excluindo a coluna alvo `'SalePrice'`.
- **Motivo**: Precisamos saber quais colunas são numéricas para aplicar técnicas de preenchimento de valores ausentes e normalização.

## Preenchimento de Valores Ausentes

```
train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].median())
test_numeric_cols = test_df.select_dtypes(include=['number']).columns
test_df[test_numeric_cols] = test_df[test_numeric_cols].fillna(test_df[test_numeric_cols].median())
```

- **Descrição**: Preenche valores ausentes nas colunas numéricas com a mediana de cada coluna.
- **Motivo**: A mediana é uma medida robusta de tendência central que não é afetada por outliers, garantindo que o preenchimento não distorça a distribuição dos dados.

## Codificação de Variáveis Categóricas

```
train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)
```
- **Descrição**: Converte variáveis categóricas em variáveis dummy (ou one-hot encoding).
- **Motivo**: Modelos de machine learning geralmente requerem dados numéricos. `get_dummies` transforma categorias em colunas binárias, permitindo que o modelo as utilize.

## Alinhamento dos Conjuntos de Dados
``train_df, test_df = train_df.align(test_df, join='left', axis=1, fill_value=0)``

- **Descrição**: Garante que os conjuntos de dados de treinamento e teste tenham as mesmas colunas.
- **Motivo**: Após a codificação, pode haver discrepâncias nas colunas entre os conjuntos de dados. Alinhar os conjuntos garante que o modelo receba a mesma estrutura de dados para treinamento e previsão.

## Separação dos Dados

```
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

- **Descrição**: Separa os dados de treinamento em características (`X`) e alvo (`y`), e divide em conjuntos de treinamento e validação.
- **Motivo**: A separação permite treinar o modelo em uma parte dos dados e validar seu desempenho em outra, ajudando a evitar overfitting.

## Treinamento do Modelo
``model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)``

- **Descrição**: Inicializa e treina um modelo de regressão usando `GradientBoostingRegressor`.
- **Motivo**: O `GradientBoostingRegressor` é um algoritmo poderoso que combina múltiplas árvores de decisão para melhorar a precisão. Ele é escolhido por sua capacidade de lidar bem com dados complexos e fornecer previsões precisas.

### Funcionamento do GradientBoostingRegressor

- **Como funciona**: O Gradient Boosting cria um modelo forte a partir de modelos fracos (árvores de decisão), treinando cada nova árvore para corrigir os erros das árvores anteriores. Ele otimiza uma função de perda usando gradientes, daí o nome "boosting".

## Avaliação do Modelo
```
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f'Validation RMSE: {rmse:.2f}')
```

- **Descrição**: Faz previsões no conjunto de validação e calcula o erro quadrático médio (RMSE).
- **Motivo**: RMSE é uma métrica que mede a diferença entre os valores previstos e os reais. Um RMSE menor indica um modelo mais preciso.

### Significado do RMSE

- **O que é**: RMSE (Root Mean Squared Error) é a raiz quadrada da média dos quadrados dos erros. É uma medida de quão bem o modelo está prevendo os valores reais.
- **Por que é útil**: Fornece uma medida de erro em unidades comparáveis aos dados originais, facilitando a interpretação.

## Previsões no Conjunto de Teste
``test_predictions = model.predict(test_df.drop('SalePrice', axis=1, errors='ignore'))``

- **Descrição**: Faz previsões para o conjunto de dados de teste.
- **Motivo**: Precisamos prever os preços das casas no conjunto de teste para gerar um arquivo de submissão.

## Preparação do Arquivo de Submissão
```
submission = pd.DataFrame({
'Id': test_df['Id'],
'SalePrice': test_predictions
})
submission.to_csv('submission.csv', index=False)
```

- **Descrição**: Cria um DataFrame com IDs e preços previstos, e salva como um arquivo CSV.
- **Motivo**: O arquivo `submission.csv` é usado para enviar previsões em competições (como Kaggle). Os IDs correspondem às casas no conjunto de teste, garantindo que cada previsão seja associada à casa correta.

## Detalhes Adicionais

- **get_dummies**: Esta função transforma variáveis categóricas em colunas binárias. Por exemplo, uma coluna com valores ['A', 'B', 'C'] se torna três colunas: `A`, `B`, `C`, com valores 0 ou 1 indicando a presença de cada categoria.
- **Importância do Alinhamento**: Após a codificação, é crucial alinhar os conjuntos de dados para garantir que ambos tenham as mesmas colunas, evitando erros durante a previsão.

Espero que esta explicação detalhada ajude a entender cada passo do projeto e o raciocínio por trás das escolhas feitas. Se tiver mais perguntas ou precisar de mais detalhes, sinta-se à vontade para perguntar no meu Linkedln: linkedin.com/in/airton-lira-junior-6b81a661