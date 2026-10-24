# Regressão Linear – California Housing

Projeto de estudo: construir, passo a passo e do jeito que se faz no mercado, um pipeline de
Machine Learning que prevê o **valor médio das casas** em bairros da Califórnia.

> Este README é também um diário de aprendizado: cada etapa registra **o que foi feito** e
> **o que foi aprendido**.

## Sobre os dados

- **Dataset:** California Housing (censo dos EUA de 1990), disponível no scikit-learn via
  `sklearn.datasets.fetch_california_housing`.
- **Tamanho:** 20.640 bairros (linhas) × 9 colunas.
- **Alvo:** `MedHouseVal`, o valor médio das casas, em centenas de milhares de dólares.

| Coluna | Significado |
|---|---|
| `MedInc` | renda média do bairro (dezenas de milhares de dólares) |
| `HouseAge` | idade média das casas |
| `AveRooms` | média de cômodos por casa |
| `AveBedrms` | média de quartos por casa |
| `Population` | população do bairro |
| `AveOccup` | média de moradores por casa |
| `Latitude` / `Longitude` | localização |
| `MedHouseVal` | **alvo**: valor médio das casas |

O CSV **não é versionado**. Para recriar `data/raw/california_housing.csv`:

```python
from sklearn.datasets import fetch_california_housing

fetch_california_housing(as_frame=True).frame.to_csv("data/raw/california_housing.csv", index=False)
```

## Estrutura do projeto

```
regressao-linear/
├── data/
│   ├── raw/              # dados originais – nunca editados
│   └── processed/        # dados limpos/transformados
├── notebooks/            # exploração (EDA): rascunho, gráficos, perguntas
├── src/                  # código reaproveitável do pipeline
├── models/               # modelos treinados (.joblib)
├── reports/figures/      # gráficos de resultado
├── requirements.txt      # bibliotecas com versões fixadas
├── .gitignore
└── README.md
```

## Como rodar

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Roteiro

- [x] **0. Estrutura e ambiente:** pastas, `requirements.txt`, `.gitignore`, Git e GitHub
- [x] **1. Análise exploratória (EDA):** `notebooks/01_eda.ipynb`
- [x] **2. Métrica e modelo baseline:** `notebooks/02_baseline.ipynb`
- [ ] **3. Pré-processamento + regressão linear:** `notebooks/03_regressao.ipynb` *(em andamento)*
- [ ] **4. Validação cruzada**
- [ ] **5. Avaliação final no teste**
- [ ] **6. Salvar o modelo e fazer previsões**

---

## Diário de aprendizado

### Etapa 0 – Estrutura e ambiente

**O que foi feito**
- Criada a estrutura de pastas separando dados, exploração, código, modelos e relatórios.
- O CSV original foi colocado em `data/raw/`, que nunca é editada.
- O `requirements.txt` foi escrito com as versões fixadas.
- O `.gitignore` foi criado, o Git iniciado e o repositório publicado no GitHub com `gh repo create`.

**O que foi aprendido**
- **Notebook × `src/`:** o notebook serve para *explorar e decidir*, e o `src/` para *executar*.
  O que funciona no notebook vira código limpo no `src/`.
- **`data/raw` é intocável:** toda transformação gera um arquivo novo em `data/processed`, o que
  permite refazer tudo do zero.
- **Ambiente virtual (`.venv`):** uma "caixa" isolada de bibliotecas por projeto. O terminal
  mostra `(.venv)` quando ela está ativa.
- **`requirements.txt` não é código:** é uma lista de pacotes para o `pip`, um por linha e sem
  `import`. O nome para instalar pode ser diferente do nome no código, por exemplo
  `scikit-learn` → `import sklearn`.
- **Fixar versões (`pacote==versão`):** garante que o projeto continue funcionando no futuro. A
  versão escrita precisa ser a que está **realmente instalada**, conferida com `pip show pacote`.
- **`pip freeze` × lista manual:** o `freeze` despeja todas as dependências internas. A lista à
  mão contém só o que o código usa diretamente.
- **`.gitignore`:** fica fora do Git tudo que pode ser **recriado** (`.venv`, `__pycache__`,
  dados, modelos) ou que é **pessoal da máquina** (`.idea`). O código sempre entra.
- **`.gitkeep`:** o Git não guarda pastas vazias. Um arquivo vazio dentro delas garante que a
  estrutura chegue ao GitHub.
- **Commit em duas etapas:** `git add` escolhe o que entra e `git commit -m "..."` salva. Antes do
  commit, `git status` confere o que vai entrar. Uma boa mensagem diz o que o commit faz.

### Etapa 1 – Análise exploratória (EDA)

Notebook: `notebooks/01_eda.ipynb`

**O que foi feito no notebook**
- Carregamento do CSV com `pd.read_csv` e tamanho com `.shape` → 20.640 linhas × 9 colunas.
- Primeiro olhar com `.head()`, tipos e valores faltando com `.info()`.
- Resumo estatístico com `.describe()` e `.describe().T`.
- Correlação de cada coluna com o alvo: `dados.corr()["MedHouseVal"].sort_values(ascending=False)`.
- Dispersão `MedInc` × `MedHouseVal` (`dados.plot.scatter`).
- Histogramas de todas as colunas (`dados.hist(bins=50)`).
- Células Markdown com as observações, `## Conclusões` e `## Ideias para o modelo`.

**O que encontramos**

| Achado | Evidência no notebook | Impacto no modelo |
|---|---|---|
| Dados **completos** | `.info()`: todas as 9 colunas com 20.640 valores não nulos, todas `float64` | não é preciso preencher valores |
| `MedInc` é a variável mais relacionada ao preço | correlação ≈ 0.69; as demais ficam entre -0.14 e 0.15 | candidata principal; vamos testar "só renda × todas" |
| **Teto** no alvo | `.describe()`: máximo = 5.00001; na dispersão aparece como uma "parede" de pontos no topo | valores acima de US$ 500 mil foram cortados; o modelo vai errar nos bairros caros |
| **Valores errados** em `AveOccup` | `.describe()`: máximo ≈ 1.243 moradores por residência | outliers (alojamentos, presídios ou erro de dado) |
| **Teto** em `HouseAge` | histograma: barra alta isolada em 52 | idade "52" significa "52 ou mais" |
| Distribuições **assimétricas à direita** | histogramas de `MedInc` e `MedHouseVal`: cauda longa (típico de variáveis de dinheiro) | testar a transformação log |
| Escalas muito diferentes | `Population` na casa dos milhares, `MedInc` entre 0.5 e 15 | padronizar antes de treinar |

**Hipóteses para testar na Etapa 3**
- Só renda × todas as colunas.
- Com × sem os bairros no teto de 5.
- Log em `MedInc` e `MedHouseVal`.
- Localização: criar a *feature* "distância até LA/SF".

**Ficou de fora (opcional)**
- Mapa de calor (`sns.heatmap`): multicolinearidade `AveRooms` × `AveBedrms` (≈ 0.85).
- Mapa da Califórnia colorido pelo valor das casas.

**O que foi aprendido**
- **Cada linha é um bairro**, não uma casa. Quase todas as colunas são médias ou medianas do bairro, e
  cada uma tem a sua unidade (renda em US$ 10 mil, valor em US$ 100 mil).
- **Correlação de Pearson** (-1 a +1) mede relação **em linha reta**. Ela não prova causa, e um valor
  perto de 0 não significa "sem relação": a localização é o exemplo deste projeto.
- **Ler o mapa de calor:** a linha do alvo responde "o que se relaciona com o preço?". Os pares
  entrada × entrada respondem "tem colunas repetindo informação?".
- **A intuição gera hipóteses, mas quem decide são os dados:** "mais quartos = mais caro" deu ≈ 0.
- **Sem valores faltando ≠ dados limpos.** No `describe()`, compare o 75% com o máximo para achar
  outliers; valores máximos "redondos" demais (5.00001) indicam teto.
- **Outlier × teto (censura):** um outlier são poucos valores extremos e raros; um teto são muitos
  valores presos no mesmo limite porque o dado foi cortado na coleta.
- **Gráfico > número:** a correlação de 0.69 não mostra o teto, mas a dispersão mostra.
- **Histogramas parecidos ≠ variáveis relacionadas.** O histograma olha cada coluna sozinha. Embaralhar
  uma coluna mantém o histograma e zera a correlação. Quem mostra relação é a dispersão.
- **Cauda longa à direita** é típica de dinheiro (renda, preço). O `log` costuma deixar a relação mais linear.
- **Cada descoberta vira hipótese, e cada hipótese vira experimento** medido na etapa de modelagem.

### Etapa 2 – Métrica e modelo baseline

Notebook: `notebooks/02_baseline.ipynb`

**O que foi feito**
- Separação das entradas `X` (8 colunas) e do alvo `y` (`MedHouseVal`).
- Divisão em treino (80%, 16.512 bairros) e teste (20%, 4.128 bairros) com `train_test_split(..., random_state=42)`.
- Baseline `DummyRegressor(strategy="mean")`: prevê sempre a média do treino (≈ 2.07 = US$ 207 mil).
- Avaliação no teste com MAE, RMSE e R².

**Resultado do baseline (a linha de chegada mínima)**

| Métrica | Valor | Em dólares |
|---|---|---|
| MAE | 0.906 | erra em média ≈ US$ 90.600 |
| RMSE | 1.145 | ≈ US$ 114.500 (pune mais os erros grandes) |
| R² | -0.000 | não explica nada da variação dos preços |

Qualquer modelo daqui em diante precisa ter **RMSE bem abaixo de 1.145** e **R² bem acima de 0**.

**O que foi aprendido**
- **Um número sozinho não diz se o modelo é bom:** é preciso uma métrica e um baseline para comparar.
- **MAE** é o erro médio, fácil de explicar. **RMSE** eleva os erros ao quadrado antes da média, então
  pune mais os erros grandes; é a métrica principal do projeto. **R²** é a fração da variação explicada:
  1 = perfeito, 0 = igual a chutar a média, negativo = pior que a média.
- **Por que o R² deu -0.000:** o baseline chuta a média do **treino** (2.072), mas o R² compara com a
  média do **teste** (2.055). A diferença é minúscula, então o R² fica um "zero levemente negativo".
- **Treino × teste:** o modelo só é avaliado em dados que nunca viu. O teste fica "no cofre" até o fim.
- **`random_state`** fixa o sorteio da divisão e deixa o experimento reprodutível.
- **Convenção:** `X` maiúsculo (tabela de entradas) e `y` minúsculo (vetor do alvo).

### Etapa 3 – Regressão linear *(em andamento)*

Notebook: `notebooks/03_regressao.ipynb`

**Meta definida antes de rodar:** R² ≥ 0.60 e RMSE ≤ 0.75 na validação. Desafio: RMSE < 0.70.

**O que foi feito**
- Nova divisão: treino 60%, **validação 20%** (para comparar modelos) e teste 20% (guardado até a Etapa 5).
- Função `avaliar()` para treinar e medir qualquer modelo na validação.
- Pipeline `make_pipeline(StandardScaler(), LinearRegression())`.
- Experimento 1: **só renda × todas as colunas**.

**Experimento 1: resultados na validação**

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Baseline (média) | 0.925 | 1.172 | -0.000 |
| Só `MedInc` | 0.631 | 0.843 | 0.483 |
| **Todas as colunas** | **0.533** | **0.728** | **0.614** |

- A regressão com todas as colunas **bateu a meta principal** (R² 0.614 ≥ 0.60; RMSE 0.728 ≤ 0.75).
- O erro médio caiu de ≈ US$ 92 mil (baseline) para ≈ US$ 53 mil, cerca de 42% menos.
- As colunas com correlação baixa **ajudaram**: o RMSE caiu de 0.843 para 0.728 em relação a "só renda".
- O baseline mudou de 1.145 (Etapa 2) para 1.172 porque agora é medido na validação, não no teste.

**Coeficientes do modelo com todas as colunas (variáveis padronizadas)**

| Coluna | Coeficiente | Leitura |
|---|---|---|
| `Latitude` | -0.888 | mais ao norte → mais barato (mantendo as outras fixas) |
| `Longitude` | -0.862 | mais a leste (longe do mar) → mais barato |
| `MedInc` | 0.860 | mais renda → mais caro |
| `AveBedrms` | 0.312 | sinais opostos por causa da multicolinearidade: |
| `AveRooms` | -0.280 | não interpretar separadamente |
| `HouseAge` | 0.120 | efeito pequeno |
| `AveOccup` | -0.026 | quase nada |
| `Population` | -0.010 | quase nada |

- A **localização tem os maiores pesos**, apesar da correlação ≈ 0 na EDA. Juntas, latitude e
  longitude negativas formam um "eixo litoral": sudoeste (costa, LA) → caro; nordeste (interior) → barato.
- `AveRooms` e `AveBedrms` "brigam" entre si (correlação ≈ 0.85): um peso compensa o outro.

**O que foi aprendido**
- **Coeficiente padronizado:** quanto o preço muda quando a coluna sobe um desvio padrão, **mantendo as
  outras fixas**. Como estão na mesma escala, dá para compará-los.
- **Correlação isolada × peso no modelo:** a latitude tinha correlação -0.14 e virou o maior peso. A
  correlação olha uma coluna sozinha; o modelo olha o efeito dela descontando as outras.
- **Multicolinearidade na prática:** colunas muito correlacionadas ganham pesos grandes e de sinais
  opostos. O modelo continua prevendo bem, mas os coeficientes individuais deixam de ser confiáveis.
- **Não comparar modelos no teste:** escolher o melhor olhando o teste "contamina" a prova final. Para
  comparar, use um conjunto de **validação** (o simulado) e deixe o teste para o fim.
- **Meta antes de rodar:** combina o piso (baseline), o uso real (erro relativo aceitável) e o realista
  (o limite do modelo). Assim evita-se aceitar qualquer número depois.
- **Padronização (`StandardScaler`):** coloca as colunas na mesma escala (média 0, desvio 1).
- **`Pipeline` e data leakage:** o pipeline garante que o scaler aprenda só com o treino; aprender com a
  validação ou o teste vazaria informação.
- **Correlação baixa ≠ coluna inútil:** juntas, as colunas se complementam. A localização ajuda a
  diferenciar bairros de mesma renda.
- **`X[["coluna"]]`** (colchetes duplos) mantém o formato de tabela que o scikit-learn exige.
