import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1. Dados
dados = pd.read_csv("data/raw/california_housing.csv")
X = dados.drop(columns="MedHouseVal")
y = dados["MedHouseVal"]

# 2. Mesma divisão dos notebooks (o teste continua de fora)
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Treinar o modelo escolhido
modelo = make_pipeline(StandardScaler(), LinearRegression())
modelo.fit(X_treino, y_treino)

# 4. Salvar
joblib.dump(modelo, "models/regressao_linear.joblib")
print("Modelo salvo em models/regressao_linear.joblib")