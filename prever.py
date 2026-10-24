"""Usa o modelo salvo para prever o valor das casas de um bairro novo."""
import joblib
import pandas as pd

# 1. Carregar o modelo treinado
modelo = joblib.load("models/regressao_linear.joblib")

# 2. Um bairro novo
bairro = pd.DataFrame([{
    "MedInc": 5.0,        # renda: US$ 50 mil/ano
    "HouseAge": 20,
    "AveRooms": 5.5,
    "AveBedrms": 1.0,
    "Population": 1200,
    "AveOccup": 3.0,
    "Latitude": 34.05,    # Los Angeles
    "Longitude": -118.25,
}])

# 3. Prever e converter para dólares
valor = modelo.predict(bairro)[0]
print(f"Valor previsto: US$ {valor * 100_000:,.0f}")