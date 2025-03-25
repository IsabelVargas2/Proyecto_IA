import joblib
import pandas as pd

# Cargar el scaler
scaler = joblib.load("scaler_actualizado.joblib")

# Verificar los valores mínimos y máximos usados para escalar
print("Mínimos usados para escalar:", scaler.data_min_)
print("Máximos usados para escalar:", scaler.data_max_)
