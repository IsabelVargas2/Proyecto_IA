from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import joblib
import numpy as np
import pandas as pd  # 🔹 Importar pandas

# 🔹 Aplicar estilo azul oscuro con CSS
st.markdown(
    """
    <style>
        body {
            background-color: #1e1e2f;
            color: white;
        }
        .stApp {
            background-color: #1e1e2f;
        }
        .css-18e3th9 {
            background-color: #1e1e2f !important;
        }
        .st-bw {
            color: white;
        }
        .st-dc {
            color: white;
        }
        .stSlider {
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# 🔹 Título personalizado
st.title("🌍 ¡Prediga su Aire! 💨")
st.subheader("Hecho por Isabel Vargas y Juan David Contreras")

# Cargar modelo y scaler
modelo = joblib.load("Mejor_Modelo.joblib")
scaler = joblib.load("scaler_actualizado.joblib")

# Verificar qué características espera el scaler
columnas_correctas = scaler.feature_names_in_  # 🔹 Nombres de columnas esperadas
st.write("📊 **Características esperadas por el scaler:**", columnas_correctas)

# Rango de valores reales
pm10_min, pm10_max = 0, 500
pm2_5_min, pm2_5_max = 0, 300
temp_min, temp_max = -10, 50
humedad_min, humedad_max = 0, 100

# 🔹 Controles de entrada con sliders
pm10 = st.slider("🌫 PM10 (µg/m³)", pm10_min, pm10_max, 50)
pm2_5 = st.slider("🌫 PM2.5 (µg/m³)", pm2_5_min, pm2_5_max, 25)
temperatura = st.slider("🌡 Temperatura (°C)", temp_min, temp_max, 20)
humedad = st.slider("💧 Humedad (%)", humedad_min, humedad_max, 70)

if st.button("🔍 Predecir Calidad del Aire"):
    # Crear un DataFrame con los nombres exactos que espera el scaler
    datos_usuario = np.array([[pm10, pm2_5, temperatura, humedad]])
    datos_usuario_df = pd.DataFrame(datos_usuario, columns=columnas_correctas)  # 🔹 Ajuste aquí

    st.write("📌 **Datos originales ingresados:**", datos_usuario_df)

    # Escalar los datos con el mismo scaler usado en entrenamiento
    datos_usuario_escalados = scaler.transform(datos_usuario_df)

    st.write("⚙️ **Datos después de escalar:**", datos_usuario_escalados)

    # Predecir con el modelo
    prediccion = modelo.predict(datos_usuario_escalados)

    # Verificar valores predichos
    st.write("🔮 **Predicción cruda del modelo:**", prediccion)

    # Mapeo de clases
    mapeo_calidad_aire = {0: "Buena ✅", 1: "Mala ❌", 2: "Regular ⚠️"}
    calidad_predicha = mapeo_calidad_aire.get(prediccion[0], "Desconocido")

    # Mostrar el resultado con color destacado
    st.success(f"🌱 **Calidad del Aire Predicha: {calidad_predicha}**")
