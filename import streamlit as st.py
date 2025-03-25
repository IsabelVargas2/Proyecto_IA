import streamlit as st
import pickle
import numpy as np

# Cargar el modelo y el scaler
with open("modelo_svm.pkl", "rb") as model_file:
    modelo_svm = pickle.load(model_file)
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Configuración de la página
st.set_page_config(page_title="Sepa la calidad de su aire!!!!", page_icon="🌍", layout="centered")
st.markdown("""
    <style>
        .stApp {
            background-color: #ADD8E6;
        }
    </style>
""", unsafe_allow_html=True)

# Título y subtítulo
st.title("Sepa la calidad de su aire!!!!")
st.subheader("Hecho por Isabel Verga y Juan David Contreras")

# Sliders para entrada de datos
temperature = st.slider("Temperatura (°C)", -10, 50, 25)
humidity = st.slider("Humedad (%)", 0, 100, 50)
pm25 = st.slider("PM2.5 (µg/m³)", 0, 500, 50)
pm10 = st.slider("PM10 (µg/m³)", 0, 500, 50)
no2 = st.slider("NO2 (ppb)", 0, 200, 50)
so2 = st.slider("SO2 (ppb)", 0, 100, 10)
co = st.slider("CO (ppm)", 0.0, 10.0, 1.0)

# Botón de predicción
if st.button("Predecir", help="Haz clic para obtener la predicción de calidad del aire", key="predecir", use_container_width=True):
    # Preparar los datos para la predicción
    datos_usuario = np.array([[temperature, humidity, pm25, pm10, no2, so2, co]])
    datos_usuario_escalados = scaler.transform(datos_usuario)
    
    # Realizar la predicción
    prediccion = modelo_svm.predict(datos_usuario_escalados)[0]
    
    # Mapear categorías
    categorias = {0: "Buena", 1: "Excelente", 2: "Mala", 3: "Regular"}
    resultado = categorias.get(prediccion, "Desconocido")
    
    # Mensaje para población vulnerable
    mensajes = {
        "Buena": "El aire es saludable para todos. Puedes realizar actividades al aire libre sin problemas.",
        "Excelente": "El aire está en su mejor condición. Ideal para actividades al aire libre.",
        "Mala": "La calidad del aire puede afectar a grupos sensibles. Se recomienda limitar la exposición prolongada.",
        "Regular": "Algunas personas pueden experimentar molestias. Se aconseja precaución a quienes tienen problemas respiratorios."
    }
    
    # Mostrar el resultado
    st.markdown(f"### Predicción: **{resultado}**")
    st.info(mensajes.get(resultado, "No hay información disponible."))
