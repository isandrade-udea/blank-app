import streamlit as st
import pandas as pd
import numpy as np


st.title(":bus: Cootracovi ")
st.write(
    "Empresa trasnportadora de las rutas "
)

# Cargar datos desde un enlace de GitHub
st.header("Cargar Datos desde un Link de GitHub")
url = st.text_input("Introduce la URL del archivo CSV en GitHub")

# Verificar si se ha introducido una URL
if url:
    try:
        # Leer el archivo CSV directamente desde la URL
        df = pd.read_csv(url)
        
        # Mostrar las primeras filas del DataFrame
        st.write("Datos Cargados:")
        st.dataframe(df.head())  # Muestra las primeras filas del DataFrame
        
        # Aquí puedes agregar más análisis o visualizaciones usando df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
else:
    st.write("Por favor, introduce la URL de un archivo CSV en GitHub para continuar.")
#preprocesamiento

df = df.drop('Unnamed: 10', axis=1)
df = df.dropna()

# Eliminar la fila donde se repiten los nombres de las columnas
df = df[df['Fecha'] != 'Fecha']

#Tamaño del dataset
st.write(f"El tamaño del dataset es: {df.shape[0]} filas y {df.shape[1]} columnas.")


        
# Mostrar estadísticas descriptivas
st.subheader("Estadísticas Descriptivas")
st.write(df.describe())

