#librerias
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

st.title(":bus: Cootracovi ")
st.write(
    "Empresa trasnportadora de las rutas: HONDA - ORIENTAL - HONDA"
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

# prompt: eliminar la columna Unnamed: 10 y las filas vacias

df = df.drop('Unnamed: 10', axis=1)
df = df.dropna()

# Eliminar la fila donde se repiten los nombres de las columnas
df = df[df['Fecha'] != 'Fecha']

# prompt: cambiar formato de veichulos y pasajeros a int, viaje y kilometros a float, fecha

df['Vehiculo'] = df['Vehiculo'].astype(int)
df['Pasaj'] = df['Pasaj'].astype(int)
df['Viaje'] = df['Viaje'].astype(float)
df['Kms'] = df['Kms'].astype(float)

df['Fecha_Hora_Salida'] = pd.to_datetime(df['Fecha'] + ' ' + df['Hora salida'])
df['Fecha_Hora_Salida'] = pd.to_datetime(df['Fecha_Hora_Salida'], format='%Y-%m-%d %H:%M:%S')

df['Fecha_Hora_Final'] = pd.to_datetime(df['Fecha'] + ' ' + df['Hora Final'])
df['Fecha_Hora_Final'] = pd.to_datetime(df['Fecha_Hora_Final'], format='%Y-%m-%d %H:%M:%S')


# Convertir la columna 'Hora' al tipo datetime

def get_seconds(time_str):
    #print('Time in hh:mm:ss:', time_str)
    # split in hh, mm, ss
    hh, mm, ss = time_str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + int(ss)

#df['Hora_salida_s'] = df['Hora salida'].apply(get_seconds)
df['Tiempo_viaje_s'] = df['Tiempo Viaje'].apply(get_seconds)
df['Tiempo_viaje_s'] = df['Tiempo_viaje_s'].astype(float)
df['Tiempo_muerto_s'] = df['Tiempo Muerto'].apply(get_seconds)
df['Tiempo_muerto_s'] = df['Tiempo_muerto_s'].astype(float)

# Generar una columna para el día
df['Dia'] = df['Fecha_Hora_Salida'].dt.day
# crear columna con numero de ruta
df['n_ruta'] = df['Ruta'].str.split(' ', expand=True)[1].astype(float)

# Eliminar la columna fecha, ruta, Hora salida,	Hora Final,	Tiempo Viaje,Tiempo Muerto
#df = df.drop('Fecha', axis=1)

df = df.drop('Ruta', axis=1)
df = df.drop('Hora salida', axis=1)
df = df.drop('Hora Final', axis=1)
df = df.drop('Tiempo Viaje', axis=1)
df = df.drop('Tiempo Muerto', axis=1)
df = df.drop('Fecha', axis=1)

# Crear la columna 'Hora' con la hora extraída de 'Fecha_Hora_Salida'
df.loc[:, 'Hora'] = df['Fecha_Hora_Salida'].dt.hour

# Función para clasificar las horas en jornada, incluyendo la madrugada
def clasificar_jornada(hora):
    if 0 <= hora < 6:
        return 'Madrugada'
    elif 6 <= hora < 12:
        return 'Mañana'
    elif 12 <= hora < 18:
        return 'Tarde'
    else:
        return 'Noche'

# Aplicar la función para crear la columna 'Jornada'
df.loc[:, 'Jornada'] = df['Hora'].apply(clasificar_jornada)

#Resumen

# Borrar las filas donde la columna 'n_ruta' es igual0 a 2, 6 u 8
df = df[~df['n_ruta'].isin([2, 6, 8])]

# Eliminar las columnas 'Fecha_Hora_Final' y 'n_ruta'
df = df.drop(['Fecha_Hora_Final', 'n_ruta','Viaje'], axis=1)

# fijamos la columna como indice
df = df.set_index('Fecha_Hora_Salida')

# Paso 1: Eliminar índices duplicados, manteniendo la primera ocurrencia
df = df[~df.index.duplicated(keep='first')]

#Tamaño del dataset
st.write(f"El tamaño del dataset es: {df.shape[0]} filas y {df.shape[1]} columnas.")

# Análisis de la periodicidad del dataset
df['df_time_diffs'] = df.index.to_series().diff().dt.total_seconds()

fig, ax = plt.subplots(figsize=(4,3))
sns.histplot(df['df_time_diffs'].dropna(),ax=ax)

# Obtener los valores mínimo y máximo de la columna 'df_time_diffs'
min_val = df['df_time_diffs'].min()
max_val = df['df_time_diffs'].max()

# Configurar los límites de los ejes X e Y
ax.set_xlim(min_val, max_val)
# Mostrar el gráfico en Streamlit
st.pyplot(fig)






st.dataframe(df.head(30))
