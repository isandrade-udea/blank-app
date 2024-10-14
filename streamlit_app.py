#librerias
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
# Modelado y Forecasting
import skforecast
import lightgbm
import sklearn
from lightgbm import LGBMRegressor
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from xgboost import XGBRegressor
from skforecast.model_selection import backtesting_forecaster

st.title(":bus: Cootracovi ")
st.write(
    "Empresa trasnportadora de las rutas: HONDA - ORIENTAL - HONDA"
)

# Cargar datos desde un enlace de GitHub
st.header("Cargar Datos desde un Link de GitHub")
url = st.text_input("Introduce la URL del archivo CSV en GitHub")

def cargar_csv_desde_url(url):
    # Verificar si se ha introducido una URL
    if url:
        try:
            # Leer el archivo CSV directamente desde la URL
            df = pd.read_csv(url)
            return df
        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")
            return None
    else:
        st.write("Por favor, introduce la URL de un archivo CSV en GitHub para continuar.")
        return None

df = cargar_csv_desde_url(url)

# Mostrar las primeras filas del DataFrame
st.write("Datos Cargados:")
st.dataframe(df.head(8)) # Muestra las primeras filas del DataFrame

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

#organizamos en orden cronologico
df.sort_index(inplace=True)

# Paso 1: Eliminar índices duplicados, manteniendo la primera ocurrencia
df = df[~df.index.duplicated(keep='first')]

#Tamaño del dataset
st.write(f"El tamaño del dataset es: {df.shape[0]} filas y {df.shape[1]} columnas.")

# Análisis de la periodicidad del dataset
df['df_time_diffs'] = df.index.to_series().diff().dt.total_seconds()

fig, ax = plt.subplots(figsize=(6.5,2))
# Crear el histograma con KDE
sns.histplot(df['df_time_diffs'].dropna(), kde=True, ax=ax)

# Obtener los valores mínimo y máximo de la columna 'df_time_diffs'
min_val = df['df_time_diffs'].min()
max_val = df['df_time_diffs'].max()

# Calcular la mediana de las diferencias de tiempo
mediana_dif = df['df_time_diffs'].median()
# Convertir la mediana a minutos
mediana_minutos = mediana_dif / 60

# Configurar los límites del eje X
ax.set_xlim(0, max_val)

# Asignar nombres a los ejes y el título
ax.set_xlabel('Diferencia entre observaciones (segundos)')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribución de la periodicidad ')

# Agregar texto sobre la mediana en el gráfico
ax.axvline(mediana_dif, color='r', linestyle='--', label='Mediana: {:.2f} s ({:.2f} min)'.format(mediana_dif, mediana_minutos))
ax.legend()

# Mostrar el gráfico en Streamlit
st.pyplot(fig)

# Mensaje sobre la mediana
st.write(f"La frecuencia mediana es de {mediana_dif:.2f} segundos, que son {mediana_minutos:.2f} minutos. La vamos a tomar como 5 minutos.")

# Cambiar la frecuencia a 5 minutos ('5T') y rellenar valores faltantes con bfill
df2 = df.asfreq(freq='5T', method='bfill')

df2 = df2.rename(columns={'Fecha_Hora_Salida': 'Fecha_Hora'})

# Separación datos train-val-test 70% 15% 15%

train_size = 0.7  # 70% para entrenamiento
val_size = 0.15   # 15% para validación
test_size = 0.25  # 15% para prueba

# Calcular los índices para hacer la separación
n = len(df2)
train_end = int(train_size * n)
val_end = int((train_size + val_size) * n)

# Separar los datos en conjuntos de entrenamiento, validación y prueba
train = df2[:train_end]  # Desde el inicio hasta el 70% de los datos
val = df2[train_end:val_end]  # Del 70% al 85%
test = df2[val_end:]  # Desde el 85% hasta el final

# Verificar el tamaño de cada conjunto
print(f'Tamaño conjunto de entrenamiento: {len(train)}')
print(f'Tamaño conjunto de validación: {len(val)}')
print(f'Tamaño conjunto de prueba: {len(test)}')

st.dataframe(df2.head())

st.write("Serie de tiempo para pasajeros")
# Crear la figura
fig = go.Figure()

# Agregar las trazas para entrenamiento, validación y prueba
fig.add_trace(go.Scatter(x=train.index, y=train['Pasaj'], mode='lines', name='Train'))
fig.add_trace(go.Scatter(x=val.index, y=val['Pasaj'], mode='lines', name='Validation'))
fig.add_trace(go.Scatter(x=test.index, y=test['Pasaj'], mode='lines', name='Test'))

# Configurar el layout de la figura
fig.update_layout(
    xaxis_title="Fecha",
    yaxis_title="Pasajeros",
    legend_title="Partición:",
    width=850,
    height=400,
    margin=dict(l=20, r=20, t=35, b=20),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1,
        xanchor="left",
        x=0.001,
    )
)

# Mostrar el range slider en el eje X
fig.update_xaxes(rangeslider_visible=True)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

st.write("Gráficos de estacionalidad")

col1, col2, col3 = st.columns(3)

# Agregar contenido en la primera columna
with col1:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    df2['dia'] = df2.index.day_name()
    df2.boxplot(column='Pasaj', by='dia', ax=ax,)
    df2.groupby('dia')['Pasaj'].median().plot(style='o-', linewidth=0.8, ax=ax)
    ax.set_ylabel('Pasajeros')
    ax.set_title('Distribución pasajeros por dia')
    st.pyplot(fig)

# Agregar contenido en la segunda columna
with col2:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    df2['hora'] = df2.index.hour
    df2.boxplot(column='Pasaj', by='hora', ax=ax,)
    df2.groupby('hora')['Pasaj'].median().plot(style='o-', linewidth=0.8, ax=ax)
    ax.set_ylabel('Pasajeros')
    ax.set_title('Distribución pasajeros por hora')
    st.pyplot(fig)

with col3:
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    # Orden para las jornadas
    jornada_order = ['Madrugada', 'Mañana', 'Tarde', 'Noche']
    
    # Crear el boxplot con transparencia
    sns.boxplot(x='Jornada', y='Pasaj', data=df, ax=ax, order=jornada_order,
                boxprops=dict(facecolor='none', edgecolor='blue'),  # Transparente con bordes azules
                medianprops=dict(color='green', lw=1),  # Línea de la mediana más gruesa y verde
                whiskerprops=dict(color='blue'),  # Líneas de los bigotes
                capprops=dict(color='blue'),  # Extremos de los bigotes
                flierprops=dict(marker='o', color='red', alpha=0.5))  # Outliers en rojo y semi-transparentes
    
    # Añadir la línea de mediana por jornada
    medianas = df.groupby('Jornada',observed=False)['Pasaj'].median().reindex(jornada_order)
    ax.plot(jornada_order, medianas, 'o-', color='blue', markersize=8, label='Mediana',lw=0.5)  # Mediana como bola azul
    
    # Etiquetas y título
    ax.set_ylabel('Pasajeros')
    ax.set_title('Distribución de pasajeros por jornada')
    st.pyplot(fig)

st.write("Autocorrelacion")

# Calcula los valores de autocorrelación
acf_values = acf(df2.Pasaj, nlags=720)

# Grafica la autocorrelación
fig, ax = plt.subplots(figsize=(6.5,2))
ax.plot(acf_values)

# Agrega una línea vertical en el lag 275
ax.axvline(x=286, color='red', linestyle='--')
ax.axvline(x=286*2, color='red', linestyle='--')

ax.set_xlabel('Lags')
ax.set_ylabel('ACF')

st.pyplot(fig)

# Asegúrate de que el índice tenga frecuencia
df2 = df2.asfreq('300s')  # 300 segundos

lags = 300

# Crear el forecaster
params = {
    'n_estimators': 700,
    'max_depth': 11,
    'learning_rate': 0.0551314,
    'reg_alpha': 0.4,
    'reg_lambda': 0.4,
    'random_state': 15926,
    'verbose': -1
}
forecaster = ForecasterAutoreg(
                regressor = XGBRegressor(random_state=15926, verbose=0),
                 #regressor = LGBMRegressor(random_state=15926, verbose=-1), # regresor
                 lags      = lags
             )

# Entrena el forecaster
fecha_fin_val = str(df2.index[val_end])
forecaster.fit(y=df2.loc[:fecha_fin_val, 'Pasaj'])

# Crear variables dummy para la columna 'Jornada'
jornada_dummy = pd.get_dummies(df2['Jornada'], prefix='Jornada')

# Concatenar las variables dummy al DataFrame original
df2 = pd.concat([df2, jornada_dummy], axis=1)

# Reemplazar True por 1 y False por 0 en las columnas Jornada
df2[['Jornada_Madrugada', 'Jornada_Mañana', 'Jornada_Noche', 'Jornada_Tarde']] = \
    df2[['Jornada_Madrugada', 'Jornada_Mañana', 'Jornada_Noche', 'Jornada_Tarde']].astype(int)

# Crear una nueva columna 'Dia_Semana_Fin_Semana'
df2['Dia_Semana_Fin_Semana'] = np.where(df2.index.weekday < 5, 'Dia_Semana', 'Fin_Semana')

# Crear variables dummy para la nueva columna
dia_semana_fin_semana_dummy = pd.get_dummies(df2['Dia_Semana_Fin_Semana'], prefix='Dia')

# Concatenar las variables dummy al DataFrame original
df2 = pd.concat([df2, dia_semana_fin_semana_dummy], axis=1)

# Reemplazar True por 1 y False por 0 en las columnas Dia
df2[['Dia_Dia_Semana', 'Dia_Fin_Semana']] = \
    df2[['Dia_Dia_Semana', 'Dia_Fin_Semana']].astype(int)

#st.dataframe(df2.head())

# Crear un DataFrame de variables exógenas
exog_df = df2[['Vehiculo', 'Kms', 'Tiempo_viaje_s', 'Tiempo_muerto_s','hora', 'Jornada_Madrugada',
               'Jornada_Mañana', 'Jornada_Noche', 'Jornada_Tarde', 'Dia_Dia_Semana','Dia_Fin_Semana']]

metrica, predicciones = backtesting_forecaster(
                            forecaster         = forecaster,
                            y                  = df2['Pasaj'],
                            exog               = exog_df,
                            steps              = lags,
                            metric             = 'mean_absolute_error',
                            initial_train_size = len(df2.loc[:fecha_fin_val]),
                            refit              = False,
                            n_jobs             = 'auto',
                            verbose            = False,
                            show_progress      = True
                        )


# Crear la figura
fig = go.Figure()

# Agregar las trazas para entrenamiento, validación y prueba
trace1 = go.Scatter(x=test.index, y=test['Pasaj'], name="test", mode="lines")
trace2 = go.Scatter(x=predicciones.index, y=predicciones['pred'], name="prediction", mode="lines")
fig.add_trace(trace1)
fig.add_trace(trace2)

# Configurar el layout de la figura
fig.update_layout(
    xaxis_title="Date time",
    yaxis_title="Pasajeros",
    width=850,
    height=400,
    margin=dict(l=20, r=20, t=35, b=20),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1,
        xanchor="left",
        x=0.001,
    )
)

# Mostrar el range slider en el eje X
fig.update_xaxes(rangeslider_visible=True)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

# Mostrar la métrica en un recuadro
st.metric(label="Mean Absolute Error", value=round(metrica, 2))

