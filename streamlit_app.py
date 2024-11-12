#librerias
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
# Modelado y Forecasting
# import skforecast
from statsmodels.tsa.stattools import acf
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

# Definir una paleta de colores personalizada basada en la imagen
cootracovi_palette = ["#1E90FF",  # Azul del fondo (tranquilidad y profesionalismo)
                      "#4CAF50",  # Verde del logo (sostenibilidad y eco-conducción)
                      "#FFD700",  # Amarillo o beige para cercanía y calidez
                      "#FFA07A"]  # Naranja suave para dinamismo y energía

# Establecer la paleta en Seaborn
sns.set_palette(cootracovi_palette)

#from skforecast.model_selection import backtesting_forecaster

# Borrar la caché en cada ejecución
#st.cache_data.clear()

#python -m pip install {package_name}

#sns.set_theme(style="whitegrid", palette="pastel")

st.title(":bus: Cootracovi ")
st.write(
    "Modelo de prediccion de forecast para predecir parametros de la empresa trasnportadora de las rutas: HONDA - ORIENTAL - HONDA"
)

# Cargar datos desde un enlace de GitHub
st.subheader("Cargar Datos desde un Link de GitHub")
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
st.dataframe(df) # Muestra las primeras filas del DataFrame

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
st.write(f"El tamaño del dataset despues de el preprocesamiento es de: {df.shape[0]} filas y {df.shape[1]} columnas:")
st.write(", ".join(df.columns))

st.subheader('Analisis de las variables')
opciones_columnas = [
        'Pasaj', 
        'Kms', 
        'Tiempo_viaje_s', 
        'Tiempo_muerto_s', 
        'Vehiculo']

    # Selección de columna con 'tipo_negocio' como predeterminado
columna_seleccionada = st.selectbox(
        "Selecciona la columna para graficar:", 
        opciones_columnas, 
        index=opciones_columnas.index('Pasaj'),
        key='columna_seleccionada')


# Cálculos
valor_medio = round(df[columna_seleccionada].mean(), 2)
sesgo = round(df[columna_seleccionada].skew(), 2)
percentil_25 = df[columna_seleccionada].quantile(0.25)
percentil_75 = df[columna_seleccionada].quantile(0.75)
iqr = percentil_75 - percentil_25

# Cálculo de valores atípicos
outliers = df[(df[columna_seleccionada] < (percentil_25 - 1.5 * iqr)) | 
              (df[columna_seleccionada] > (percentil_75 + 1.5 * iqr))]
porcentaje_atipicos = round((len(outliers) / len(df)) * 100, 2)

st.markdown("<h5>Valores estadisticos</h5>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"Valor medio: {valor_medio}")
with col2:
    st.write(f"Sesgo: {sesgo}")
with col3:
    st.write(f"% de valores atípicos: {porcentaje_atipicos}%")

df['dia'] = df.index.day_name() 

st.markdown("<h5>Distribuciones</h5>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

# Opciones de columnas para graficar

if columna_seleccionada =='Vehiculo':
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[columna_seleccionada], kde=True, ax=ax)
    ax.set_title(f'Distribución de {columna_seleccionada.replace("_", " ").capitalize()}', fontsize=18)
    ax.set_xlabel(columna_seleccionada, fontsize=16)  # Etiqueta del eje x más grande
    ax.set_ylabel('Frecuencia', fontsize=16)
    # Ajustar el tamaño de los ticks
    ax.tick_params(axis='both', labelsize=14)
    # Mostrar gráfico en Streamlit
    st.pyplot(fig)

    # Crear una tabla pivote que cuente los viajes por cada vehículo y día de la semana
    pivot_table = df.pivot_table(index='dia', columns='Vehiculo', aggfunc='size', fill_value=0)

    # Configuración del mapa de calor
    plt.figure(figsize=(18, 6))
    heatmap = sns.heatmap(pivot_table, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label': 'Cantidad de viajes'})

    # Etiquetas y título
    plt.ylabel('Día de la semana',fontsize=16)
    plt.xlabel('Vehículo',fontsize=16)
    plt.title('Cantidad de viajes por vehículo y día de la semana', fontsize=18)
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=14, rotation=0)
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14, rotation=0)
    # Mostrar el heatmap en Streamlit
    st.pyplot(plt)

else:
    with col1:    

        fig, ax = plt.subplots()
        sns.histplot(df[columna_seleccionada], kde=True, ax=ax)
        ax.set_title(f'Distribución de {columna_seleccionada.replace("_", " ").capitalize()}', fontsize=18)
        ax.set_xlabel(columna_seleccionada, fontsize=16)  # Etiqueta del eje x más grande
        ax.set_ylabel('Frecuencia', fontsize=16)
        # Ajustar el tamaño de los ticks
        ax.tick_params(axis='both', labelsize=14)
        # Mostrar gráfico en Streamlit
        st.pyplot(fig)

    with col2:

        fig, ax = plt.subplots()
        sns.boxplot(x=df[columna_seleccionada], ax=ax)
        ax.set_title(f'Distribución de {columna_seleccionada.replace("_", " ").capitalize()}', fontsize=18)
        ax.set_xlabel(columna_seleccionada, fontsize=16)  # Etiqueta del eje x más grande
        # Ajustar el tamaño de los ticks
        ax.tick_params(axis='both', labelsize=14)

        # Mostrar gráfico en Streamlit
        st.pyplot(fig)


# Cambiar la frecuencia a 5 minutos ('5T') y rellenar valores faltantes con bfill
df2 = df.asfreq(freq='5T', method='bfill')

df2 = df2.rename(columns={'Fecha_Hora_Salida': 'Fecha_Hora'})

if columna_seleccionada != 'Vehiculo':
    st.markdown("<h5>Gráficos de estacionalidad</h5>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Agregar contenido en la primera columna
    with col1:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        df['dia'] = df.index.day_name()
        medianas = df.groupby('dia')[columna_seleccionada].median()
        sns.boxplot(df, x='dia',y=columna_seleccionada, ax=ax, order=medianas.index)
        medianas.plot(style='o-',color="cyan", markersize=8, label='Mediana',lw=0.5, ax=ax)
        ax.set_ylabel(columna_seleccionada, fontsize=16)
        ax.set_xlabel('dia', fontsize=16) 
        # Ajustar el tamaño de los ticks
        ax.tick_params(axis='both', labelsize=14)
        st.pyplot(fig)

        # Agregar contenido en la segunda columna
    with col2:
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        # Orden para las jornadas
        jornada_order = ['Madrugada', 'Mañana', 'Tarde', 'Noche']

        # Crear el boxplot con transparencia
        sns.boxplot(x='Jornada', y=columna_seleccionada, data=df, ax=ax, order=jornada_order)  # Outliers en rojo y semi-transparentes

        # Añadir la línea de mediana por jornada
        medianas = df.groupby('Jornada',observed=False)[columna_seleccionada].median().reindex(jornada_order)
        ax.plot(jornada_order, medianas, 'o-', color="cyan", markersize=8, label='Mediana',lw=0.5)  # Mediana como bola azul

        # Etiquetas y título
        ax.set_ylabel(columna_seleccionada, fontsize=16)
        ax.set_xlabel('jornada', fontsize=16) 
        # Ajustar el tamaño de los ticks
        ax.tick_params(axis='both', labelsize=14)
        st.pyplot(fig)


    fig, ax = plt.subplots(figsize=(8.5, 3))
    df2['hora'] = df2.index.hour
    medianas = df2.groupby('hora')[columna_seleccionada].median()
    sns.boxplot(df2, x='hora',y=columna_seleccionada, ax=ax, order=medianas.index)
    ax.plot(medianas.index, medianas.values, 'o-', color="cyan", markersize=8, label='Mediana', lw=0.5)
    ax.set_ylabel(columna_seleccionada, fontsize=12)
    ax.set_xlabel('hora', fontsize=12) 
    # Ajustar el tamaño de los ticks
    ax.tick_params(axis='both', labelsize=10)
    st.pyplot(fig)


# Separación datos train-val-test 70% 15% 15%

train_size = 0.7  # 70% para entrenamiento
val_size = 0.30   # 15% para validación
test_size = 0  # 15% para prueba

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

if columna_seleccionada != 'Vehiculo':
    st.markdown("<h5>Series de tiempo</h5>", unsafe_allow_html=True)
    # Crear la figura
    fig = go.Figure()

    # Agregar las trazas para entrenamiento, validación y prueba
    fig.add_trace(go.Scatter(x=df.index, y=df[columna_seleccionada], mode='lines', name='Train'))
    #fig.add_trace(go.Scatter(x=val.index, y=val[columna_seleccionada], mode='lines', name='Validation'))
    #fig.add_trace(go.Scatter(x=test.index, y=test[columna_seleccionada], mode='lines', name='Test'))

    # Configurar el layout de la figura
    fig.update_layout(
        xaxis_title="Fecha",
        yaxis_title=columna_seleccionada,
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


st.subheader('Análisis de la periodicidad del dataset')
# Análisis de la periodicidad del dataset
df['df_time_diffs'] = df.index.to_series().diff().dt.total_seconds()

# Obtener los valores mínimo y máximo de la columna 'df_time_diffs'
min_val = df['df_time_diffs'].min()
max_val = df['df_time_diffs'].max()

# Configurar los límites del zoom mediante un slider en Streamlit
zoom_min, zoom_max = st.slider(
    'Selecciona el rango para hacer zoom en el eje X',
    min_value=float(min_val), max_value=float(max_val),
    value=(float(min_val), float(max_val / 4))
)

# Configurar el tamaño de la figura
fig, ax = plt.subplots(figsize=(6.5, 2))

# Crear el histograma con KDE
sns.histplot(df['df_time_diffs'].dropna(), kde=True, ax=ax)

# Calcular la mediana de las diferencias de tiempo
mediana_dif = df['df_time_diffs'].median()
# Convertir la mediana a minutos
mediana_minutos = mediana_dif / 60

# Configurar los límites del eje X basados en el slider
ax.set_xlim(zoom_min, zoom_max)

# Asignar nombres a los ejes y el título
ax.set_xlabel('Diferencia entre observaciones (segundos)')
ax.set_ylabel('Frecuencia')

# Agregar una línea vertical en la mediana
ax.axvline(mediana_dif, color='r', linestyle='--', label='Mediana: {:.2f} s ({:.2f} min)'.format(mediana_dif, mediana_minutos))
ax.legend()

# Mostrar el gráfico en Streamlit
st.pyplot(fig)

# Mensaje sobre la mediana
st.write(f"La frecuencia mediana es de {mediana_dif:.2f} segundos, que son {mediana_minutos:.2f} minutos. La vamos a tomar como 5 minutos.")

st.subheader('Modelos de Machine Learning')

# Crear un selector para elegir la columna
columna_modelo = st.selectbox(
    "Selecciona la columna para el modelo:",
    ['Pasaj', 'Tiempo_viaje_s', 'Tiempo_muerto_s'],
    index=0,
    key='columna_modelo_seleccion'
)



#decision tree
st.write('##### DecisionTreeRegressor')
st.write('El modelo de Árbol de Decisión es un clasificador supervisado que utiliza una estructura de árbol para tomar decisiones basadas en reglas de decisión derivadas de los datos de entrenamiento.')


# Separar variables predictoras (X) y variable objetivo (y)
X = df[['Dia', 'Hora']]
y = df[columna_modelo]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el árbol de decisión con criterio 'mse'
model = DecisionTreeRegressor(criterion='absolute_error', max_depth=10, min_samples_split=15, random_state=42)
model.fit(X_train, y_train)

# Predecir los valores para el conjunto de prueba
y_pred = model.predict(X_test)

# Evaluar el modelo
mse =mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Lógica para formatear el MAE
if columna_modelo == 'Tiempo_viaje_s' or columna_modelo == 'Tiempo_muerto_s':
    hours, remainder = divmod(mse, 3600)  # Obtener horas y el resto
    minutes, seconds = divmod(remainder, 60)  # Obtener minutos y segundos
        
    mensaje = (
        f"Para este modelo en promedio, la diferencia entre las predicciones del modelo y los valores reales es de {round(mse, 1)} segundos. En h, min y s:  {int(hours)}:{int(minutes)}:{seconds:.0f}"
        )
else:
    mensaje = (
    f"Para este modelo en promedio, la diferencia entre las predicciones del modelo y los valores reales es de {round(mse)} pasajeros por observación"
    )

st.write(mensaje)

# Selección del día de la semana
dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
dia_seleccionado = st.selectbox('Selecciona el día de la semana:', dias_semana)

# Mapeo del día seleccionado al número correspondiente
dia_num = dias_semana.index(dia_seleccionado)

# Selección de la hora
hora_seleccionada = st.slider('Selecciona la hora del día:', min_value=df2['Hora'].min(), max_value=df2['Hora'].max(), value=12)


# Botón para predecir
if st.button('Predecir'):
    # Realizar la predicción
    prediccion = model.predict([[dia_num, hora_seleccionada]])
    if columna_modelo == 'Tiempo_viaje_s' or columna_modelo == 'Tiempo_muerto_s':
        hours, remainder = divmod(prediccion[0], 3600)  # Obtener horas y el resto
        minutes, seconds = divmod(remainder, 60)  # Obtener minutos y segundos
        
        mensaje = (
            f"La cantidad predicha de {columna_modelo} es: {int(prediccion[0])} segundos. En h, min y s:  {int(hours)}:{int(minutes)}:{seconds:.0f}"
            )
    else:
        mensaje = (
        f"La cantidad predicha de {columna_modelo} es: {int(prediccion[0])} pasajeros"
        )
    
    st.success(mensaje)


st.write('##### Forecasting')
#Variables exogenas 

st.write('Los modelos de pronóstico (forecasting) se utilizan para predecir valores futuros de una variable en función de patrones y tendencias observados en datos históricos.')

# Crear un DataFrame de variables exógenas
df2['hora'] = df2.index.hour
exog_df = df2[['hora']]

# Extraer el día de la semana del índice 
exog_df['dia_semana'] = exog_df.index.dayofweek  # Lunes = 0, Domingo = 6



# Asegúrate de que el índice tenga frecuencia
df2 = df2.asfreq('300s')  # 300 segundos
exog_df = exog_df.asfreq('300s')
train = train.asfreq('300s')
val = val.asfreq('300s')


lags = 300

# Crear el forecaster con los parámetros especificados

parametros = {'Pasaj':{
    'n_estimators': 550,
    'max_depth': 11,
    'learning_rate':  0.05513142057308684,
    'reg_alpha': 0.4,
    'reg_lambda': 0.4,
    'verbosity': 0,
    'tree_method': 'hist',
    'max_bin': 80},
    'Tiempo_viaje_s': {
    'n_estimators': 150,
    'max_depth': 7, 
    'learning_rate': 0.44680119239545024,
    'reg_alpha': 1, 
    'reg_lambda': 0.5,
    'verbosity': 0,
    'tree_method': 'hist',
    'max_bin': 130},
    'Tiempo_muerto_s':{
    'n_estimators': 450, 
    'max_depth': 5, 
    'learning_rate': 0.3655775100531092, 
    'reg_alpha': 0.5, 
    'reg_lambda': 0.3,
    'verbosity': 0,
    'tree_method': 'hist',
    'max_bin': 128}
              }

params = parametros[columna_modelo]

# Inicializar el regressor con los parámetros
regressor = XGBRegressor(**params,n_jobs=-1)

# Crear el forecaster
forecaster = ForecasterAutoreg(
    regressor=regressor,
    lags=lags
)


# Entrenar el forecaster con la serie temporal y las variables exógenas
forecaster.fit(y=train[columna_modelo],
               exog=exog_df.loc[train.index])

# Selector de fecha con límites
fecha_fin_input = st.date_input(
    f"Selecciona la fecha de finalización para la predicción (posterior a {train.index[-1]}):",
    min_value=train.index[-1] + timedelta(days=1),  # Un día después del último valor en train
    max_value=train.index[-1] + timedelta(days=30)  # Máximo un mes después
)

# Añadir un selector de hora para la predicción
hora_fin_input = st.time_input("Selecciona la hora de finalización para la predicción:")

# Crear la fecha y hora de finalización
fecha_fin = pd.Timestamp.combine(fecha_fin_input, hora_fin_input)

# Obtener la última fecha de validación
ultima_fecha_train = train.index[-1] + pd.Timedelta(minutes=5)  # Sumar 5 minutos

# Calcular la cantidad de pasos a predecir
pasos_a_predecir = (fecha_fin - ultima_fecha_train).total_seconds() // 300  # 300 segundos = 5 minutos


# Generar las fechas futuras
fechas_futuras = pd.date_range(start=ultima_fecha_train, periods=int(pasos_a_predecir), freq='5T')

# Crear las variables exógenas
exog_futuro = pd.DataFrame({
    'hora': fechas_futuras.hour,
    'dia_semana': fechas_futuras.dayofweek
}, index=fechas_futuras)

y_pred_futuro = forecaster.predict(steps=int(pasos_a_predecir), exog=exog_futuro)

fig = go.Figure()

# Trazas para los datos de validación y las predicciones futuras
fig.add_trace(go.Scatter(x=val.index, y=val[columna_modelo], name="Validación", mode="lines"))
fig.add_trace(go.Scatter(x=fechas_futuras, y=round(y_pred_futuro,0), name="Predicción Futuro", mode="lines"))

# Configurar el layout
fig.update_layout(
    xaxis_title="Fecha y hora",
    yaxis_title=columna_modelo,
    width=850,
    height=400,
    margin=dict(l=20, r=20, t=35, b=20),
    legend=dict(orientation="h", yanchor="top", y=1, xanchor="left", x=0.001)
)

# Mostrar el range slider en el eje X
fig.update_xaxes(rangeslider_visible=True)

# Mostrar el gráfico en Streamlit
st.plotly_chart(fig)

# Calcular el error absoluto medio (MAE) entre la columna 'Pasaj' y las predicciones 'y_pred'
mae = mean_absolute_error(val[columna_modelo], y_pred_futuro.loc[val.index])


#st.write(f"Error absoluto medio (MAE): {round(mae,1)}")

# Lógica para formatear el MAE
if columna_modelo == 'Tiempo_viaje_s' or columna_modelo == 'Tiempo_muerto_s':
    minutes, seconds = divmod(mae, 60)  # Obtener minutos y segundos
    hours, minutes = divmod(minutes, 60)  # Obtener horas y minutos
    
    mensaje = (
        f"Para este modelo en promedio, la diferencia entre las predicciones del modelo y los valores reales es de {round(mae, 1)} segundos. En h, min y s:  {int(hours)}:{int(minutes)}:{seconds:.0f}"
        )
else:
    mensaje = (
    f"Para este modelo en promedio, la diferencia entre las predicciones del modelo y los valores reales es de {round(mae, 0)} pasajeros por observación"
    )

st.write(mensaje)

st.markdown("<h3>Referencias</h3>", unsafe_allow_html=True)  # Usando HTML para subtítulo
referencias_html = """
<ol>
    <li> Notebook con EDA.  <a href="https://colab.research.google.com/drive/1ngMe6wLYAksJzvvYI0VQEnUUjgh-cdz4?usp=sharing">link</a></li>
    <li> Keller, C., Glück, F., Gerlach, C. F., & Schlegel, T. (2022). Investigating the potential of data science methods for sustainable public transport. Sustainability, 14(7), 4211. <a href="https://www.mdpi.com/2071-1050/14/7/4211">link</a> .</li>
</ol>
"""

st.markdown(referencias_html, unsafe_allow_html=True)
