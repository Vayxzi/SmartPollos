import sys
import os
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error

# 1. Configurar la ruta para que Python detecte las carpetas en la raíz
directorio_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(directorio_raiz)

# 2. Importar la base de datos desde la carpeta 'data'
from data.database import obtener_datos_completos

print("Extrayendo año de historial desde PostgreSQL...")
df = obtener_datos_completos()

# Feature Engineering: Asegurar que la fecha sea tipo datetime
df['fecha'] = pd.to_datetime(df['fecha'])
df['dia_semana'] = df['fecha'].dt.dayofweek
df['promedio_7_dias'] = df['venta_real'].rolling(window=7).mean()
df['promedio_14_dias'] = df['venta_real'].rolling(window=14).mean()
df.dropna(inplace=True)

# Target: predecir la demanda del día siguiente
df['demanda_siguiente_dia'] = df['venta_real'].shift(-1)
df.dropna(inplace=True)

features = ['dia_semana', 'promedio_7_dias', 'promedio_14_dias']
X = df[features]
y = df['demanda_siguiente_dia']

# Separar datos para medir el MAPE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print("Entrenando modelo XGBoost...")
modelo = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4)
modelo.fit(X_train, y_train)

# Evaluación
predicciones = modelo.predict(X_test)
mape = mean_absolute_percentage_error(y_test, predicciones) * 100
print(f"Precisión del modelo validada. MAPE: {mape:.2f}%")

# 3. Guardar el modelo en la carpeta 'modelos' de la raíz
ruta_modelos = os.path.join(directorio_raiz, "modelos")
os.makedirs(ruta_modelos, exist_ok=True)
ruta_archivo = os.path.join(ruta_modelos, "xgboost_pedidos.json")

modelo.save_model(ruta_archivo)
print(f"Modelo guardado exitosamente en: {ruta_archivo}")