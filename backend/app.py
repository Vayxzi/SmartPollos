import sys
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import xgboost as xgb
import pandas as pd

# 1. Configurar la ruta para que Python detecte las carpetas en la raíz (como 'data' y 'modelos')]
directorio_raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(directorio_raiz)

# 2. Ahora sí podrá encontrar la carpeta 'data'
from data.database import obtener_historial_cliente

app = Flask(__name__)
CORS(app)

modelo_xgb = xgb.XGBRegressor()

# 3. Apuntar la carga del modelo usando la ruta absoluta hacia la carpeta 'modelos'
ruta_modelo = os.path.join(directorio_raiz, "modelos", "xgboost_pedidos.json")
modelo_xgb.load_model(ruta_modelo)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"estado": "API de SmartP en línea y lista para predicciones"})

@app.route("/api/predecir", methods=["POST"])
def predecir_pedido():
    datos = request.get_json()
    id_cliente = datos.get("id_cliente")
    sku_producto = datos.get("sku_producto")

    # Extraer historial del cliente desde la BD
    df = obtener_historial_cliente(id_cliente, sku_producto)
    
    # Convertir la columna de texto a formato fecha para evitar el error 'dayofweek'
    df['fecha'] = pd.to_datetime(df['fecha'])
    
    if len(df) < 14:
        return jsonify({"error": "No hay suficientes datos históricos para este cliente (mínimo 14 días)"}), 400

    ultimo_dia = df['fecha'].iloc[-1]
    promedio_7 = float(df['venta_real'].iloc[-7:].mean())
    promedio_14 = float(df['venta_real'].iloc[-14:].mean())
    dia_semana = int((ultimo_dia.dayofweek + 1) % 7)
    
    features = pd.DataFrame({
        'dia_semana': [dia_semana],
        'promedio_7_dias': [promedio_7],
        'promedio_14_dias': [promedio_14]
    })
    
    prediccion = modelo_xgb.predict(features)[0]
    
    return jsonify({
        "cliente": id_cliente,
        "producto": sku_producto,
        "pedido_sugerido": int(prediccion),
        "estado": "Aprobación pendiente"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)