import pandas as pd
from sqlalchemy import create_engine

# Cambia estos datos cuando el ingeniero te pase el backup de la base de datos
DB_USER = "postgres"
DB_PASS = "Admin123!"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "postgres"

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

def obtener_datos_completos():
    """Extrae el año completo de ventas para entrenar el modelo."""
    # Ajusta el nombre de la tabla según lo que entregue el ingeniero
    query = """
        SELECT fecha, id_cliente, sku_producto, venta_real, quiebre_stock 
        FROM ventas_historicas 
        ORDER BY fecha ASC
    """
    df = pd.read_sql(query, engine)
    
    # Limpieza básica: rellenar días sin stock con la última venta válida
    df.loc[df['quiebre_stock'] == 1, 'venta_real'] = None
    df['venta_real'] = df['venta_real'].ffill()
    return df

def obtener_historial_cliente(id_cliente: str, sku_producto: str):
    """Extrae datos recientes de un cliente específico para la predicción diaria."""
    query = f"""
        SELECT fecha, venta_real 
        FROM ventas_historicas 
        WHERE id_cliente = '{id_cliente}' AND sku_producto = '{sku_producto}'
        ORDER BY fecha ASC
    """
    return pd.read_sql(query, engine)