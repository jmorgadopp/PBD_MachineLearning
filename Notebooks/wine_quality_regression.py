# =============================================================================
# wine_quality_regression.py
# Regresión de la calidad del vino utilizando varios algoritmos clásicos.
#
# Requisitos: 
#   pip install pandas numpy scikit-learn matplotlib
#
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Ruta del archivo CSV
WINE_CSV_PATH = "../DataSets/winequality-red.csv"  # Cambia esto si el CSV está en otra ruta

def cargar_vino(path_csv: str) -> pd.DataFrame: 
    """Carga el dataset de calidad del vino desde un CSV."""
    df = pd.read_csv(path_csv, delimiter=",")  # Usando coma como delimitador
    print("Primeras filas del dataset original:")
    print(df.head())
    print("\nInformación del dataset original:")
    print(df.info())
    return df

def preprocesar_vino(df: pd.DataFrame):
    """
    Preprocesa el dataset de calidad del vino:
      - Selección de columnas relevantes.
      - Imputación de valores nulos.
      - Escalado de características.
    Devuelve X (features) y y (target).
    """
    df_model = df.copy()

    # Imputación de valores nulos (si los hubiera)
    df_model.fillna(df_model.median(), inplace=True)

    # Dividir en características (X) y target (y), donde "quality" es la etiqueta
    X = df_model.drop("quality", axis=1)
    y = df_model["quality"]

    # Escalado de características (recomendado para modelos como SVC y KNN)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def entrenar_modelos(X_train, y_train):
    """
    Entrena varios modelos de regresión y devuelve un diccionario
    {nombre_modelo: modelo_entrenado}.
    """
    modelos = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor(),
    }

    for nombre, modelo in modelos.items():
        print(f"\nEntrenando modelo: {nombre}")
        modelo.fit(X_train, y_train)

    return modelos

def evaluar_modelos(modelos, X_test, y_test) -> pd.DataFrame:
    """
    Evalúa cada modelo sobre el conjunto de test.
    Imprime métricas y devuelve un DataFrame con los resultados.
    """
    resultados = []

    for nombre, modelo in modelos.items():
        print("\n" + "=" * 80)
        print(f"Modelo: {nombre}")

        y_pred = modelo.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        print(f"R2:   {r2:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")

        resultados.append({"modelo": nombre, "R2": r2, "MAE": mae, "RMSE": rmse})

    resultados_df = pd.DataFrame(resultados).sort_values(
        "R2", ascending=False
    )
    return resultados_df

def plot_resultados(resultados_df: pd.DataFrame) -> None:
    """Dibuja un gráfico de barras con el R² de cada modelo."""
    plt.figure(figsize=(10, 6))
    plt.bar(resultados_df["modelo"], resultados_df["R2"], color='skyblue')
    plt.ylim(0, 1)
    plt.ylabel("R²")
    plt.title("Comparación de modelos de regresión (R²)")
    for idx, row in resultados_df.iterrows():
        plt.text(
            x=row["modelo"],
            y=row["R2"] + 0.02,
            s=f"{row['R2']:.2f}",
            ha="center",
        )
    plt.tight_layout()
    plt.show()

def main():
    # 1) Cargar datos
    df = cargar_vino(WINE_CSV_PATH)

    # 2) Preprocesado
    X, y = preprocesar_vino(df)

    # 3) División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,  # Para mantener la proporción de clases
    )

    print("\nTamaños de los conjuntos:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    # 4) Entrenamiento de varios modelos
    modelos = entrenar_modelos(X_train, y_train)

    # 5) Evaluación y comparación
    resultados_df = evaluar_modelos(modelos, X_test, y_test)
    print("\nResumen de R² y métricas:")
    print(resultados_df)

    # 6) Gráfico de resultados
    plot_resultados(resultados_df)

if __name__ == "__main__":
    main()
