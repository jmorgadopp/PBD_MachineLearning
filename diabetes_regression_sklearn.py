# =============================================================================
# diabetes_regression_sklearn.py
# Ejemplo de regresión utilizando el dataset de diabetes de scikit-learn.
#
# Requisitos:
#   pip install pandas numpy scikit-learn matplotlib
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def cargar_diabetes():
    """Carga el dataset de diabetes de sklearn y lo devuelve como X, y."""
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="disease_progression")

    print("Primeras filas de X:")
    print(X.head())
    print("\nDescripción de y:")
    print(y.describe())

    return X, y


def entrenar_modelos_regresion(X_train, y_train):
    """Entrena varios modelos de regresión y devuelve un diccionario."""
    modelos = {
        "LinearRegression": LinearRegression(),
        "KNNRegressor": KNeighborsRegressor(),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
        "RandomForestRegressor": RandomForestRegressor(random_state=42),
    }

    for nombre, modelo in modelos.items():
        print(f"\nEntrenando modelo: {nombre}")
        modelo.fit(X_train, y_train)

    return modelos


def evaluar_modelos_regresion(modelos, X_test, y_test) -> pd.DataFrame:
    """
    Evalúa cada modelo de regresión sobre el conjunto de test.
    Calcula MAE, RMSE y R2 y devuelve un DataFrame resumen.
    """
    resultados = []

    for nombre, modelo in modelos.items():
        print("\n" + "=" * 80)
        print(f"Modelo: {nombre}")
        y_pred = modelo.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"MAE:  {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"R2:   {r2:.4f}")

        resultados.append(
            {
                "modelo": nombre,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )

    resultados_df = pd.DataFrame(resultados).sort_values(
        "R2", ascending=False
    )
    return resultados_df


def plot_pred_vs_real(y_test, y_pred, titulo="Predicción vs Real"):
    """Dibuja un scatter plot de valores reales vs predichos."""
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.6)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.xlabel("Valor real")
    plt.ylabel("Valor predicho")
    plt.title(titulo)
    plt.tight_layout()
    plt.show()


def main():
    # 1) Cargar dataset
    X, y = cargar_diabetes()

    # 2) División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    print("\nTamaños de los conjuntos:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    # 3) Entrenar varios modelos de regresión
    modelos = entrenar_modelos_regresion(X_train, y_train)

    # 4) Evaluación
    resultados_df = evaluar_modelos_regresion(modelos, X_test, y_test)
    print("\nResumen de métricas de regresión:")
    print(resultados_df)

    # 5) Elegir el mejor modelo (por R2) y dibujar pred vs real
    mejor_nombre = resultados_df.iloc[0]["modelo"]
    print(f"\nMejor modelo según R2: {mejor_nombre}")
    mejor_modelo = modelos[mejor_nombre]

    y_pred_mejor = mejor_modelo.predict(X_test)
    plot_pred_vs_real(
        y_test,
        y_pred_mejor,
        titulo=f"Predicción vs Real ({mejor_nombre})",
    )


if __name__ == "__main__":
    main()
