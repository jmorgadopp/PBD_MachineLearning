# =============================================================================
# diabetes_tpot_regression.py  (versión sin TPOT, 100% scikit-learn)
#
# Objetivo:
#   - Ejemplo de "AutoML" clásico usando solo scikit-learn:
#       * probamos varios modelos de regresión
#       * hacemos RandomizedSearchCV para buscar buenos hiperparámetros
#       * elegimos el mejor modelo según R²
#
# Ventaja:
#   - No depende de TPOT ni Dask, así que funciona bien en tu entorno actual
#     con Python 3.12.
# =============================================================================

import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def cargar_diabetes():
    """Carga el dataset de diabetes de sklearn como DataFrame."""
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="disease_progression")

    print("Primeras filas de X:")
    print(X.head())
    print("\nDescripción de y:")
    print(y.describe())

    return X, y


def evaluar_modelo(nombre, modelo, X_train, X_test, y_train, y_test):
    """Entrena un modelo y calcula métricas en test."""
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n=== Resultados {nombre} ===")
    print(f"R2:   {r2:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return {"modelo": nombre, "R2": r2, "MAE": mae, "RMSE": rmse}


def main():
    # 1) Cargar datos
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

    # 3) Definir modelos base
    modelos_base = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
        "SVR": SVR(),
    }

    # 4) Espacios de búsqueda de hiperparámetros
    espacios_param = {
        "RandomForest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 3, 5, 7],
            "min_samples_split": [2, 4, 6],
        },
        "GradientBoosting": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4],
        },
        "SVR": {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto"],
            "kernel": ["rbf", "linear"],
        },
    }

    resultados = []
    mejores_modelos = {}

    # 5) Búsqueda aleatoria de hiperparámetros para cada modelo
    for nombre, modelo in modelos_base.items():
        print("\n" + "=" * 80)
        print(f"Buscando mejores hiperparámetros para: {nombre}")

        param_dist = espacios_param[nombre]

        buscador = RandomizedSearchCV(
            estimator=modelo,
            param_distributions=param_dist,
            n_iter=10,             # puedes subir a 20 si quieres más búsqueda
            scoring="r2",
            cv=5,
            random_state=42,
            n_jobs=-1,
        )

        buscador.fit(X_train, y_train)

        print("Mejores hiperparámetros encontrados:")
        print(buscador.best_params_)
        print(f"Mejor R2 CV: {buscador.best_score_:.4f}")

        mejor_modelo = buscador.best_estimator_
        mejores_modelos[nombre] = mejor_modelo

        # Evaluación final en el conjunto de test
        res = evaluar_modelo(
            nombre + " (mejor encontrado)",
            mejor_modelo,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        resultados.append(res)

    # 6) Resumen y mejor modelo global según R2
    resultados_df = pd.DataFrame(resultados).sort_values("R2", ascending=False)

    print("\n" + "=" * 80)
    print("Resumen de modelos ordenados por R2 en test:")
    print(resultados_df)

    mejor_nombre = resultados_df.iloc[0]["modelo"]
    print(f"\nMejor modelo global según R2 en test: {mejor_nombre}")


if __name__ == "__main__":
    main()
