# =============================================================================
# titanic_classification_sklearn.py
# Clasificación de supervivencia en el Titanic con varios algoritmos clásicos.
#
# Requisitos:
#   pip install pandas numpy scikit-learn matplotlib
#
# Asegúrate de tener el fichero "train.csv" de Titanic en la misma carpeta
# que este script, o cambia TITANIC_CSV_PATH a la ruta correcta.
# =============================================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import matplotlib.pyplot as plt


TITANIC_CSV_PATH = "train.csv"  # Cambia esto si el CSV está en otra ruta


def cargar_titanic(path_csv: str) -> pd.DataFrame: 
    """Carga el dataset de Titanic desde un CSV."""
    df = pd.read_csv(path_csv)
    print("Primeras filas del dataset original:")
    print(df.head())
    print("\nInformación del dataset original:")
    print(df.info())
    return df


def preprocesar_titanic(df: pd.DataFrame):
    """
    Preprocesa el dataset de Titanic:
      - Selección de columnas relevantes.
      - Imputación de valores nulos.
      - Codificación de variables categóricas con get_dummies.
    Devuelve X (features) y y (target).
    """
    columnas = [
        "Survived",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
    ]
    df_model = df[columnas].copy()

    # Imputación de valores nulos
    df_model["Age"].fillna(df_model["Age"].median(), inplace=True)
    df_model["Fare"].fillna(df_model["Fare"].median(), inplace=True)
    df_model["Embarked"].fillna(df_model["Embarked"].mode()[0], inplace=True)

    # One-hot encoding de variables categóricas
    df_model = pd.get_dummies(
        df_model,
        columns=["Sex", "Embarked", "Pclass"],
        drop_first=True,
    )

    print("\nColumnas después del preprocesado:")
    print(df_model.columns)

    X = df_model.drop("Survived", axis=1)
    y = df_model["Survived"]

    return X, y


def entrenar_modelos(X_train, y_train):
    """
    Entrena varios modelos de clasificación y devuelve un diccionario
    {nombre_modelo: modelo_entrenado}.
    """
    modelos = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "SVC": SVC(),
    }

    for nombre, modelo in modelos.items():
        print(f"\nEntrenando modelo: {nombre}")
        modelo.fit(X_train, y_train)

    return modelos


def evaluar_modelos(modelos, X_test, y_test) -> pd.DataFrame:
    """
    Evalúa cada modelo sobre el conjunto de test.
    Imprime métricas y devuelve un DataFrame con las accuracies.
    """
    resultados = []

    for nombre, modelo in modelos.items():
        print("\n" + "=" * 80)
        print(f"Modelo: {nombre}")

        y_pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"Accuracy: {acc:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))

        print("Matriz de confusión:")
        print(confusion_matrix(y_test, y_pred))

        resultados.append({"modelo": nombre, "accuracy": acc})

    resultados_df = pd.DataFrame(resultados).sort_values(
        "accuracy", ascending=False
    )
    return resultados_df


def plot_resultados(resultados_df: pd.DataFrame) -> None:
    """Dibuja un gráfico de barras con la accuracy de cada modelo."""
    plt.figure(figsize=(8, 5))
    plt.bar(resultados_df["modelo"], resultados_df["accuracy"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Comparación de modelos en Titanic (Accuracy)")
    for idx, row in resultados_df.iterrows():
        plt.text(
            x=row["modelo"],
            y=row["accuracy"] + 0.01,
            s=f"{row['accuracy']:.2f}",
            ha="center",
        )
    plt.tight_layout()
    plt.show()


def main():
    # 1) Carga de datos
    df = cargar_titanic(TITANIC_CSV_PATH)

    # 2) Preprocesado
    X, y = preprocesar_titanic(df)

    # 3) División train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nTamaños de los conjuntos:")
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    # 4) Entrenamiento de varios modelos
    modelos = entrenar_modelos(X_train, y_train)

    # 5) Evaluación y comparación
    resultados_df = evaluar_modelos(modelos, X_test, y_test)
    print("\nResumen de accuracies:")
    print(resultados_df)

    # 6) Gráfico comparativo
    plot_resultados(resultados_df)


if __name__ == "__main__":
    main()
