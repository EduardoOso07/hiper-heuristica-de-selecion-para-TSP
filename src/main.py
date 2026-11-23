
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.dummy import DummyClassifier

# Directorio principal del proyecto
ROOT = Path(__file__).resolve().parent

# Rutas de los archivos
TRAIN_PATH = ROOT / "results" / "data" / "train_labeled.csv"
TEST_PATH = ROOT / "results" / "data" / "test_labeled.csv"

# Caracteristicas utilizadas para entrenar
FEATURES = [
    "mean_pairwise_dist",
    "std_coords",
    "mean_centroid_dist",
    "mean_nn_dist"
]

TARGET = "best_algorithm"


def main():

    # 1. Cargar los conjuntos de datos
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    print("Datos cargados correctamente.")
    print("Entrenamiento:", len(train))
    print("Prueba:", len(test))

    # 2. Comprobar que existan las columnas necesarias
    for column in FEATURES + [TARGET]:
        if column not in train.columns or column not in test.columns:
            raise ValueError(
                f"No se encontro la columna necesaria: {column}"
            )

    # 3. Separar caracteristicas y etiquetas
    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    # 4. Verificar valores faltantes
    if X_train.isnull().any().any() or X_test.isnull().any().any():
        raise ValueError("Existen valores faltantes en las caracteristicas.")

    # 5. Crear el clasificador
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        max_features='sqrt'
        
    )

    # 6. Entrenar el modelo
    print("\nEntrenando Random Forest...")
    model.fit(X_train, y_train)

    # 7. Realizar predicciones
    predictions = model.predict(X_test)

    # 8. Evaluar el modelo
    accuracy = accuracy_score(y_test, predictions)

    print("\n========== RESULTADOS ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nReporte de clasificacion:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nMatriz de confusion:")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=model.classes_
        )
    )

    # 9. Importancia de las caracteristicas
    importance = pd.DataFrame({
        "Caracteristica": FEATURES,
        "Importancia": model.feature_importances_
    }).sort_values(
        by="Importancia",
        ascending=False
    )

    print("\nImportancia de las caracteristicas:")
    print(importance.to_string(index=False))

    # 10. Guardar predicciones
    results = test.copy()
    results["predicted_algorithm"] = predictions

    output_path = ROOT / "results" / "data" / "predictions.csv"
    results.to_csv(output_path, index=False)

    print(f"\nPredicciones guardadas en: {output_path}")
    
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    baseline_predictions = baseline.predict(X_test)

    print("\n========== BASELINE ==========")
    print(
        "Accuracy:",
        accuracy_score(y_test, baseline_predictions)
    )

    print(
        "Macro F1:",
        classification_report(
            y_test,
            baseline_predictions,
            output_dict=True,
            zero_division=0
        )["macro avg"]["f1-score"]
    )

if __name__ == "__main__":
    main()
