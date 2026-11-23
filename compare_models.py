
import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

ROOT = Path(__file__).resolve().parent

TRAIN_PATH = ROOT / "results" / "data" / "train_labeled.csv"

FEATURES = [
    "mean_pairwise_dist",
    "std_coords",
    "mean_centroid_dist",
    "mean_nn_dist"
]

TARGET = "best_algorithm"


def optimize_random_forest(X, y):

    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )

    parameters = {
        "n_estimators": [200, 300, 500],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", None]
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    print("\nOptimizando Random Forest Balanced...")

    search.fit(X, y)

    print("\nMejores parametros:")
    print(search.best_params_)

    print("\nMejor Macro F1:")
    print(search.best_score_)

    return search.best_estimator_



def main():

    train = pd.read_csv(TRAIN_PATH)

    X = train[FEATURES]
    y = train[TARGET]

    best_model = optimize_random_forest(X, y)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42
        ),

        "Random Forest Balanced": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
            random_state=42
        ),

        "Decision Tree": DecisionTreeClassifier(
            random_state=42
        ),

        "SVM": make_pipeline(
            StandardScaler(),
            SVC(class_weight="balanced")
        )
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro"
    }

    results = []

    print("Comparacion de modelos")
    print("=" * 60)

    for name, model in models.items():

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring
        )

        accuracy = scores["test_accuracy"].mean()
        f1 = scores["test_f1_macro"].mean()

        results.append({
            "Modelo": name,
            "Accuracy": accuracy,
            "Macro F1": f1
        })

        print(f"\nModelo: {name}")
        print(f"Accuracy promedio: {accuracy:.4f}")
        print(f"Macro F1 promedio: {f1:.4f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="Macro F1",
        ascending=False
    )

    print("\n========== RESUMEN ==========")
    print(results_df.to_string(index=False))

    output_path = ROOT / "results" / "data" / "model_comparison.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nResultados guardados en: {output_path}")


if __name__ == "__main__":
    main()
