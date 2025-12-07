import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import numpy as np
import warnings
import sys

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # ⬇️ FIX PENTING: buang MLFLOW_RUN_ID dari environment
    os.environ.pop("MLFLOW_RUN_ID", None)

    # Ambil path CSV dari argumen ke-3 atau default train_pca.csv di folder yang sama
    file_path = (
        sys.argv[3]
        if len(sys.argv) > 3
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_pca.csv")
    )
    data = pd.read_csv(file_path)

    # Target kolom Credit_Score
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Credit_Score", axis=1),
        data["Credit_Score"],
        random_state=42,
        test_size=0.2,
    )

    input_example = X_train[0:5]

    # Hyperparameter dari argumen CLI
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 35

    # ⬇️ Mulai run MLflow baru (sekarang aman)
    with mlflow.start_run(run_name="rf-credit-score"):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
        )
        model.fit(X_train, y_train)

        predicted_qualities = model.predict(X_test)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",   # warning boleh diabaikan buat sekarang
            input_example=input_example,
        )

        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
