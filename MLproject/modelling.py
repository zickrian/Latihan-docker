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

    file_path = (
        sys.argv[3]
        if len(sys.argv) > 3
        else os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_pca.csv")
    )
    data = pd.read_csv(file_path)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Credit_Score", axis=1),
        data["Credit_Score"],
        random_state=42,
        test_size=0.2,
    )

    input_example = X_train[0:5]

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 35

    # BUANG MLFLOW_RUN_ID kalau ada, biar run baru bersih
    os.environ.pop("MLFLOW_RUN_ID", None)

    # PENTING: simpan objek run jadi variable
    with mlflow.start_run(run_name="rf-credit-score") as run:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
        )
        model.fit(X_train, y_train)

        predicted_qualities = model.predict(X_test)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",   # <-- path yang mau dipakai build-docker
            input_example=input_example,
        )

        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        # TULIS run_id ke file agar bisa dipakai di CI/CD
        run_id = run.info.run_id
        print(f"Training run id: {run_id}")

        # file ini akan muncul di dalam folder MLproject/
        with open("run_id.txt", "w") as f:
            f.write(run_id)
