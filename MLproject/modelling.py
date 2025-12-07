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
    
    # Fix tracking URI if running under mlflow run with relative path
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', '')
    if tracking_uri.startswith('sqlite:///') and not tracking_uri.startswith('sqlite:////'):
        # It's a relative path, make it absolute from the project root
        db_path = tracking_uri.replace('sqlite:///', '')
        if not os.path.isabs(db_path):
            # We're in MLproject directory, go up one level
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            abs_db_path = os.path.join(parent_dir, db_path)
            new_tracking_uri = f'sqlite:///{abs_db_path}'
            mlflow.set_tracking_uri(new_tracking_uri)

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

    # Train the model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    model.fit(X_train, y_train)

    predicted_qualities = model.predict(X_test)

    # SIMPAN MODEL SEBAGAI ARTIFACT "model"
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",   # <--- ini yang dipakai di build-docker
        input_example=input_example,
    )

    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Get the current run ID (works both with mlflow run and standalone)
    run = mlflow.active_run()
    if run:
        run_id = run.info.run_id
    else:
        # If no active run, get it from the tracking client
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(
            experiment_ids=["0"],
            max_results=1,
            order_by=["attributes.start_time DESC"],
        )
        run_id = runs[0].info.run_id if runs else "unknown"
    
    print(f"Training run id: {run_id}")
    with open(os.path.join(os.path.dirname(__file__), "run_id.txt"), "w") as f:
        f.write(run_id)
