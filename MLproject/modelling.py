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
    
    # Get the parent directory (repository root)
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Always fix the tracking URI to use absolute paths
    # This ensures that when running under `mlflow run`, which changes to the MLproject directory,
    # we still access the same database as the parent process
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', '')
    if tracking_uri.startswith('sqlite:///') and not tracking_uri.startswith('sqlite:////'):
        db_path = tracking_uri.replace('sqlite:///', '')
        if not os.path.isabs(db_path):
            # The database path is relative, make it absolute from the repository root
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

    # Note: predictions are not used, but keeping for potential future use
    # predicted_qualities = model.predict(X_test)

    # Check if we're running under mlflow run (has MLFLOW_RUN_ID env var)
    mlflow_run_id = os.environ.get('MLFLOW_RUN_ID')
    
    # Create or use a run context for logging
    # When running under `mlflow run`, we use the existing run ID
    # Otherwise, we create a new run with a specific name
    run_name = None if mlflow_run_id else "rf-credit-score"
    with mlflow.start_run(run_id=mlflow_run_id, run_name=run_name) as run:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example,
        )
        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        run_id = run.info.run_id

    print(f"Training run id: {run_id}")
    with open(os.path.join(os.path.dirname(__file__), "run_id.txt"), "w") as f:
        f.write(run_id)
