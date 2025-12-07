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
    
    # Debug: print environment and paths
    print(f"Current directory: {os.getcwd()}")
    print(f"Script location: {os.path.abspath(__file__)}")
    print(f"MLFLOW_RUN_ID: {os.environ.get('MLFLOW_RUN_ID', 'NOT SET')}")
    print(f"MLFLOW_TRACKING_URI: {os.environ.get('MLFLOW_TRACKING_URI', 'NOT SET')}")
    
    # Get the parent directory (repository root)
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Always fix the tracking URI to use absolute paths
    tracking_uri = os.environ.get('MLFLOW_TRACKING_URI', '')
    if tracking_uri.startswith('sqlite:///') and not tracking_uri.startswith('sqlite:////'):
        db_path = tracking_uri.replace('sqlite:///', '')
        if not os.path.isabs(db_path):
            # The database path is relative, make it absolute
            abs_db_path = os.path.join(parent_dir, db_path)
            new_tracking_uri = f'sqlite:///{abs_db_path}'
            print(f"Changing tracking URI from {tracking_uri} to {new_tracking_uri}")
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
    with mlflow.start_run(run_id=mlflow_run_id, run_name="rf-credit-score" if not mlflow_run_id else None) as run:
        if mlflow_run_id:
            print(f"Running under mlflow run with ID: {mlflow_run_id}")
        else:
            print("Standalone execution, creating new run...")
        
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
