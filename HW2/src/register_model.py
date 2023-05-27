import os
import pickle
import argparse
import mlflow

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-model-v1"
RF_PARAMS = [
    "max_depth",
    "n_estimators",
    "min_samples_split",
    "min_samples_leaf",
    "random_state",
    "n_jobs",
]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        for param in RF_PARAMS:
            params[param] = int(params[param])

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        # Evaluate model on the validation and test sets
        val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)
        mlflow.sklearn.log_model(rf, "model")


def run_register_model(data_path: str, top_n: int):
    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"],
    )
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    for run in runs:
        print(f"run id: {run.info.run_id}, rmse: {run.data.metrics['rmse']:.4f}")

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"],
    )

    for best_run in best_runs:
        print(
            f"run id: {best_run.info.run_id}, rmse: {best_run.data.metrics['test_rmse']:.4f}"
        )

        # Register the best model
        run_id = best_run.info.run_id
        mlflow.register_model(model_uri=f"runs:/{run_id}/models", name=EXPERIMENT_NAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Location where the processed NYC taxi trip data was saved"
    )
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to the train data"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        required=True,
        default=5,
        help="Number of top models that need to be evaluated to decide which one to promote",
    )
    args = parser.parse_args()
    data_path = args.data_path
    top_n = args.top_n
    data_path = os.path.join(data_path)
    run_register_model(data_path, top_n)
