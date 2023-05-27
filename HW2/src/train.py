import os
import pickle
import argparse

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def run_train(data_path: str):
    with mlflow.start_run():
        mlflow.set_tag("developer", "Camila")
        mlflow.sklearn.autolog()
        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        run_id = mlflow.last_active_run().info.run_id
        print("Logged data and model in run: {}".format(run_id))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Location where the processed NYC taxi trip data was saved')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the train data')
    args = parser.parse_args()
    data_path = args.data_path
    data_path = os.path.join(data_path)
    run_train(data_path)
