import datetime
import time
import random
import logging
from prefect.tasks import task_input_hash
import pandas as pd
import psycopg
import joblib

from prefect import task, flow

from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import DatasetDriftMetric, DatasetMissingValuesMetric
from evidently.metrics import ColumnDriftMetric, ColumnQuantileMetric

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists metrics;
create table metrics(
	timestamp timestamp,
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float,
	quantile_values float
)
"""

reference_data = pd.read_parquet("HW5/reference.parquet")
with open("HW5/lin_reg.bin", "rb") as f_in:
    model = joblib.load(f_in)

raw_data = pd.read_parquet("data/green/green_tripdata_2023-03.parquet")

begin = datetime.datetime(2023, 3, 1, 0, 0)
num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
cat_features = ["PULocationID", "DOLocationID"]
column_mapping = ColumnMapping(
    prediction="prediction",
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None,
)

report = Report(
    metrics=[
        ColumnDriftMetric(column_name="prediction"),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        ColumnQuantileMetric(column_name="fare_amount", quantile=0.5),
    ]
)


@task(
    retries=2,
    retry_delay_seconds=5,
    name="Prepare database",
    cache_key_fn=task_input_hash,
)
def prep_db():
	with psycopg.connect("host=localhost port=5432 user=postgres password=mlops", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		if len(res.fetchall()) == 0:
			conn.execute("create database test;")
		with psycopg.connect("host=localhost port=5432 dbname=test user=postgres password=mlops") as conn:
			conn.execute(create_table_statement)


@task(
    retries=2,
    retry_delay_seconds=5,
    name="Calculate metrics",
    cache_key_fn=task_input_hash,
)
def calculate_metrics_postgresql(curr, i):
    current_data = raw_data[
        (raw_data.lpep_pickup_datetime >= (begin + datetime.timedelta(i)))
        & (raw_data.lpep_pickup_datetime < (begin + datetime.timedelta(i + 1)))
    ]

    # current_data.fillna(0, inplace=True)
    current_data["prediction"] = model.predict(
        current_data[num_features + cat_features].fillna(0)
    )

    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    result = report.as_dict()

    prediction_drift = result["metrics"][0]["result"]["drift_score"]
    num_drifted_columns = result["metrics"][1]["result"]["number_of_drifted_columns"]
    share_missing_values = result["metrics"][2]["result"]["current"][
        "share_of_missing_values"
    ]
    quantile_values = result["metrics"][3]["result"]["current"]["value"]

    curr.execute(
        "insert into metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values, quantile_values) values (%s, %s, %s, %s, %s)",
        (
            begin + datetime.timedelta(i),
            prediction_drift,
            num_drifted_columns,
            share_missing_values,
            quantile_values,
        ),
    )


@flow
def batch_monitoring_backfill():
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    with psycopg.connect(
        "host=localhost port=5432 dbname=test user=postgres password=mlops",
        autocommit=True,
    ) as conn:
        for i in range(0, 31):
            with conn.cursor() as curr:
                calculate_metrics_postgresql(curr, i)

            new_send = datetime.datetime.now()
            seconds_elapsed = (new_send - last_send).total_seconds()
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=10)
            logging.info("data sent")


if __name__ == "__main__":
    batch_monitoring_backfill()
