#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import FunctionTransformer
import xgboost as xgb

import seaborn as sns
import matplotlib.pyplot as plt

import mlflow
from prefect import task, flow, get_run_logger
from prefect.context import get_run_context

import pickle
import pathlib
import os
import uuid
import sys
from dateutil.relativedelta import relativedelta
from datetime import datetime


@task
def read_data(filename):
    df = pd.read_parquet(filename)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    
    df["duration"] = df.duration.apply(lambda x: x.total_seconds()/60)
    
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    ride_ids = [str(uuid.uuid4()) for _ in range(len(df))]
    df["ride_id"] = ride_ids
    
    categorical = ["PULocationID", "DOLocationID"]
    
    df[categorical] = df[categorical].astype(str)

    return df

@task
def prepare_data(df):
    df['PU_DU'] = df["PULocationID"] + '_' + df["DOLocationID"]

    categorical = ["PU_DU"]
    numeric = ["trip_distance"]

    dicts = df[categorical + numeric].to_dict(orient="records")

    return dicts

@task
def load_model(run_id):
    logged_model = f"gs://mlflow_artifacts_nyc_taxi/1/{run_id}/artifacts/models_mlflow"
    model = mlflow.pyfunc.load_model(logged_model)

    return model

@task
def save_results(df, output_file, y_pred, run_id):
    df_results = df[['ride_id', "lpep_pickup_datetime", "PULocationID", "DOLocationID", "duration"]].copy()
    df_results.rename(columns={"duration": "actual_duration"}, inplace=True)
    df_results["predicted_duration"] = y_pred
    df_results["diff"] = df_results["predicted_duration"] - df_results["actual_duration"]
    df_results["model_version"] = run_id

    pathlib.Path("output").mkdir(exist_ok=True)
    df_results.to_parquet(output_file, index=False)


@task
def apply_model(input_file, run_id, output_file):
    logger = get_run_logger()

    logger.info(f'reading data from the {input_file}...')
    df= read_data(input_file)
    dicts = prepare_data(df)

    logger.info(f'loading the model with RUN_ID={run_id}...')
    model = load_model(run_id)

    logger.info(f'predicting the duration...')
    y_pred = model.predict(dicts)

    logger.info(f'saving the results to {output_file}...')
    save_results(df, output_file, y_pred, run_id)

    return output_file


def get_paths(run_date, taxi_type, run_id):
    prev_month = run_date - relativedelta(months=1)
    year = prev_month.year
    month = prev_month.month

    input_file = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet"
    output_file = f"gs://taxi_bucket_1/taxi_type={taxi_type}/year={year:04d}/month={month:02d}/{run_id}.parquet"

    return input_file, output_file

@flow
def ride_duration_prediction(
    taxi_type: str = "green",
    run_id: str = None,
    run_date: datetime = None
):
    if run_date is None:
        ctx = get_run_context()
        run_date = ctx.flow_run.expected_start_time
    
    input_file, output_file = get_paths(run_date, taxi_type, run_id)

    apply_model(input_file, run_id, output_file)


    

def run():
    print("hello")
    taxi_type = sys.argv[1] #"green"
    year = int(sys.argv[2]) #2021
    month = int(sys.argv[3]) #3
    run_id = sys.argv[4] #"1"
    #RUN_ID = os.getenv("RUN_ID")
    #apply_model(input_file, run_id, output_file)

    gcp_credentials_block = GcpCredentials.load(name="my-gcp-creds")
    gcs_credentials_dict = gcp_credentials_block.service_account_info

    with open('gcs_credentials.json', 'w') as f:
        json.dump(gcs_credentials_dict, f)

    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.abspath('gcs_credentials.json')

    ride_duration_prediction(taxi_type=taxi_type, run_id=run_id, run_date=datetime(year, month, 1))

if __name__ == "__main__":
    run()










