import pathlib
import pickle

import pandas as pd
import numpy as np
import scipy
#import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import root_mean_squared_error
import xgboost as xgb

import mlflow
from prefect import flow, task


@task(retries=3, retry_delay_seconds=2)
def read_data(filename: str) -> pd.DataFrame:
    """read data from parquet file"""
    df = pd.read_parquet(filename)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df["duration"] = df.duration.apply(lambda x: x.total_seconds()/60)
    
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    
    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df

@task
def add_features(
    df_train: pd.DataFrame, df_val: pd.DataFrame
) -> tuple(
    [
        scipy.sparse.csr_matrix,
        scipy.sparse.csr_matrix,
        np.ndarray,
        np.ndarray,
        DictVectorizer
    ]
):
    """add features to the model"""
    df_train['PU_DU'] = df_train["PULocationID"] + '_' + df_train["DOLocationID"]
    df_val['PU_DU'] = df_val["PULocationID"] + '_' + df_val["DOLocationID"]

    categorical = ["PU_DU"]
    numeric = ["trip_distance"]

    dv = DictVectorizer()

    train_dict = df_train[categorical + numeric].to_dict(orient="records")
    X_train = dv.fit_transform(train_dict)

    val_dict = df_val[categorical + numeric].to_dict(orient="records")
    X_val = dv.transform(val_dict)

    y_train = df_train["duration"].values
    y_val = df_val["duration"].values

    return X_train, X_val, y_train, y_val, dv

@task(log_prints=True)
def train_model(
    X_train: scipy.sparse.csr_matrix,
    X_val: scipy.sparse.csr_matrix,
    y_train: np.ndarray,
    y_val: np.ndarray,
    dv: DictVectorizer
) -> None:
    """train the model with best hyperparameters and write everything out"""

    with mlflow.start_run():
        train = xgb.DMatrix(X_train, label=y_train)
        val = xgb.DMatrix(X_val, label=y_val)

        best_params = {
            'learning_rate': 0.811061633170207,
            'max_depth': 3,
            'min_child_weight': 1.7524631655337792,
            'reg_alpha': 0.15412140074890998,
            'reg_lambda': 0.3079092817662672,
            'seed': 42,
            'objective': 'reg:linear'
             }
        
        mlflow.log_params(best_params)

        booster = xgb.train(
            best_params,
            train,
            num_boost_round=100,
            evals=[(val, "val")],
            early_stopping_rounds=10,
        )

        y_pred = booster.predict(val)
        rmse = root_mean_squared_error(y_val, y_pred)
        mlflow.log_metrics({"rmse": rmse})

        pathlib.Path("models").mkdir(exist_ok=True)
        with open("models/preprocessor.b", "wb") as f:
            pickle.dump(dv, f)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")


@flow
def main_flow(
    train_path: str = "./data/green_tripdata_2021-01.parquet",
    val_path: str = "./data/green_tripdata_2021-02.parquet"
) -> None:
    """main training pipeline"""

    #mlflow settings
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment-1")

    #load
    df_train = read_data(train_path)
    df_val = read_data(val_path)

    #transform
    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val)

    #train
    train_model(X_train, X_val, y_train, y_val, dv)

if __name__ == "__main__":
    main_flow()
