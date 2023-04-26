import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from xgboost import XGBRegressor

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    # a higher value indicating a better fit between the predicted and actual values
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Load dataset
    df = pd.read_csv("../data/my_example_data.csv")

    # Missing values
    del_cols = [i for i in df.columns if df[i].isnull().sum() / df.shape[0] > 0.4]
    df.drop(columns=del_cols, inplace=True)
    fill_cols = [i for i in df.columns if df[i].isnull().sum()>0]
    for j in fill_cols:
        df[j]=df[j].fillna(df[j].mean())

    # Move target columns to front
    target_list = ['OXO-5FI696 Augusta',
                    'OXO-5FIC600 Augusta',
                    'OXO-5FIC601 Augusta',
                    'OXO-5FIC612A Augusta',
                    'OXO-5FIC612B Augusta']
    df['Date'] = pd.to_datetime(df['Date'])
    cols_to_move = ['Date'] + target_list
    df = df[cols_to_move + [x for x in df.columns if x not in cols_to_move]]

    # Correlation
    drop_cols = ['OXO-5FIC609A Augusta','OXO-5FI661A Augusta',
                    'OXO-5LI651E Augusta','OXO-5LI652E Augusta',
                    'OXO-5LI653E Augusta','OXO-5TIC603 Augusta',
                    'OXO-5TIC605 Augusta','OXO-5TIC607 Augusta',
                    'OXO-_5FI659A Augusta','OXO-_5FI660A Augusta',
                    'OXO-_5FI662A Augusta']
    df = df.drop(columns=drop_cols)

    # Delete anomoly time stamps
    threshold = 2000
    zero_indices = df[df['OXO-5FI696 Augusta'] < threshold].index
    df.drop(df.index[zero_indices], inplace=True)

    # Variance
    df = df.drop(columns=['OXO-5RIC606_Y Augusta'])

    # Turn into a supervised ML problem
    df_2 = df.set_index('Date')
    # ensure all data is float
    values = df_2.values
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    scaled = pd.DataFrame(scaled)

    # Data Splitting
    n_features = 43
    values = scaled.values
    # set size
    train_size = int(0.9 * len(df))
    test_size = len(df) - train_size
    # train, test splitting
    train = values[:train_size, :]
    test = values[-test_size:, :]
    # x, y splitting
    train_X, train_y = train[:, 1:n_features+1], train[:, 0]
    test_X, test_y = test[:, 1:n_features+1], test[:, 0]
    # reshape input to be 2D [samples, features]
    train_X = train_X.reshape((train_X.shape[0], n_features))
    test_X = test_X.reshape((test_X.shape[0], n_features))

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 7200
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    learning_rate = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01

    with mlflow.start_run():
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                             learning_rate=learning_rate, colsample_bytree=0.2, 
                             gamma=0.0, min_child_weight=1.5, 
                             reg_alpha=0.9, reg_lambda=0.6, 
                             subsample=0.2, random_state=123)
        # Train
        model.fit(train_X, train_y)

        # Test
        test_yhat = model.predict(test_X)
        # invert scaling for forecast
        test_yhat = test_yhat.reshape((test_yhat.shape[0], 1))
        inv_yhat = np.concatenate((test_yhat, test_X), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        # invert scaling for actual
        test_y1 = test_y.reshape((len(test_y), 1))
        inv_y = np.concatenate((test_y1, test_X), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]

        (rmse, mae, r2) = eval_metrics(inv_y, inv_yhat)

        print("XGBoost model (n_estimators={:d}, max_depth={:d}, learning_rate={:f}):"
              .format(n_estimators, max_depth, learning_rate))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        # print(mlflow.get_tracking_uri())
        # print(tracking_url_type_store)

        # Model registry does not work with file store
        if tracking_url_type_store != "file":
            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(model, "model", registered_model_name="XGBoostModel")
        else:
            mlflow.sklearn.log_model(model, "model")