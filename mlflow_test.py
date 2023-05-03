import os
import warnings
import sys
import itertools

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
from xgboost import XGBRegressor
import time
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient


import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    # a higher value indicating a better fit between the predicted and actual values
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def columns_to_drop(df, *args):
    '''
    Drop columns by missing value, correlation & variance
    all args should be in list format!
    '''
    # cols with too much missing
    del_cols = [column for column in df.columns if df[column].isnull().mean() > 0.4]
    
    # additional cols to drop
    columns = [del_cols]+ [*args]
    
    flattened_list = list(itertools.chain(*columns))
    
    df = df.copy().drop(columns=flattened_list)
    return df


def put_target_front(df, target_list):
    '''
    change the order of whatever you want
    '''
    df = df[target_list + [x for x in df.columns if x not in target_list]]
    
    return df


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Load dataset
    df = pd.read_csv("./data/my_example_data.csv")

    target_col = 'OXO-5FI696 Augusta'
    target_list = ['OXO-5FI696 Augusta',
                    'OXO-5FIC600 Augusta',
                    'OXO-5FIC601 Augusta',
                    'OXO-5FIC612A Augusta',
                    'OXO-5FIC612B Augusta']

    # Correlation
    cor_to_drop = ['OXO-5FIC609A Augusta','OXO-5FI661A Augusta',
                    'OXO-5LI651E Augusta','OXO-5LI652E Augusta',
                    'OXO-5LI653E Augusta','OXO-5TIC603 Augusta',
                    'OXO-5TIC605 Augusta','OXO-5TIC607 Augusta',
                    'OXO-_5FI659A Augusta','OXO-_5FI660A Augusta',
                    'OXO-_5FI662A Augusta']

    # Variance
    var_to_drop = ['OXO-5RIC606_Y Augusta']

    # query together
    df = (df.query(f'`{target_col}`>2000')
        .pipe(columns_to_drop, cor_to_drop, var_to_drop)
        .fillna(method='ffill')
        .pipe(put_target_front, target_list)
     )

    # Turn into a supervised ML problem
    df_2 = df.set_index('Date')
    # ensure all data is float
    values = df_2.values
    values = values.astype('float32')

    # Data Splitting
    n_features = 43
    # set size
    train_size = int(0.8 * len(df))
    val_size = int(0.1 * len(df))
    test_size = len(df) - train_size - val_size
    # train, val, test splitting
    train = values[:train_size, :]
    val = values[train_size: train_size+val_size, :]
    test = values[-test_size:, :]
    # x, y splitting
    train_X, train_y = train[:, 1:n_features+1], train[:, 0]
    val_X, val_y = val[:, 1:n_features+1], val[:, 0]
    test_X, test_y = test[:, 1:n_features+1], test[:, 0]

    # reshape input to be 2D [samples, features]
    train_X = train_X.reshape((train_X.shape[0], n_features))
    val_X = val_X.reshape((val_X.shape[0], n_features))
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
        model.fit(train_X, train_y, 
                            eval_set=[(train_X, train_y), (val_X, val_y)],
                            eval_metric='rmse',
                            verbose=False)
        results = model.evals_result()
        train_rmse = results['validation_0']['rmse']
        val_rmse = results['validation_1']['rmse']
        epochs = len(results['validation_0']['rmse'])
        x_axis = range(0, epochs)

        
        fig, ax = plt.subplots()
        ax.plot(x_axis, train_rmse, label='Train')
        ax.plot(x_axis, val_rmse, label='Validation')
        ax.legend()
        plt.ylabel('RMSE')
        plt.title('XGBoost RMSE')


        client = MlflowClient()
        client.log_batch(mlflow.active_run().info.run_id, 
            metrics=[Metric(key="train rmse", value=val, timestamp=int(time.time() * 1000), step=i) for i, val in enumerate(train_rmse)])
        client.log_batch(mlflow.active_run().info.run_id, 
            metrics=[Metric(key="val rmse", value=val, timestamp=int(time.time() * 1000), step=i) for i, val in enumerate(val_rmse)])


        # Test
        test_yhat = model.predict(test_X)

        (rmse, mae, r2) = eval_metrics(test_y, test_yhat)

        print("XGBoost model (n_estimators={:d}, max_depth={:d}, learning_rate={:f}):"
              .format(n_estimators, max_depth, learning_rate))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_metric("test rmse", rmse)
        mlflow.log_metric("test mae", mae)
        mlflow.log_metric("test r2", r2)

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
        

        plt.show()