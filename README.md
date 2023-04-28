# MLflow Demo: XGBoost on SASOL_OXO Data

## Introduction

In this project I work on the SASOL OXO data, the original dataset is in `./data/my_example_data.csv`. I did EDA on the data in `./EDA.ipynb`, and then used LSTM and XGBoost to train and test the data, which is also done in `./EDA.ipynb`.  

Then I ran a MLflow demo in `./mlflow_test.py`, it's like a simplified version of that EDA file, with XGBoost implemented on the finalized data, which supports MLflow Tracking and MLflow Projects.



## Quick Start

**MLflow Tracking**: 

MLflow Tracking lets you log and query parameters, code versions, metrics, and output files when running your machine learning code and for later visualizing the results.

Run `python mlflow_test.py` in the repo directory, this will run the codes with the default XGBoost model. The default hyper-parameters are `n_estimators=7200, max_depth=5, learning_rate=0.01 `. Use the customized parameters by running `	python mlflow_test.py <n_estimators> <max_depth> <learning_rate>`

You can then run `mlflow ui`to open up a page with all the experiment details on it.

**MLflow Projects**: 

An MLflow Project is a format for packaging data science code in a reusable and reproducible way, based primarily on conventions.

- On your device, if you already download this repo, run the command with the format 

  `mlflow run /path/to/conda/project -P <parameter=xxx>`.

  For example, in the root directory, run:

  `mlflow run . -P n_estimators=720 learning_rate=0.1`

  It will follow what is indicated in `MLproject` including:

  1. Name
  2. Environment. For my example it will automatically install the conda environment specified in `./files/config/conda.yaml` on your device
  3. Entry Points. Run the python file with the command stated and the parameters provided.

- On your device, if you haven't downloaded this repo, try run:

  `mlflow run https://github.com/Rain-Elutions/OXO-Model-MLflow.git`      (HTTPS)

  or

  `mlflow run git@github.com:Rain-Elutions/OXO-Model-MLflow.git`        (SSH)

  It will do the same thing above but from different source.

  ​

## Reference

1. Official document: [MLflow Documentation — MLflow 2.3.1 documentation](https://mlflow.org/docs/latest/index.html)

2. More examples: [mlflow/examples at master · mlflow/mlflow · GitHub](https://github.com/mlflow/mlflow/tree/master/examples)

3. Video tutorials: https://www.youtube.com/watch?v=VokAGy8C6K4

   ​