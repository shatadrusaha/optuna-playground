"""                     Import libraries.                       """
import os
import pandas as pd
from datetime import datetime as dt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from pyzinga.pyzinga import data_tools as pdt
from pyzinga.pyzinga import plot_tools as ppt
from pyzinga.pyzinga import optuna_tools as pot

# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer


"""                     User defined variables.                       """
# Folders for saving artifacts (plots and files).
folder_plots = 'artifacts/plots'
folder_files = 'artifacts/files'

# Optuna study.
n_trials = 100  # Number of trials to run.
threshold = 0.5  # Threshold for binary classification.
problem_type = 'binary_classification'  # or 'regression'

# Mlflow.
# TODO - Make sure to start the mlflow server/ui on the specific port first, via terminal.
# mlflow server --host localhost --port 8080
mlflow_exp_name = 'lgbm-optuna-binary-classification'  # Experiment name.
mlflow_tracking_uri = 'http://localhost:8080' # MLflow Tracking Server URI.

# Miscellaneous.
gpu_flag = False  # Set to True if GPU is available, otherwise False.
random_state = 14  # Random state for reproducibility.


"""                     Load and preprocess the data.                       """
# Create folders for saving artifacts.
os.makedirs(folder_plots, exist_ok=True)
os.makedirs(folder_files, exist_ok=True)

"""
https://scikit-learn.org/stable/datasets/loading_other_datasets.html
https://www.openml.org/search?type=data&sort=version&status=any&order=asc&exact_name=adult&id=1590
"""
# Fetch the 'adult' dataset from OpenML.
X, y = fetch_openml(name='adult', version=2, return_X_y=True, as_frame=True)
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")

df_describe = pdt.describe_dataset(X, include='all')
print(f"Dataframe description:\n{df_describe}\n")

# Get the numerical and categorical columns.
cols_num = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cols_cat = X.select_dtypes(include=['category', 'object']).columns.tolist()
print(f"Numerical columns:\n\t{cols_num}")
print(f"Categorical columns:\n\t{cols_cat}\n")

# Display the unique values in categorical columns.
print("Unique values in categorical columns:\n")
for col in cols_cat:
    print(f"{X[col].value_counts(dropna=False)}\n")
    # print(f"Unique values in '{col}': {X[col].unique()}")
    # print(f"Number of unique values in '{col}': {X[col].nunique()}\n")

# Fill missing values in categorical columns with 'Missing'.
for col in cols_cat:
    # Count missing values in the column.
    na_count = X[col].isna().sum()
    print(f"Missing values in '{col}': {na_count}")

    if na_count > 0:
        # Categorical columns.
        if isinstance(X[col].dtype, pd.CategoricalDtype):
            # Add 'Missing' to categories, if not already present.
            if 'Missing' not in X[col].cat.categories:
                X[col] = X[col].cat.add_categories('Missing')
        
        X[col] = X[col].fillna('Missing')
        
# Display the dataframe description after filling missing values.
df_describe = pdt.describe_dataset(X, include='all')
print(f"Dataframe description after filling missing values:\n{df_describe}\n")

# Display the value counts of the target variable.
print(f"Value counts for 'y' (target):\n{y.value_counts(dropna=False)}\n")

# Encode the target labels.
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Encoded 'y' classes: {le.classes_}")
print(f"Sample encoded 'y': {y_encoded[:5]}\n")

# Correlation heatmap.
ppt.corr_heatmap_plot(
    df=pd.concat(objs=[X, pd.Series(data=y_encoded, name='class')], axis=1),
    save_dir=folder_plots
)
ppt.corr_heatmap_plot(
    df=pd.concat(objs=[X, pd.Series(data=y_encoded, name='class')], axis=1), 
    save_dir=folder_plots,
    mask_upper=False
)

"""
# Scale the numerical columns.
ct = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), cols_num),
    ],
    remainder='passthrough'
)

X_transformed = ct.fit_transform(X)

print(f"Column names - raw data:\n\t{list(ct.feature_names_in_)}\n")
print(f"Column names - transformed data:\n\t{ct.get_feature_names_out().tolist()}\n")
"""

# Split the data into training, validation, and test sets.
X_train, X_val, X_test, y_train, y_val, y_test = pdt.split_data(
    X=X,
    y=y_encoded,
    split_test=0.2,
    split_val=0.1,
    random_state=random_state
)
print(f"Shapes of the datasets:\n"
      f"\tX_train: {X_train.shape}, y_train: {y_train.shape}\n"
      f"\tX_val: {X_val.shape}, y_val: {y_val.shape}\n"
      f"\tX_test: {X_test.shape}, y_test: {y_test.shape}\n")
# Shapes of the datasets:
# 	X_train: (35165, 14), y_train: (35165,)
# 	X_val: (3908, 14), y_val: (3908,)
# 	X_test: (9769, 14), y_test: (9769,)

# pd.DataFrame(y_encoded).value_counts().plot(kind='bar')
pd.Series(y_encoded).value_counts(normalize=True)


"""                     Model training and evaluation.                       """

"""
# Objective parameter details:
binary

# Available options for parameters:
- objective --> 'binary'
- boosting --> 'gbdt', 'rf', 'dart'
- data_sample_strategy --> 'bagging', 'goss'
    - 'bagging' is only effective when 'bagging_freq' > 0 and 'bagging_fraction' < 1.0
- tree_learner --> 'serial', 'feature', 'data', 'voting'

# Default values for parameters:
- objective --> 'regression'
- boosting --> 'gbdt'
- data_sample_strategy --> 'bagging'
- num_iterations --> 100
- learning_rate --> 0.1
- num_leaves --> 31
- tree_learner --> 'serial'
- num_threads --> 0
- device_type --> 'cpu'
- seed --> None
- max_depth --> -1 (no limit)
- min_data_in_leaf --> 20
- bagging_fraction --> 1.0
- bagging_freq --> 0 (no bagging)
- feature_fraction --> 1.0
- early_stopping_round --> 0
- lambda_l1 --> 0.0
- lambda_l2 --> 0.0
- drop_rate --> 0.1 (for 'dart' ONLY)
- skip_drop --> 0.5 (for 'dart' ONLY)
- metric --> "" (based on objective)
"""

# LightGBM model build parameters for optuna study.
params_lgbm = {
    # Core parameters
    'objective': {'type': 'constant', 'value': 'binary'},
    # 'boosting': {'type': 'constant', 'value': 'dart'},
    'boosting': {'type': 'categorical', 'choices': ['gbdt', 'rf']},
    'data_sample_strategy': {'type': 'constant', 'value': 'bagging'},
    # 'data_sample_strategy': {'type': 'categorical', 'choices': ['bagging', 'goss']},
    'num_iterations': {'type': 'int', 'low': 100, 'high': 1000, 'step': 50, 'log': False},
    'learning_rate': {'type': 'float', 'low': 0.001, 'high': 0.8, 'step': None, 'log': True},
    'num_leaves': {'type': 'int', 'low': 10, 'high': 100, 'step': 5, 'log': False},
    # 'tree_learner': {'type': 'categorical', 'choices': ['serial', 'feature', 'data']},
    'num_threads': {'type': 'constant', 'value': os.cpu_count()},
    'device_type': {'type': 'constant', 'value': 'gpu' if gpu_flag else 'cpu'},
    'seed': {'type': 'constant', 'value': random_state},

    # Learning control parameters
    # TODO - check, if it's 'true' or True
    # 'force_col_wise': {'type': 'constant', 'value': 'true'},
    'max_depth': {'type': 'int', 'low': 5, 'high': 100, 'step': 5, 'log': False},
    'min_data_in_leaf': {'type': 'int', 'low': 20, 'high': 100, 'step': 5, 'log': False},
    'bagging_fraction': {'type': 'float', 'low': 0.5, 'high': 0.99, 'step': 0.01, 'log': False},
    'bagging_freq': {'type': 'int', 'low': 5, 'high': 50, 'step': 5, 'log': False},
    'feature_fraction': {'type': 'float', 'low': 0.5, 'high': 1.0, 'step': 0.01, 'log': False},
    'lambda_l1': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'step': None, 'log': True},
    'lambda_l2': {'type': 'float', 'low': 1e-8, 'high': 10.0, 'step': None, 'log': True},
    # 'drop_rate': {'type': 'float', 'low': 0.1, 'high': 0.5, 'step': 0.01, 'log': False},  # For 'dart' ONLY

    # Metric parameters
    # TODO - leave this parameter to it's default value. calculating metrics seaparately using sklearn.
    # 'metric': {'type': 'constant', 'value': ['auc', 'average_precision', 'binary_logloss']}, # Binary classification metrics.
    # 'metric': {'type': 'constant', 'value': ['mae', 'mse', 'rmse', 'mape', 'huber']}, # Regression metrics.
}

# Optimiser parameters.
"""
# LightGBM
'maximize' --> 'auc', 'average_precision'
'minimize' --> 'binary_logloss'

# Scikit-learn/Pyzinga framework
'maximize' --> 'accuracy', 'f1_score', 'precision', 'recall', 'roc_auc', 'average_precision',
'minimize' --> 'log_loss'
"""
# TODO - Use scikit-learn metrics ONLY for optimization.
optimiser = {
    'name': 'accuracy',
    'direction': 'maximize',  # Direction to optimize the metric for study.
    'alias': 'acc',
}

# Tag parameters.
try:
    boosting_type = params_lgbm['boosting']['value']
except KeyError:
    boosting_type = '-'.join(params_lgbm['boosting']['choices'])
params_tags = {
    'model_type': 'lgbm',
    'optimiser': optimiser['alias'],
    'boosting_type': boosting_type,
}

# MLflow parameters.
mlflow_run_name = f"{boosting_type}-{optimiser['alias']}-{dt.now().strftime('%Y%m%d-%H%M%S')}" # Run name.
params_mlflow = {
    'mlflow_exp_name': mlflow_exp_name,
    'mlflow_run_name': mlflow_run_name,
    'mlflow_tracking_uri': mlflow_tracking_uri,
    'parent_run_id': None,  # This will be set during the optimization process.
}

# Dataset parameters.
params_data = {
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
    'X_test': X_test,
    'y_test': y_test,
    'threshold': threshold
}

# Study parameters.
params_study = {
    "n_trials": n_trials,
    "direction": optimiser['direction'],  # Direction to optimize the metric.
    "optimiser_metric": optimiser['name'], # Metric to optimize.
    "run_parallel": True,  # Set to True for parallel execution, False for sequential.
}

# Model evaluation parameters.
params_model_eval = {
    'problem_type': problem_type,
    'threshold': threshold,  # Threshold for binary classification.
    'n_samples': 10000,  # Number of samples for SHAP analysis.
    'topn_features': 20,  # Number of top features to display in model feature importance and SHAP plots.
}

# Run the optimization process.
study = pot.run_optimization(
    params_lgbm=params_lgbm,
    params_data=params_data,
    params_mlflow=params_mlflow,
    params_study=params_study,
    params_model_eval=params_model_eval,
    params_tags=params_tags
)

# Print the best trial results.
print(f"Best trial: {study.best_trial.number}")
print(f"Best value: {study.best_value}")
print(f"Best params: {study.best_params}")
