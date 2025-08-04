"""                     Import libraries.                       """
import os
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from pyzinga.pyzinga import data_tools as pdt
from pyzinga.pyzinga import plot_tools as ppt

# from sklearn.preprocessing import StandardScaler
# from sklearn.compose import ColumnTransformer


"""                     User defined variables.                       """
# Folders for saving artifacts (plots and files).
folder_plots = 'artifacts/plots'
folder_files = 'artifacts/files'

# Random seed.
random_seed = 14


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


"""                     Model training and evaluation.                       """
# Split the data into training, validation, and test sets.
X_train, X_val, X_test, y_train, y_val, y_test = pdt.split_data(
    X=X,
    y=y_encoded,
    split_test=0.2,
    split_val=0.1,
    random_state=random_seed
)
print(f"Shapes of the datasets:\n"
      f"\tX_train: {X_train.shape}, y_train: {y_train.shape}\n"
      f"\tX_val: {X_val.shape}, y_val: {y_val.shape}\n"
      f"\tX_test: {X_test.shape}, y_test: {y_test.shape}\n")
# Shapes of the datasets:
# 	X_train: (35165, 14), y_train: (35165,)
# 	X_val: (3908, 14), y_val: (3908,)
# 	X_test: (9769, 14), y_test: (9769,)
