"""                     Import libraries.                       """
from sklearn.datasets import fetch_openml
from pyzinga


import sklearn.datasets


"""                     User defined variables.                       """
# Random seed.
random_seed = 14


"""                     Get the data.                       """
"""
https://scikit-learn.org/stable/datasets/loading_other_datasets.html
https://www.openml.org/search?type=data&sort=version&status=any&order=asc&exact_name=adult&id=1590
"""
# Fetch the 'adult' dataset from OpenML.
X, y = fetch_openml(name='adult', version=2, return_X_y=True, as_frame=True)

print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")
