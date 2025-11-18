import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

bank_data = pd.read_csv("data/bank-full.csv", delimiter=";")

# Input data
X = bank_data

# Target data
Y = bank_data

X_train, X_valid, Y_train, Y_valid  = train_test_split(X, Y)






#print(num_cols)
#print(categorical_cols)