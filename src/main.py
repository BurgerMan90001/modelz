import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

bank_data = pd.read_csv("modelz/data/bank-full.csv")


X = bank_data

y = bank_data
X_train = train_test_split()
#bank_data.describe()




