
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
#from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

from pipelines import define_pipeline
from scoring import print_cross_val_score
from scoring import print_score



shill_bidding_data = pd.read_csv("data/shill-bidding.csv")
#shill_bidding_data = pd.read_csv("data/shill-bidding.csv")


TARGET = "Class"

X = shill_bidding_data.drop(TARGET,axis=1)
y= shill_bidding_data[TARGET]

X_train, X_valid, y_train, y_valid = train_test_split(X,y, train_size=0.8,test_size=0.2)

model = XGBRegressor(n_estimators=250,
                     learning_rate=0.1,
                     random_state=0)


# non-numerical columns
categorical_cols = [col for col in X_train if (X_train[col].dtype == "object") and (X[col].nunique() < 15)]
#print(categorical_cols)
# numerical columns
numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]

pipeline = define_pipeline(model, numerical_cols, categorical_cols, y_train)


scores = -1 * cross_val_score(pipeline, X, y,
                                  cv=5, scoring="neg_mean_absolute_error", error_score="raise")

print(scores)
print(scores.mean())