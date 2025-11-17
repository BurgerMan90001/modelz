
import pandas as pd

from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

from pipelines import define_pipeline
from scoring import print_cross_val_score

shill_bidding_data = pd.read_csv("data/shill-bidding.csv")

#shill_bidding_data = shill_bidding_data.drop(["Bidder_ID"],axis=1)

TARGET = "Winning_Ratio"

X = shill_bidding_data.drop(TARGET,axis=1)
y= shill_bidding_data[TARGET]

# 80% traing 20% validation
X_train, X_valid, y_train, y_valid = train_test_split(X,y, train_size=0.8,test_size=0.2)

# non-numerical columns
categorical_cols = [col for col in X_train if X_train[col].dtype == "object"]

# numerical columns
num_cols = list(X_train.columns.drop(categorical_cols))

model = XGBRegressor(n_estimators=250,
                     learning_rate=0.1,
                     random_state=0)
#model.fit(X_train, y_train, early_stopping_rounds=5, eval_set=[(X_valid, y_valid)])

usePipline = True

if usePipline:
    pipeline = define_pipeline(model, numeric_cols=num_cols, categorical_cols=categorical_cols)
    print_cross_val_score(pipeline,X ,y, y_valid)
else:
    print_cross_val_score(model,X ,y, y_valid)
