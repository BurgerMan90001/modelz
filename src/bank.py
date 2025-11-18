import pandas as pd

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from pipelines import define_pipeline
from scoring import print_cross_val_score
from scoring import print_score


bank_data = pd.read_csv("data/bank-full.csv", delimiter=";")

TARGET = "y"
# Input data
X = bank_data.drop(TARGET,axis=1)


# Target data
y = bank_data[TARGET]
print(y)
# 80% traing 20% validation
X_train, X_valid, y_train, y_valid = train_test_split(X,y, train_size=0.8,test_size=0.2)

#print(shill_bidding_data)

# non-numerical columns
categorical_cols = [col for col in X_train if X_train[col].dtype == "object"]

# numerical columns
num_cols = list(X_train.drop(categorical_cols,axis=1))

#ord_encoder = OrdinalEncoder()

#encode_y_train = pd.DataFrame(ord_encoder.fit_transform(y_train))
#encode_y_valid = pd.DataFrame(ord_encoder.transform(y_valid))


model = XGBRegressor(n_estimators=250,
                     learning_rate=0.1,
                     random_state=0)


# non-numerical columns
categorical_cols = [col for col in X_train if (X_train [col].dtype == "object") and (X_train [col].nunique() < 15)]
#print(categorical_cols)
# numerical columns
numerical_cols = [col for col in X_train .columns if X_train [col].dtype in ['int64', 'float64']]
#print(numerical_cols)

pipeline = define_pipeline(model, numerical_cols, categorical_cols, y)


pipeline.fit(X_train, y_train)


#print(pipeline.score(X,y))
#print_cross_val_score(pipeline,X, y)

#pipeline.fit(X_train, y_train)

#predictions= pipeline.predict(X_valid)

#mae = mean_absolute_error(predictions, y_valid)

#print(mae)