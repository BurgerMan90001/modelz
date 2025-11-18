import pandas as pd

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split


from pipelines import define_pipeline
from scoring import print_cross_val_score
from scoring import print_score_mae

from sklearn.preprocessing import LabelEncoder

bank_data = pd.read_csv("data/bank-full.csv", delimiter=";")

TARGET = "y"

X = bank_data.drop(TARGET,axis=1)
y = bank_data[TARGET]

# Manually encode target data
label_encoder = LabelEncoder()
encode_y = pd.Series(label_encoder.fit_transform(y)) # pyright: ignore[reportCallIssue, reportArgumentType]

# 80% traing 20% validation
X_train, X_valid, y_train, y_valid = train_test_split(X,encode_y, train_size=0.8,test_size=0.2)

# non-numerical columns
categorical_cols = [col for col in X_train if X_train[col].dtype == "object"]

# numerical columns
num_cols = list(X_train.drop(categorical_cols,axis=1))


model = XGBRegressor(n_estimators=1000,
                     learning_rate=0.01,
                     random_state=0)


# non-numerical columns
categorical_cols = [col for col in X_train if (X_train [col].dtype == "object") and (X_train [col].nunique() < 15)]

# numerical columns
numerical_cols = [col for col in X_train.columns if X_train [col].dtype in ['int64', 'float64']]

#print(y.columns)
pipeline = define_pipeline(model, numerical_cols, categorical_cols)


print_cross_val_score(pipeline,X, encode_y)