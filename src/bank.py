import pandas as pd

from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from util.pipelines import define_pipeline
from util.scoring import print_cross_val_score
from util.scoring import print_score_mae

bank_data = pd.read_csv("data/bank-full.csv", delimiter=";")

y_col = "y"

X = bank_data.drop([y_col, "month","poutcome", "day"],axis=1)
#print(X)
y = bank_data[y_col]

# Manually encode target data
label_encoder = LabelEncoder()
encode_y = pd.Series(label_encoder.fit_transform(y)) # pyright: ignore[reportCallIssue, reportArgumentType]


#X_train, X_valid, y_train, y_valid = train_test_split(X,encode_y, train_size=0.8,test_size=0.2)

model = XGBRegressor(n_estimators=1000,
                     learning_rate=0.01,
                     random_state=41)


# non-numerical columns
categorical_cols = [col for col in X if (X [col].dtype == "object") 
                    and (X [col].nunique() < 15)]

# numerical columns
numerical_cols = [col for col in X.columns if X [col].dtype in ['int64', 'float64']]

pipeline = define_pipeline(model, numerical_cols, categorical_cols)

print_cross_val_score(pipeline,X, encode_y)
