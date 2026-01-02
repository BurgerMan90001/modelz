import pandas as pd

from xgboost import XGBClassifier
from tpot import TPOTClassifier

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder



from util.pipelines import define_pipeline
from util.scoring import print_cross_val_score


bank_data = pd.read_csv("data/bank-full.csv", delimiter=";")

y_col = "y"

X = bank_data.drop([y_col, "month","poutcome", "day"],axis=1)
#print(X)
y = bank_data[y_col]

# Manually encode target data
label_encoder = LabelEncoder()
y = pd.Series(label_encoder.fit_transform(y)) # pyright: ignore[reportCallIssue, reportArgumentType]

#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123)

#model = XGBClassifier(n_estimators=250,learning_rate=0.01,andom_state=0)

#cv = RepeatedStratifiedKFold()

model = TPOTClassifier(scorers="accuracy",generations=5, population_size=50, n_jobs=4, verbose=2)
# non-numerical columns
categorical_cols = [col for col in X.columns if (X[col].dtype == "object") 
                    and (X [col].nunique() < 15)]

# numerical columns
numerical_cols = [col for col in X.columns if X [col].dtype in ['int64', 'float64']]

pipeline = define_pipeline(model, numerical_cols, categorical_cols)


model = model.fit(X, y)
#pipeline.fit(X, y_train)
#print_cross_val_score(pipeline,X, encode_y, scoring="accuracy")
model.export("bank_best_model.py")