import pandas as pd

from xgboost import XGBClassifier
from tpot import TPOTClassifier

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


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


categorical_cols = [col for col in X.columns if (X[col].dtype == "object") 
                    and (X [col].nunique() < 15)]

# numerical columns
numerical_cols = [col for col in X.columns if X [col].dtype in ['int64', 'float64']]


num_transformer = SimpleImputer(strategy="constant")

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='constant')),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer(transformers=[
    ('numbers', num_transformer, numerical_cols),
    ('categorical', categorical_transformer, categorical_cols),
])

X = preprocessor.fit_transform(X)

#model = XGBClassifier(n_estimators=1000,learning_rate=2e-5,random_state=0)

#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=123)
#cv = RepeatedStratifiedKFold()

model = TPOTClassifier(
    scorers=["accuracy"],
    generations=5, 
    population_size=50, 
    n_jobs=6, 
    verbose=3,
    early_stop=10
)
# non-numerical columns

model.fit(X,y)

#pred = model.predict()
#pipeline.fit(X, y_train)
#print_cross_val_score(model,X, y, scoring="accuracy")
model.export("bank_best_model.py")