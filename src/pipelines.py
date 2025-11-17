from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

def define_pipeline(model, numeric_cols,categorical_cols):
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant"))
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='constant')),
        ("oh-encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numbers', num_transformer, numeric_cols),
        ('categorical', categorical_transformer, categorical_cols)
    ])
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline
