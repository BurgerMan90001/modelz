from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OrdinalEncoder

from sklearn.preprocessing import LabelEncoder


def define_pipeline(model, numeric_cols, categorical_cols, y_col):
    """_summary_

    Args:
        model (_type_): _description_
        numeric_cols (_type_): _description_
        categorical_cols (_type_): _description_

    Returns:
        _type_: _description_
    """
    num_transformer = SimpleImputer(strategy="constant")
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy='constant')),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    label_transformer = LabelEncoder()

    preprocessor = ColumnTransformer(transformers=[
        ('numbers', num_transformer, numeric_cols),
        ('categorical', categorical_transformer, categorical_cols),
        #('y-col', label_transformer, y_col),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    return pipeline
