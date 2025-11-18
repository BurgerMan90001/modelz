from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error

def print_score_mae(model, X_train, X_valid, y_train, y_valid):
    model.fit(X_train, y_train)
    predictions= model.predict(X_valid)
    mae = mean_absolute_error(predictions, y_valid)

    print(mae)

def print_cross_val_score(model, X, y):
    """ Prints the score of the model

    Args:
        y_valid (_type_): _description_
        mae (_type_): _description_

    Returns:
        _type_: _description_
    """
    scores = -1 * cross_val_score(model, X, y,
                                  cv=5, scoring="neg_mean_absolute_error", error_score="raise")

    print(scores)
    print(scores.mean())
