from sklearn.model_selection import cross_val_score

def get_percent(mae, y_valid):
    """Returns the accuracy of the mean absolute error

    Args:
        mae (_type_): _description_

    Returns:
        _type_: _description_
    """
    return ((y_valid.mean()- mae) / y_valid.mean()) * 100

def print_cross_val_score(model, X, y, y_valid):
    """ Prints the score of the model

    Args:
        y_valid (_type_): _description_
        mae (_type_): _description_

    Returns:
        _type_: _description_
    """
    scores = -1 * cross_val_score(model, X, y,
                                  cv=5, scoring="neg_mean_absolute_error")

    print(scores)
    print(get_percent(scores.mean(), y_valid))

