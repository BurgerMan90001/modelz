from sklearn.model_selection import cross_val_score

def print_score(model, data):
    pass

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
