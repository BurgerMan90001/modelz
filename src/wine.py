# import required modules
from tpot import TPOTRegressor
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np

# load boston dataset
X, y = load_wine(return_X_y=True)

# divide the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .25)

# define TpotRegressor 
reg = TPOTRegressor(verbose=3, population_size=5, generations=1, random_state=35)

# fit the regressor on training data
reg.fit(X_train, y_train)

# print the results on test data
print(reg.score(X_test, y_test))


reg.export('top_wine.py')