
from sklearn.model_selection import cross_val_score

from xgboost import XGBRegressor
import pandas as pd


data = pd.read_csv("data/Gym_Progress_Dataset.csv")

X = data.drop(["Day"],axis=1)
y= X.pop("Workout_Duration_min")


model = XGBRegressor(learning_rate=0.1,
                     random_state=0)


scores= -1 * cross_val_score(model, X,y, cv=5,
                             scoring="neg_mean_absolute_error", error_score="raise")


print(scores)
print(scores.mean())
