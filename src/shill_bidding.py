
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

shill_bidding_data = pd.read_csv("data/shill-bidding.csv")

shill_bidding_data = shill_bidding_data.drop(["Record_ID","Bidder_ID"],axis=1)
X = shill_bidding_data

y= shill_bidding_data.Starting_Price_Average
#print(X)
#print(shill_bidding_data)
#print(Y)
# 80% traing 20% validation
#print(y)
X_train, X_valid, y_train, y_valid = train_test_split(X,y, train_size=0.8,test_size=0.2)

imputer = SimpleImputer(strategy="constant")

print(X_train)
encode_X_train = imputer.fit_transform(X_train)

encode_y_train = imputer.transform(y_train)

#encode_y_valid = imputer.transform(y_valid)
#print(encode_X_train)



model = RandomForestRegressor(n_estimators=250, random_state=0)
"""""
model.fit(X,y)

prediction = model.predict(X_valid)

mae = mean_absolute_error(prediction, y_valid)

print(mae)
"""