
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


shill_bidding_data = pd.read_csv("data/shill-bidding.csv")

shill_bidding_data = shill_bidding_data.drop(["Record_ID","Bidder_ID"],axis=1)

X = shill_bidding_data.drop("Starting_Price_Average",axis=1)

y= shill_bidding_data.Starting_Price_Average


# 80% traing 20% validation
#print(y)
X_train, X_valid, y_train, y_valid = train_test_split(X,y, train_size=0.8,test_size=0.2)

imputer = SimpleImputer(strategy="mean")


encode_X_train = pd.DataFrame(imputer.fit_transform(X_train))
encode_X_valid = pd.DataFrame(imputer.fit_transform(X_valid))

encode_X_train.columns = X.columns
encode_X_valid.columns = X.columns

#print(encode_X_valid)
#print(y_train)
#encode_y_train = imputer.transform(y_train)

#encode_y_valid = imputer.transform(y_valid)
#print(encode_X_train)



#model = RandomForestRegressor(n_estimators=250, random_state=0)
model = XGBRegressor(n_estimators=1000,learning_rate=0.1,random_state=0)

model.fit(X_train,y_train)

prediction = model.predict(X_valid)

#print(prediction)
mae = mean_absolute_error(prediction, y_valid)


def get_percent(y_valid, mae):
    return ((y_valid.mean()- mae) / y_valid.mean()) * 100


print(mae)
print(get_percent(y_valid,mae))


