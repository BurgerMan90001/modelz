
import pandas as pd

from xgboost import XGBClassifier

from util.pipelines import define_pipeline
from util.scoring import print_cross_val_score
from util.scoring import print_score_mae


shill_bidding_data = pd.read_csv("data/shill-bidding.csv")

y_col = "Class"

X = shill_bidding_data.drop([y_col, "Bidder_ID", "Auction_ID"],axis=1)
y= shill_bidding_data[y_col]


model = XGBClassifier(n_estimators=250,
                     learning_rate=0.1,
                     random_state=0)


# non-numerical columns
categorical_cols = [col for col in X.columns if (X[col].dtype == "object")]
# numerical columns
numerical_cols = [col for col in X.columns if X[col].dtype in ['int64', 'float64']]

pipeline = define_pipeline(model, numerical_cols, categorical_cols)

print_cross_val_score(pipeline,X,y)
