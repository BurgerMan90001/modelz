
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import keras
import tensorflow as tf
#from sklearn.preprocessing import OrdinalEncoder
test = OneHotEncoder()


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
