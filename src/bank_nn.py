
import pandas as pd

from keras import Sequential, layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer


bank_data = pd.read_csv("data/bank-full.csv", delimiter=";")

y_col = "y"

X = bank_data.drop([y_col, "month","poutcome", "day"],axis=1)

y = bank_data[y_col]

# Manually encode target data
label_encoder = LabelEncoder()
oh_encoder = OneHotEncoder()

X_train, X_valid, y_train, y_valid = train_test_split(X,y, train_size=0.8,test_size=0.2)

# encode y values
y_train = label_encoder.fit_transform(y_train) # pyright: ignore[reportCallIssue, reportArgumentType]
y_valid = label_encoder.transform(y_valid)

# non-numerical columns
categorical_cols = [col for col in X_train.columns if (X_train[col].dtype == "object") 
                    and (X_train [col].nunique() < 15)]

# numerical columns
numerical_cols = [col for col in X_train.columns if X_train[col].dtype in ['int64', 'float64']]

preprocessor = make_column_transformer(
    (StandardScaler(), numerical_cols),
    (OneHotEncoder(), categorical_cols),
)

X_train = preprocessor.fit_transform(X_train)
X_valid = preprocessor.transform(X_valid)

input_shape = [X_train.shape[1]]


layers = [
    layers.BatchNormalization(input_shape=input_shape),
    
    layers.Dense(512,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.3),
    
    
    layers.Dense(256,activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(rate=0.3),
    
    # relu for classification output
    layers.Dense(1,activation='sigmoid')
]
model = Sequential(layers=layers)

early_stopping = callbacks.EarlyStopping(patience=5,
                                      min_delta=0.0001,restore_best_weights=True)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy']
    
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid,y_valid),
    callbacks=[early_stopping],
    batch_size=256,
    epochs=50,
)


# needs notebook
#history_df = pd.DataFrame(history.history)
#history_df.loc[:, ['loss', 'val_loss']].plot()