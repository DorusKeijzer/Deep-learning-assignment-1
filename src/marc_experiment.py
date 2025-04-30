import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from scipy.io import loadmat
from sklearn.metrics import mean_absolute_error, mean_squared_error
data = loadmat("data/Xtrain.mat")

print(data["Xtrain"])


scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data["Xtrain"])

print(scaled_data)

window_size = 5
mae_list = []
mse_list = []
for i in range(1, 10):
    window_size = i*10

    x, y = [], []

    target_values = np.arange(window_size, len(data["Xtrain"]))

    for i in range(window_size, len(scaled_data)):
        x.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i, 0])

    x = np.array(x)
    y = np.array(y)

    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        x, y, target_values, test_size=0.2, shuffle=False
    )

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=128))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(X_train, y_train, epochs=40, batch_size=32, validation_split=0.1)

    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions).flatten()
    y_test = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()


    mae_list.append(mean_absolute_error(y_test, predictions))
    mse_list.append(mean_squared_error(y_test, predictions))



print("MAE: ", mae_list)
print("MSE: ", mse_list)
