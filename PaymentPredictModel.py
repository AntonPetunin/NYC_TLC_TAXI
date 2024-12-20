# %%
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os

# %% [markdown]
# ## Data Pre-Processing

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

dataset_dir = os.path.join(os.environ["BIG_DATA"], "input")

# %%
df = pd.read_parquet(f"{dataset_dir}\\fhvhv_tripdata_2021-12.parquet", engine="pyarrow")
print(df.head())

# %%
df['driver_pay_per_mile'] = df['driver_pay'] / df['trip_miles']
df['trip_duration'] = (df['dropoff_datetime'] - df['pickup_datetime']).dt.total_seconds() / 60
df['driver_pay_per_minute'] = df['driver_pay'] / df['trip_duration']
print(df.head(3))

# %%
df.isna().sum()

# %%
df = df.drop(columns=['originating_base_num', 'on_scene_datetime'])
mean_dppm = df['driver_pay_per_mile'].mean()
# Fill null values with the mean
df['driver_pay_per_mile'].fillna(mean_dppm, inplace=True)

# %%
df.isna().sum()

# %%
data = df.iloc[:2000000].copy()
data.shape

# %% [markdown]
# ## MODEL 1 : Neural Network

# %%
import tensorflow as tf
from tensorflow import keras

# %%
# Neural Network
model1 = keras.Sequential([
    keras.layers.Flatten(input_shape=(5,)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='linear')
])

# training data
training_data = data[:1400000]
target_data = data[1400000:]
model1.compile(optimizer='adam', loss='mean_squared_error')

# m1 = model1.fit(training_data[['trip_miles', 'trip_duration', 'bcf', 'tips', 'congestion_surcharge']],
#                 training_data['driver_pay'], epochs=10)
predictions = model1.predict(target_data[['trip_miles', 'trip_duration', 'bcf', 'tips', 'congestion_surcharge']])
print(predictions[:5])
print(target_data['driver_pay'][:5])

# %%
print(model1.evaluate(target_data[['trip_miles', 'trip_duration', 'bcf', 'tips', 'congestion_surcharge']],
                      target_data['driver_pay']))

# %%
# model1.save('m1.keras')
m1 = tf.keras.models.load_model('m1.keras')
y_pred_m1 = m1.predict(target_data[['trip_miles', 'trip_duration', 'bcf', 'tips', 'congestion_surcharge']])
print(y_pred_m1)

# %%
from sklearn.metrics import f1_score

predictions = y_pred_m1
# Discretize predictions into binary classes
threshold = np.median(predictions)
binary_predictions = (predictions > threshold).astype(int)

# Discretize actual values into binary classes
threshold_actual = np.median(target_data['driver_pay'])
binary_actual = (target_data['driver_pay'] > threshold_actual).astype(int)

# Calculate F1 score
f1 = f1_score(binary_actual, binary_predictions)

print("F1 Score:", f1)

# %%

a = np.mean(target_data['driver_pay'][:5])
b = np.mean(predictions[:5])
print("Average Driver Pay(Actual):", a, "\nAverage Driver Pay(Predicted):", b, "\nDifference: ", b - a, )

# %%
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

rmse = np.sqrt(mean_squared_error(target_data['driver_pay'][:5], predictions[:5]))
mse = mean_squared_error(target_data['driver_pay'][:5], predictions[:5])
r2 = r2_score(target_data['driver_pay'][:5], predictions[:5])

print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2) Score:", r2)

# %%
import matplotlib.pyplot as plt

# Assuming 'predictions' contains the predicted values and 'target_data' contains the actual values
plt.figure(figsize=(10, 6))

# Plotting predicted values in blue
plt.scatter(range(5), predictions[:5], color='blue', label='Predicted', alpha=0.5)
# Plotting target data in red
plt.scatter(range(5), target_data['driver_pay'][:5], color='red', label='Actual', alpha=0.5)

plt.title('Predicted vs Actual Values')
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.legend()
plt.show()

# %% [markdown]
# ## MODEL 2 : Random Forest

# %%
data2 = data[:2000000].copy()
data2.shape

# %%
feature_list = ['trip_miles', 'trip_duration', 'bcf', 'tips', 'congestion_surcharge', 'driver_pay']
features = data2[feature_list]
labels = features['driver_pay']
features = features.drop('driver_pay', axis=1)
print(features.head())

# %%
from sklearn.model_selection import train_test_split

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3,
                                                                            random_state=42)

# %%
from sklearn.ensemble import RandomForestRegressor
import time

# Parallelization and reducing tree depth
start_time = time.time()
model2 = RandomForestRegressor(n_estimators=1000, max_depth=15, n_jobs=-1, random_state=119, verbose=2)
model2.fit(train_features, train_labels)
end_time = time.time()
# Calculate training time
training_time = end_time - start_time
print("Training time:", training_time, "seconds")

# %%
import matplotlib.pyplot as plt

y_pred_m2 = model2.predict(test_features)
print(y_pred_m2[:5])

# %%
# Predict the last 5 values of driver pay
predicted_values = y_pred_m2[-5:]
actual_values = test_labels[-5:]

# Scatter graph: Actual vs Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(range(5), predicted_values, color='blue', label='Predicted vs Actual', alpha=0.5)
plt.scatter(range(5), actual_values, color='red', label='Perfect Fit', alpha=0.3)
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

# %%
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))
mse = mean_squared_error(actual_values, predicted_values)
r2 = r2_score(actual_values, predicted_values)

print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Squared Error (MSE):", mse)
print("R2  score: ", r2)

# %% [markdown]
# ## MODEL 3 - LSTM

# %%
weather = pd.read_csv(f"{dataset_dir}\\nyc 2021-01-01 to 2021-12-31.csv")
weather.isnull().sum()

# %%
weather.drop(columns=['severerisk', 'windgust'], inplace=True)
weather.info()

# %%
weather_object_columns = weather.select_dtypes(include=['object']).columns
weather_object_columns

# %%
data.info()

# %%
# for x in weather_object_columns:
#     if x != 'datetime':
#         weather[x] = label_encoder.fit_transform(weather[x])
# data['request_date'] = data['request_datetime'].dt.date
# weather['date'] = pd.to_datetime(weather['datetime']).dt.date

# # Merge the two DataFrames on the 'request_date' column from df1 and 'date' column from df2
# merged_df = pd.merge(data, weather, left_on='request_date', right_on='date', how='inner')

# # Drop the redundant 'date' column if necessary
# # merged_df.drop(columns=['date'], inplace=True)

# # Print the merged DataFrame
# print(merged_df)

from sklearn.preprocessing import LabelEncoder

# Encode object columns in the weather DataFrame
label_encoder = LabelEncoder()
for x in weather_object_columns:
    if x != 'datetime':
        weather[x] = label_encoder.fit_transform(weather[x])

data['request_date'] = data['request_datetime'].dt.date
weather['date'] = pd.to_datetime(weather['datetime']).dt.date
merged_df = pd.merge(data2, weather, left_on='request_date', right_on='date', how='inner')
print(merged_df)

# %%
features = merged_df.columns
features

# %%
object_columns = data.select_dtypes(include=['object']).columns
label_encoder = LabelEncoder()
for x in object_columns:
    merged_df[x] = label_encoder.fit_transform(merged_df[x])

# %%
merged_df.info()

# %%
merged_df = merged_df.drop(
    columns=['datetime', 'date', 'request_datetime', 'request_date', 'pickup_datetime', 'dropoff_datetime', 'date'],
    axis=1)

# %%
merged_df.info()

# %%
import seaborn as sns

correlation_matrix = merged_df.corr()

# Plotting the correlation matrix using a heatmap
plt.figure(figsize=(50, 50))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.savefig('correlation_heatmap.png')
plt.show()

# %%
correlation_with_driver_pay = merged_df.corr()['driver_pay'].sort_values(ascending=False)
print(correlation_with_driver_pay)

# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Check if TensorFlow is using GPU
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

# %%
# Preprocessing
# driver_pay               1.000000
# base_passenger_fare      0.948162
# bcf                      0.943677
# trip_miles               0.888677
# trip_duration            0.876624
# trip_time                0.876201
# sales_tax                0.731104
# tolls                    0.480044
# tips                     0.329006
# airport_fee              0.328361
# congestion_surcharge     0.149102
# DOLocationID             0.111447
# windspeed                0.043215
# temp                     0.040197
# precipprob               0.035741
# precip                   0.035741
# feelslike                0.035005


# Assuming you want to use these columns for prediction
features = ['base_passenger_fare', 'bcf', 'trip_miles', 'trip_duration',
            'trip_time', 'sales_tax', 'tolls', 'tips', 'airport_fee',
            'congestion_surcharge', 'DOLocationID', 'windspeed', 'temp',
            'precipprob']

target_variable = 'driver_pay'
# Select features and target variable
X = merged_df[features].values
y = merged_df[target_variable].values

# Splitting the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshaping the input data for LSTM
# LSTM expects input data to be 3D in the form of (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# %%
# Building the LSTM model
model3 = Sequential()
model3.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model3.add(Dropout(0.2))
model3.add(LSTM(units=50, return_sequences=True))
model3.add(Dropout(0.2))
model3.add(LSTM(units=50))
model3.add(Dropout(0.2))
model3.add(Dense(units=1))

# Compiling the model
model3.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
history1 = model3.fit(X_train, y_train, epochs=9, batch_size=32, validation_data=(X_test, y_test))

# Evaluating the model
loss1 = model3.evaluate(X_test, y_test)
print(f'Test Loss - LSTM: {loss1}')

# %%
# Plotting the training and validation loss
plt.plot(history1.history['loss'], label='Training Loss')
plt.plot(history1.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# %%
y_pred_m3 = model3.predict(X_test)
print(y_pred_m3[:5])

# %%
from sklearn.metrics import f1_score

threshold = np.median(y_pred_m3)
threshold_actual = np.median(y_test)
y_pred_classes = (y_pred_m3 > threshold).astype(int)
y_test_classes = (y_test > threshold_actual).astype(int)
f1_lstm = f1_score(y_test_classes, y_pred_classes)

# threshold = np.median(y_pred_m3)
# binary_predictions = (predictions > threshold).astype(int)

# # Discretize actual values into binary classes
# threshold_actual = np.median(target_data['driver_pay'])
# binary_actual = (target_data['driver_pay'] > threshold_actual).astype(int)
# f1_lstm = f1_score(binary_actual, binary_predictions)

print("F1 score: LSTM", f1_lstm)

# %%
plt.figure(figsize=(10, 6))
plt.scatter(range(5), y_pred_m3[:5], color='blue', label='Predicted', alpha=0.5)
plt.scatter(range(5), y_test[:5], color='red', label='Actual', alpha=0.5)
plt.xlabel('Index')
plt.ylabel('Driver Pay')
plt.title('Predicted vs Actual Driver Pay')
plt.legend()
plt.show()

# %% [markdown]
# ## MODEL 4 - LSTM-Bidirectional

# %%
from tensorflow.keras.layers import Bidirectional

# %%
model4 = Sequential([
    Bidirectional(LSTM(30, activation='relu', return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])),
    Bidirectional(LSTM(30, activation='relu')),
    Dropout(0.2),
    Dense(1)
])

# Compiling the model
model4.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
history2 = model4.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluating the model
loss2 = model4.evaluate(X_test, y_test)
print("Test Loss:", loss2)

# %%
# Plotting the training and validation loss
plt.plot(history2.history['loss'], label='Training Loss')
plt.plot(history2.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# %%
y_pred_m4 = model4.predict(X_test)
print(y_pred_m4)

# %%
from sklearn.metrics import f1_score

# threshold = 0.6
# y_pred_classes = (y_pred_m4 > threshold).astype(int)
# y_test_classes = (y_test > threshold).astype(int)
# f1_2 = f1_score(y_test_classes, y_pred_classes)
threshold2 = np.median(y_pred_m4)
threshold_actual2 = np.median(y_test)
y_pred_classes2 = (y_pred_m4 > threshold2).astype(int)
y_test_classes2 = (y_test > threshold_actual2).astype(int)
f1_lstm2 = f1_score(y_test_classes2, y_pred_classes2)

# threshold = np.median(y_pred_m3)
# binary_predictions = (predictions > threshold).astype(int)

# # Discretize actual values into binary classes
# threshold_actual = np.median(target_data['driver_pay'])
# binary_actual = (target_data['driver_pay'] > threshold_actual).astype(int)
# f1_lstm = f1_score(binary_actual, binary_predictions)

print("F1 score: LSTM-Bidirectional", f1_lstm2)

# print("F1 score: - LSTM-BIDIRECTIONAL", f1_2)

# %%
plt.figure(figsize=(10, 6))
plt.scatter(range(5), y_pred_m4[:5], color='blue', label='Predicted', alpha=0.5)
plt.scatter(range(5), y_test[:5], color='red', label='Actual', alpha=0.5)
plt.xlabel('Index')
plt.ylabel('Driver Pay')
plt.title('Predicted vs Actual Driver Pay')
plt.legend()
plt.show()

# %% [markdown]
# ## MODEL 5 -


# %%
from keras.layers import GRU

# %%
model_lstm_gru = Sequential()
model_lstm_gru.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model_lstm_gru.add(Dropout(0.2))
model_lstm_gru.add(GRU(units=50, return_sequences=True))
model_lstm_gru.add(Dropout(0.2))
model_lstm_gru.add(LSTM(units=50))
model_lstm_gru.add(Dropout(0.2))
model_lstm_gru.add(Dense(units=1))
model_lstm_gru.compile(optimizer='adam', loss='mean_squared_error')

# %%
# Training the model
history_lstm_gru = model_lstm_gru.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# %%
# Evaluation for LSTM-GRU model
loss_lstm_gru = model_lstm_gru.evaluate(X_test, y_test)
print("Test Loss for LSTM-GRU:", loss_lstm_gru)

# Plotting the training and validation loss for LSTM-GRU
plt.plot(history_lstm_gru.history['loss'], label='Training Loss')
plt.plot(history_lstm_gru.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss for LSTM-GRU')
plt.legend()
plt.show()

# %%
y_pred_lstm_gru = model_lstm_gru.predict(X_test)
print(y_pred_lstm_gru[:5])

# %%
# Calculate F1 score for LSTM-GRU
# threshold = 0.9
# y_pred_classes_lstm_gru = (y_pred_lstm_gru > threshold).astype(int)
# y_test_classes_lstm_gru = (y_test > threshold).astype(int)
# f1_lstm_gru = f1_score(y_test_classes_lstm_gru, y_pred_classes_lstm_gru)

# print("F1 score for LSTM-GRU:", f1_lstm_gru)

from sklearn.metrics import f1_score

threshold3 = np.median(y_pred_lstm_gru)
threshold_actual3 = np.median(y_test)
y_pred_classes3 = (y_pred_lstm_gru > threshold3).astype(int)
y_test_classes3 = (y_test > threshold_actual3).astype(int)
f1_lstm_gru = f1_score(y_test_classes3, y_pred_classes3)
print("F1 score: LSTM-GRU", f1_lstm_gru)

# %%
plt.figure(figsize=(10, 6))
plt.scatter(range(5), y_pred_lstm_gru[:5], color='blue', label='Predicted', alpha=0.5)
plt.scatter(range(5), y_test[:5], color='red', label='Actual', alpha=0.5)
plt.xlabel('Index')
plt.ylabel('Driver Pay')
plt.title('Predicted vs Actual Driver Pay')
plt.legend()
plt.show()
