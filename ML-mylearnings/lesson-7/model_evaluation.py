import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =====================================
# DATA
# =====================================

temperature = np.array([
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44
])

water_usage = np.array([
    700,
    760,
    820,
    850,
    900,
    980,
    1050,
    1100,
    1200,
    1250,
    1320,
    1400,
    1480,
    1550,
    1640,
    1720,
    1800,
    1900,
    1980,
    2070
])


# =====================================
# PREPARE FEATURES AND TARGET
# =====================================

X = temperature.reshape(-1, 1)

y = water_usage


# =====================================
# TRAIN / TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Training Samples:")
print(len(X_train))

print("\nTesting Samples:")
print(len(X_test))


# =====================================
# CREATE MODEL
# =====================================

model = LinearRegression()


# =====================================
# TRAIN MODEL
# =====================================

model.fit(X_train, y_train)


# =====================================
# MAKE TEST PREDICTIONS
# =====================================

y_pred = model.predict(X_test)


print("\nActual Values:")
print(y_test)

print("\nPredicted Values:")
print(y_pred)


# =====================================
# MEAN ABSOLUTE ERROR
# =====================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

print("\nMean Absolute Error:")
print(mae)


# =====================================
# MEAN SQUARED ERROR
# =====================================

mse = mean_squared_error(
    y_test,
    y_pred
)

print("\nMean Squared Error:")
print(mse)


# =====================================
# ROOT MEAN SQUARED ERROR
# =====================================

rmse = np.sqrt(mse)

print("\nRoot Mean Squared Error:")
print(rmse)


# =====================================
# R² SCORE
# =====================================

r2 = r2_score(
    y_test,
    y_pred
)

print("\nR² Score:")
print(r2)


# =====================================
# CHALLENGE 1
# Predict a completely new temperature
# =====================================

new_temperature = np.array([
    [45]
])

new_prediction = model.predict(
    new_temperature
)

print("\nPrediction at 45°C:")
print(new_prediction[0])


# =====================================
# CHALLENGE 2
# Compare actual vs predicted
# =====================================

print("\nActual vs Predicted:")

for actual, predicted in zip(
    y_test,
    y_pred
):
    print(
        "Actual:",
        actual,
        "| Predicted:",
        predicted
    )


# =====================================
# CHALLENGE 3
# Find the largest prediction error
# =====================================

errors = np.abs(
    y_test - y_pred
)

largest_error = np.max(errors)

print("\nLargest Prediction Error:")
print(largest_error)


# =====================================
# CHALLENGE 4
# Find average prediction error
# =====================================

average_error = np.mean(errors)

print("\nAverage Prediction Error:")
print(average_error)


# =====================================
# CHALLENGE 5
# Identify the model
# =====================================

# Model:
# Feature:
# Target:
# Training Data:
# Testing Data: