import numpy as np

from sklearn.linear_model import LinearRegression


# =====================================
# TRAINING DATA
# =====================================

temperature = np.array([
    28,
    29,
    30,
    35,
    32,
    31,
    34,
    27,
    36,
    33
])

water_usage = np.array([
    850,
    1200,
    950,
    1800,
    1350,
    1100,
    1450,
    900,
    1600,
    1250
])


# =====================================
# PREPARE FEATURES AND TARGET
# =====================================

X = temperature.reshape(-1, 1)

y = water_usage


# =====================================
# CREATE THE MODEL
# =====================================

model = LinearRegression()


# =====================================
# TRAIN THE MODEL
# =====================================

model.fit(X, y)


# =====================================
# BASIC PREDICTION
# =====================================

new_temperature = np.array([[37]])

prediction = model.predict(new_temperature)

print("Temperature:", new_temperature[0][0])
print("Predicted Water Usage:", prediction[0])


# =====================================
# CHALLENGE 1
# Predict water usage at 40°C
# =====================================

temperature_40 = np.array([[40]])

prediction_40 = model.predict(temperature_40)

print("\nPrediction at 40°C:")
print(prediction_40[0])


# =====================================
# CHALLENGE 2
# Predict multiple temperatures
# =====================================

new_temperatures = np.array([
    [25],
    [30],
    [35],
    [40]
])

predictions = model.predict(new_temperatures)

print("\nMultiple Predictions:")

for temperature_value, prediction_value in zip(
    new_temperatures,
    predictions
):
    print(
        temperature_value[0],
        "°C →",
        prediction_value
    )


# =====================================
# CHALLENGE 3
# Display model coefficient
# and intercept
# =====================================

print("\nModel Coefficient:")
print(model.coef_[0])

print("\nModel Intercept:")
print(model.intercept_)


# =====================================
# CHALLENGE 4
# Identify the ML components
# =====================================

# Feature: Temperature
# Target: Water Usage
# Model: Linear Regression
# Prediction: Estimated water usage


# =====================================
# BONUS
# Predict at 45°C
# =====================================

temperature_45 = np.array([[45]])

prediction_45 = model.predict(temperature_45)

print("\nPrediction at 45°C:")
print(prediction_45[0])

score = model.score(X, y)

print("\nModel R² Score:")
print(score)