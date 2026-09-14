import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression


# =====================================
# RAW DATA
# =====================================

data = {
    "temperature": [
        25, 26, 27, 28, 29,
        30, 31, 32, 33, 34,
        35, 36, 37, 38, 39
    ],
    "humidity": [
        40, 42, 45, 48, 50,
        52, 55, 58, 60, 62,
        65, 67, 70, 72, 75
    ],
    "servers": [
        100, 105, 110, 115, 120,
        125, 130, 135, 140, 145,
        150, 155, 160, 165, 170
    ],
    "water_usage": [
        700, 760, 820, 850, 900,
        980, 1050, 1100, 1200, 1250,
        1320, 1400, 1480, 1550, 1640
    ]
}


df = pd.DataFrame(data)


print("Original Data:")
print(df)


# =====================================
# FEATURE ENGINEERING
# =====================================

df["temperature_squared"] = df["temperature"] ** 2

df["servers_per_degree"] = (
    df["servers"] / df["temperature"]
)


print("\nData After Feature Engineering:")
print(df)


# =====================================
# FEATURES AND TARGET
# =====================================

features = [
    "temperature",
    "humidity",
    "servers",
    "temperature_squared",
    "servers_per_degree"
]

X = df[features]

y = df["water_usage"]


print("\nFeatures:")
print(X)

print("\nTarget:")
print(y)


# =====================================
# CREATE MODEL
# =====================================

model = LinearRegression()


# =====================================
# TRAIN MODEL
# =====================================

model.fit(X, y)


# =====================================
# MAKE A PREDICTION
# =====================================

new_data = pd.DataFrame({
    "temperature": [40],
    "humidity": [78],
    "servers": [175]
})


new_data["temperature_squared"] = (
    new_data["temperature"] ** 2
)

new_data["servers_per_degree"] = (
    new_data["servers"] /
    new_data["temperature"]
)


prediction = model.predict(new_data[features])


print("\nPredicted Water Usage:")
print(prediction[0])


# =====================================
# CHALLENGE 1
# Display model coefficients
# =====================================

print("\nModel Coefficients:")

for feature, coefficient in zip(
    features,
    model.coef_
):
    print(feature, ":", coefficient)


# =====================================
# CHALLENGE 2
# Predict another data center
# =====================================

another_data = pd.DataFrame({
    "temperature": [35],
    "humidity": [70],
    "servers": [150]
})


another_data["temperature_squared"] = (
    another_data["temperature"] ** 2
)

another_data["servers_per_degree"] = (
    another_data["servers"] /
    another_data["temperature"]
)


another_prediction = model.predict(
    another_data[features]
)


print("\nPrediction for Second Data Center:")
print(another_prediction[0])


# =====================================
# CHALLENGE 3
# Compare predictions
# =====================================

print("\nPrediction Comparison:")

print("40°C / 175 servers:", prediction[0])

print("35°C / 150 servers:", another_prediction[0])


# =====================================
# CHALLENGE 4
# Identify your ML components
# =====================================

# Features:
# Target:
# Model:
# Engineered Feature 1:
# Engineered Feature 2: