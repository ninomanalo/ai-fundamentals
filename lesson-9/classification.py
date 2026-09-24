import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# =====================================
# DATA
# =====================================

data = {
    "temperature": [
        25, 26, 27, 28, 29,
        30, 31, 32, 33, 34,
        35, 36, 37, 38, 39,
        40, 41, 42, 43, 44
    ],
    "humidity": [
        40, 42, 45, 48, 50,
        52, 55, 58, 60, 62,
        65, 67, 70, 72, 75,
        78, 80, 82, 85, 88
    ],
    "servers": [
        100, 105, 110, 115, 120,
        125, 130, 135, 140, 145,
        150, 155, 160, 165, 170,
        175, 180, 185, 190, 195
    ],
    "water_usage": [
        700, 760, 820, 850, 900,
        980, 1050, 1100, 1200, 1250,
        1320, 1400, 1480, 1550, 1640,
        1720, 1800, 1900, 1980, 2070
    ]
}


df = pd.DataFrame(data)


# =====================================
# CREATE TARGET CATEGORY
# =====================================

df["usage_category"] = np.where(
    df["water_usage"] > 1000,
    "High",
    "Normal"
)


print("Dataset:")
print(df)


# =====================================
# FEATURES AND TARGET
# =====================================

features = [
    "temperature",
    "humidity",
    "servers"
]

X = df[features]

y = df["usage_category"]


print("\nFeatures:")
print(X)

print("\nTarget:")
print(y)


# =====================================
# TRAIN / TEST SPLIT
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTraining Samples:")
print(len(X_train))

print("\nTesting Samples:")
print(len(X_test))


# =====================================
# CREATE CLASSIFICATION MODEL
# =====================================

model = LogisticRegression()


# =====================================
# TRAIN MODEL
# =====================================

model.fit(X_train, y_train)


# =====================================
# MAKE PREDICTIONS
# =====================================

y_pred = model.predict(X_test)


print("\nActual Classes:")
print(y_test.to_numpy())

print("\nPredicted Classes:")
print(y_pred)


# =====================================
# ACCURACY
# =====================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(accuracy)


# =====================================
# CONFUSION MATRIX
# =====================================

matrix = confusion_matrix(
    y_test,
    y_pred,
    labels=["Normal", "High"]
)

print("\nConfusion Matrix:")
print(matrix)


# =====================================
# CLASSIFICATION REPORT
# =====================================

report = classification_report(
    y_test,
    y_pred
)

print("\nClassification Report:")
print(report)


# =====================================
# CHALLENGE 1
# Predict a new data center
# =====================================

new_center = pd.DataFrame({
    "temperature": [36],
    "humidity": [70],
    "servers": [160]
})


new_prediction = model.predict(
    new_center
)


print("\nNew Data Center Prediction:")
print(new_prediction[0])


# =====================================
# CHALLENGE 2
# Get prediction probability
# =====================================

probabilities = model.predict_proba(
    new_center
)

print("\nPrediction Probabilities:")

for class_name, probability in zip(
    model.classes_,
    probabilities[0]
):
    print(
        class_name,
        ":",
        probability
    )


# =====================================
# CHALLENGE 3
# Predict multiple data centers
# =====================================

multiple_centers = pd.DataFrame({
    "temperature": [28, 32, 36, 40],
    "humidity": [48, 58, 70, 78],
    "servers": [115, 135, 160, 175]
})


multiple_predictions = model.predict(
    multiple_centers
)


print("\nMultiple Predictions:")

for index, prediction in enumerate(
    multiple_predictions
):
    print(
        "Data Center",
        index + 1,
        "→",
        prediction
    )


# =====================================
# CHALLENGE 4
# Count each predicted class
# =====================================

predicted_counts = pd.Series(
    multiple_predictions
).value_counts()


print("\nPredicted Class Counts:")
print(predicted_counts)


