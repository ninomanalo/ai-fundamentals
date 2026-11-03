import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
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
# CREATE TARGET
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
# FEATURE SCALING
# =====================================

scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


print("\nScaled Training Data:")
print(X_train_scaled)


print("\nScaled Testing Data:")
print(X_test_scaled)


# =====================================
# CREATE MODEL
# =====================================

model = LogisticRegression()


# =====================================
# TRAIN MODEL
# =====================================

model.fit(
    X_train_scaled,
    y_train
)


# =====================================
# PREDICTIONS
# =====================================

y_pred = model.predict(
    X_test_scaled
)


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
# PRECISION
# =====================================

precision = precision_score(
    y_test,
    y_pred,
    pos_label="High"
)

print("\nPrecision:")
print(precision)


# =====================================
# RECALL
# =====================================

recall = recall_score(
    y_test,
    y_pred,
    pos_label="High"
)

print("\nRecall:")
print(recall)


# =====================================
# F1 SCORE
# =====================================

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="High"
)

print("\nF1 Score:")
print(f1)


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
# Compare all evaluation metrics
# =====================================

print("\nEvaluation Summary:")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# =====================================
# CHALLENGE 2
# Predict a new data center
# =====================================

new_center = pd.DataFrame({
    "temperature": [38],
    "humidity": [74],
    "servers": [170]
})


new_center_scaled = scaler.transform(
    new_center
)


new_prediction = model.predict(
    new_center_scaled
)


print("\nNew Data Center Prediction:")
print(new_prediction[0])


# =====================================
# CHALLENGE 3
# Get prediction probability
# =====================================

new_probability = model.predict_proba(
    new_center_scaled
)


print("\nNew Data Center Probabilities:")

for class_name, probability in zip(
    model.classes_,
    new_probability[0]
):
    print(
        class_name,
        ":",
        probability
    )


# =====================================
# CHALLENGE 4
# Find the number of correct predictions
# =====================================

correct_predictions = np.sum(
    y_test.to_numpy() == y_pred
)


print("\nCorrect Predictions:")
print(correct_predictions)
