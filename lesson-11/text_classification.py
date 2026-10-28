from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

texts = [
    "I love this product",
    "The product is excellent",
    "This application works perfectly",
    "I am very happy with the service",
    "The experience was wonderful",
    "The product is amazing",
    "I hate this product",
    "The application keeps crashing",
    "The service was terrible",
    "I am very disappointed",
    "This experience was awful",
    "The product is broken",
    "The package arrived today",
    "The order contains two items",
    "The meeting starts tomorrow",
    "I received the delivery",
    "The product is available in stores",
    "My appointment is scheduled for Monday"
]

labels = [
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "positive",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "negative",
    "neutral",
    "neutral",
    "neutral",
    "neutral",
    "neutral",
    "neutral"
]

X_train, X_test, y_train, y_test = train_test_split(
    texts,
    labels,
    test_size=0.33,
    random_state=42,
    stratify=labels
)

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Text Classification Model")
print("-------------------------")
print(f"Training examples: {len(X_train)}")
print(f"Testing examples: {len(X_test)}")
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

new_feedback = [
    "I really love this application",
    "The service is terrible",
    "My package arrived today"
]

predicted_labels = model.predict(new_feedback)

print("\nNew Feedback Predictions:")

for feedback, label in zip(new_feedback, predicted_labels):
    print(f"Feedback: {feedback}")
    print(f"Predicted category: {label}")
    print()