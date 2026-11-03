import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


feedback = [
    "I love this product",
    "The service is excellent",
    "This application is amazing",
    "I am very happy with the experience",
    "The product works perfectly",
    "Customer support was wonderful",
    "I hate this product",
    "The service is terrible",
    "This application keeps crashing",
    "I am very disappointed",
    "The experience was awful",
    "Customer support was unhelpful",
    "The package arrived today",
    "I received my order yesterday",
    "The product is available in stores",
    "My appointment is scheduled tomorrow",
    "The delivery arrived this morning",
    "The meeting starts at nine"
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


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


cleaned_feedback = [
    clean_text(text)
    for text in feedback
]


X_train, X_test, y_train, y_test = train_test_split(
    cleaned_feedback,
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


print("CUSTOMER FEEDBACK ANALYZER")
print("==========================")

print(f"\nTraining Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

print(f"\nModel Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))


print("\nTEST YOUR OWN CUSTOMER FEEDBACK")
print("================================")

while True:

    user_feedback = input(
        "\nEnter customer feedback (or type 'exit'): "
    )

    if user_feedback.lower() == "exit":
        print("Thank you for using the analyzer!")
        break

    cleaned_text = clean_text(user_feedback)

    if not cleaned_text:
        print("Please enter valid text.")
        continue

    sentiment = model.predict([cleaned_text])[0]

    print(f"\nOriginal Feedback: {user_feedback}")
    print(f"Cleaned Feedback: {cleaned_text}")
    print(f"Predicted Sentiment: {sentiment.upper()}")