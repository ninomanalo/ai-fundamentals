import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")

text = """
The AI systems are learning quickly.
The developers were building intelligent applications.
These applications process customer reviews and customer feedback.
"""

normalized_text = text.lower()
normalized_text = re.sub(r"[^a-z\s]", "", normalized_text)
normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

print("Normalized Text:")
print(normalized_text)

words = word_tokenize(normalized_text)

print("\nOriginal Words:")
print(words)

lemmatizer = WordNetLemmatizer()

lemmatized_words = [
    lemmatizer.lemmatize(word, pos="v")
    for word in words
]

print("\nLemmatized Words:")
print(lemmatized_words)

bigrams = list(ngrams(words, 2))

print("\nBigrams:")
print(bigrams)

trigrams = list(ngrams(words, 3))

print("\nTrigrams:")
print(trigrams)

print("\nWord Count:")
print(len(words))

print("\nBigram Count:")
print(len(bigrams))

print("\nTrigram Count:")
print(len(trigrams))