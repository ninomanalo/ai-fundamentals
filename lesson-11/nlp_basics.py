
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

text = """
Artificial intelligence helps computers understand human language.
AI systems can analyze text, answer questions, and summarize documents.
"""

print("Original Text:")
print(text)

cleaned_text = text.lower()
cleaned_text = re.sub(r"[^a-z\s]", "", cleaned_text)
cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

print("\nCleaned Text:")
print(cleaned_text)

words = word_tokenize(cleaned_text)

print("\nTokens:")
print(words)

sentences = sent_tokenize(text)

print("\nSentences:")
print(sentences)

stop_words = set(stopwords.words("english"))

filtered_words = [
    word for word in words
    if word not in stop_words
]

print("\nFiltered Words:")
print(filtered_words)

stemmer = PorterStemmer()

stemmed_words = [
    stemmer.stem(word)
    for word in filtered_words
]

print("\nStemmed Words:")
print(stemmed_words)

print("\nWord Count:")
print(len(words))

print("\nFiltered Word Count:")
print(len(filtered_words))
