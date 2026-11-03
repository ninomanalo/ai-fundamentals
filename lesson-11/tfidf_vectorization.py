from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

documents = [
    "Artificial intelligence helps businesses automate tasks",
    "Machine learning helps computers learn from data",
    "AI automation improves business productivity",
    "Businesses use artificial intelligence for automation",
    "Machine learning is an important part of AI"
]

print("Original Documents:")
for document in documents:
    print(document)

count_vectorizer = CountVectorizer()

bow_matrix = count_vectorizer.fit_transform(documents)

print("\nBag of Words Vocabulary:")
print(count_vectorizer.get_feature_names_out())

print("\nBag of Words Matrix:")
print(bow_matrix.toarray())

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

print("\nTF-IDF Vocabulary:")
print(tfidf_vectorizer.get_feature_names_out())

print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())

print("\nTF-IDF Feature Names:")
print(tfidf_vectorizer.get_feature_names_out())

print("\nTF-IDF Scores for First Document:")

first_document_scores = tfidf_matrix[0].toarray()[0]

for word, score in zip(
    tfidf_vectorizer.get_feature_names_out(),
    first_document_scores
):
    if score > 0:
        print(f"{word}: {score:.4f}")