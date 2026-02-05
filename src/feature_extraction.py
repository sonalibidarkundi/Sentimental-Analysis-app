from sklearn.feature_extraction.text import TfidfVectorizer

def extract_tfidf_features(texts):
    """
    Extract TF-IDF features from the text data.
    """
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform(texts)
    return X, vectorizer
