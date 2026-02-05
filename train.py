import numpy as np
from src.preprocessing import preprocess_text
from src.feature_extraction import extract_tfidf_features
from src.model_training import train_logistic_regression, save_model, save_vectorizer

# Generate dummy data
reviews = [
    "This product is amazing, I love it!",
    "Great quality and fast delivery.",
    "Excellent service, highly recommend.",
    "Best purchase I've ever made.",
    "Wonderful experience, will buy again.",
    "This is terrible, waste of money.",
    "Poor quality, do not buy.",
    "Awful customer service.",
    "Disappointed with the product.",
    "Never buying from here again.",
    "I hate this item, it's horrible.",
    "Worst thing I ever bought.",
    "Absolutely fantastic, five stars.",
    "Love the design and functionality.",
    "Bad experience, not satisfied.",
    "Cheap material, breaks easily.",
    "Superb value for money.",
    "Highly disappointed, avoid this.",
    "Outstanding performance.",
    "Regret buying this junk."
]

labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0]  # 1 for positive, 0 for negative

# Preprocess and extract features
processed_reviews = [preprocess_text(review) for review in reviews]
features, vectorizer = extract_tfidf_features(processed_reviews)

# Train the model
model = train_logistic_regression(features, labels)

# Save the model as model.pkl
save_model(model, 'model.pkl')
save_vectorizer(vectorizer, 'vectorizer.pkl')

print("Model trained and saved as models/model.pkl")
print("Vectorizer saved as models/vectorizer.pkl")
