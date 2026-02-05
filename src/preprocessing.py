import re
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Ensure wordnet is loaded
try:
    wordnet.ensure_loaded()
except AttributeError:
    pass  # In case ensure_loaded doesn't exist

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    """
    Clean the text by removing special characters, punctuation, and converting to lowercase.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

def remove_stopwords(text):
    """
    Remove stopwords from the text.
    """
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def lemmatize_text(text):
    """
    Perform lemmatization on the text.
    """
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def preprocess_text(text):
    """
    Full preprocessing pipeline: clean, remove stopwords, lemmatize.
    """
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text
