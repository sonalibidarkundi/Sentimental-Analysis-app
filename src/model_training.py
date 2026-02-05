from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import joblib
import os

def train_logistic_regression(X_train, y_train):
    """
    Train a Logistic Regression model.
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    """
    Train a Support Vector Machine model.
    """
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model

def save_model(model, filename):
    """
    Save the trained model to disk.
    """
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, f'models/{filename}', protocol=4)

def load_model(filename):
    """
    Load a trained model from disk.
    """
    if filename.startswith('models/'):
        return joblib.load(filename)
    else:
        return joblib.load(f'models/{filename}')

def train_and_evaluate_models(X, y):
    """
    Train and evaluate Logistic Regression and SVM models.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_f1 = f1_score(y_test, lr_pred, average='weighted')
    print(f"Logistic Regression F1-Score: {lr_f1}")
    save_model(lr_model, 'logistic_regression.pkl')

    # Train SVM
    svm_model = train_svm(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    svm_f1 = f1_score(y_test, svm_pred, average='weighted')
    print(f"SVM F1-Score: {svm_f1}")
    save_model(svm_model, 'svm.pkl')

    return lr_model, svm_model, lr_f1, svm_f1

def save_vectorizer(vectorizer, filename):
    """
    Save the trained vectorizer to disk.
    """
    os.makedirs('models', exist_ok=True)
    joblib.dump(vectorizer, f'models/{filename}', protocol=4)

def load_vectorizer(filename):
    """
    Load a trained vectorizer from disk.
    """
    if filename.startswith('models/'):
        return joblib.load(filename)
    else:
        return joblib.load(f'models/{filename}')
