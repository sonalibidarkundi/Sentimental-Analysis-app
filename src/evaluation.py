from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_true, y_pred, model_name):
    """
    Evaluate the model and print metrics.
    """
    print(f"Evaluation for {model_name}:")
    print(classification_report(y_true, y_pred))
    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"Weighted F1-Score: {f1}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    
    return f1
