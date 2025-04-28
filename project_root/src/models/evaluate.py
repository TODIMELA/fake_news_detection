import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import logging

logging.basicConfig(level=logging.INFO)

def evaluate_model(model, X, y, n_splits=5):
    """
    Evaluates a machine learning model using k-fold cross-validation.

    Args:
        model: The machine learning model to evaluate.
        X: The feature data.
        y: The target data.
        n_splits: The number of folds for cross-validation (default: 5).

    Returns:
        A dictionary containing the average evaluation metrics and confusion matrix.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_auc_scores = []
    confusion_matrices = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        try:
          y_prob = model.predict_proba(X_val)[:, 1]
          roc_auc = roc_auc_score(y_val, y_prob)
          roc_auc_scores.append(roc_auc)
        except AttributeError:
          logging.warning('Model does not support predict_proba. ROC-AUC will be skipped.')
          
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=1)
        recall = recall_score(y_val, y_pred, zero_division=1)
        f1 = f1_score(y_val, y_pred, zero_division=1)
        cm = confusion_matrix(y_val, y_pred)
        

        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
        confusion_matrices.append(cm)

    avg_accuracy = np.mean(accuracy_scores)
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    avg_f1 = np.mean(f1_scores)
    if len(roc_auc_scores) >0:
      avg_roc_auc = np.mean(roc_auc_scores)
    else:
      avg_roc_auc = None
    
    avg_confusion_matrix = np.mean(confusion_matrices, axis=0).tolist()


    logging.info(f"Average Accuracy: {avg_accuracy:.4f}")
    logging.info(f"Average Precision: {avg_precision:.4f}")
    logging.info(f"Average Recall: {avg_recall:.4f}")
    logging.info(f"Average F1 Score: {avg_f1:.4f}")
    if avg_roc_auc is not None:
      logging.info(f"Average ROC-AUC Score: {avg_roc_auc:.4f}")
    logging.info(f"Average Confusion Matrix: {avg_confusion_matrix}")


    return {
        "accuracy": avg_accuracy,
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "roc_auc": avg_roc_auc,
        "confusion_matrix": avg_confusion_matrix
    }