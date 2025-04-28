import os
import joblib
import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.models.text_model import BertClassifier, RobertaClassifier, DistilBertClassifier
from src.models.image_model import ResNetClassifier, VGGClassifier
from src.feature_engineering.extract_features import FeatureExtractor
from transformers import BertTokenizer, RobertaTokenizer, DistilBertTokenizer
import numpy as np

def train_text_model(X, y, model_type='bert', tuning_method='optuna', n_trials=100, max_length=512):
    """Trains a text model with optional hyperparameter tuning."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select tokenizer based on model type
    if model_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model_class = BertClassifier
    elif model_type == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model_class = RobertaClassifier
    elif model_type == 'distilbert':
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        model_class = DistilBertClassifier
    else:
        raise ValueError(f"Invalid model_type: {model_type}")

    # Tokenize the input data
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    # Create TensorDatasets and DataLoaders
    train_dataset = TensorDataset(X_train_tokenized['input_ids'], X_train_tokenized['attention_mask'], torch.tensor(y_train))
    test_dataset = TensorDataset(X_test_tokenized['input_ids'], X_test_tokenized['attention_mask'], torch.tensor(y_test))

    def objective(trial):
        """Objective function for hyperparameter tuning."""
        # Hyperparameters to tune
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5)
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        use_attention = trial.suggest_categorical('use_attention', [True, False])

        # Create and train the model
        model = model_class(num_labels=2, dropout_rate=dropout_rate, use_attention=use_attention)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        loss_function = torch.nn.CrossEntropyLoss()

        model.train()
        for input_ids, attention_mask, labels in train_loader:
            optimizer.zero_grad()
            logits, _ = model(input_ids, attention_mask)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()

        # Evaluate the model
        model.eval()
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for input_ids, attention_mask, labels in test_loader:
                logits, _ = model(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        accuracy = total_correct / total_samples
        return accuracy

    # Hyperparameter tuning
    if tuning_method == 'optuna':
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        print(f"Best hyperparameters: {best_params}")
        best_model = model_class(num_labels=2, dropout_rate=best_params['dropout_rate'], use_attention=best_params['use_attention'])
        optimizer = torch.optim.AdamW(best_model.parameters(), lr=best_params['learning_rate'])
        loss_function = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid tuning_method specified.")

    # Train with best hyperparameters
    best_model.train()
    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    for input_ids, attention_mask, labels in train_loader:
        optimizer.zero_grad()
        logits, _ = best_model(input_ids, attention_mask)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

    return best_model, X_test_tokenized, y_test


def train_image_model(X, y, model_type = 'resnet', tuning_method='optuna', n_trials=100):
    """Trains an image model with optional hyperparameter tuning."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = torch.tensor(np.array(X_train)).permute(0, 3, 1, 2)
    X_test = torch.tensor(np.array(X_test)).permute(0, 3, 1, 2)
    y_train = torch.tensor(y_train)
    y_test = torch.tensor(y_test)
    train_dataset = TensorDataset(X_train,y_train)
    test_dataset = TensorDataset(X_test, y_test)
    if model_type == 'resnet':
        model_class = ResNetClassifier
    elif model_type == 'vgg':
        model_class = VGGClassifier
    else:
        raise ValueError(f"Invalid model_type: {model_type}")
    
    def objective(trial):
        """Objective function for hyperparameter tuning."""
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)
        model = model_class(num_classes=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = torch.nn.CrossEntropyLoss()
        
        model.train()
        train_loader = DataLoader(train_dataset, batch_size=32)
        for X_train_batch, y_train_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_train_batch)
            loss = loss_function(logits, y_train_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        test_loader = DataLoader(test_dataset, batch_size=32)
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for X_test_batch, y_test_batch in test_loader:
                logits = model(X_test_batch)
                predictions = torch.argmax(logits, dim=1)
                total_correct += (predictions == y_test_batch).sum().item()
                total_samples += y_test_batch.size(0)
        accuracy = total_correct / total_samples
        return accuracy

    # Hyperparameter tuning
    if tuning_method == 'optuna':
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best_params = study.best_params
        print(f"Best hyperparameters: {best_params}")
        best_model = model_class(num_classes=2)
        optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params["learning_rate"])

    else:
        raise ValueError("Invalid tuning_method specified for image model.")
    
    best_model.train()
    train_loader = DataLoader(train_dataset, batch_size=32)
    loss_function = torch.nn.CrossEntropyLoss()
    for X_train_batch, y_train_batch in train_loader:
            optimizer.zero_grad()
            logits = best_model(X_train_batch)
            loss = loss_function(logits, y_train_batch)
            loss.backward()
            optimizer.step()
    return best_model,X_test, y_test

def evaluate_model(model, X_test, y_test, model_type = 'text'):
    """Evaluates a trained model and prints the evaluation metrics."""
    if model_type == 'text':
      model.eval()
      with torch.no_grad():
          y_pred = []
          for input_ids, attention_mask in zip(X_test['input_ids'], X_test['attention_mask']):
            logits, _ = model(input_ids.unsqueeze(0), attention_mask.unsqueeze(0))
            y_pred.append(torch.argmax(logits, dim=1).item())

    else:
      model.eval()
      with torch.no_grad():
        y_pred = [torch.argmax(model(x), dim=0).item() for x in X_test]

    y_pred = np.array(y_pred)
    y_test = np.array(y_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test, y_pred, average='weighted')
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }
def load_data(data_dir):
    """Loads text and image data from the specified directory."""
    text_data_path = os.path.join(data_dir, 'processed_text.txt')
    image_data_path = os.path.join(data_dir, 'processed_images.npy')

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Processed data directory '{data_dir}' not found.")
    if not os.path.exists(text_data_path):
        raise FileNotFoundError(f"Text data file '{text_data_path}' not found.")
    if not os.path.exists(image_data_path):
        raise FileNotFoundError(f"Image data file '{image_data_path}' not found.")
    
    with open(text_data_path, 'r', encoding='utf-8') as f:
        text_data = [line.strip() for line in f]
    image_data = np.load(image_data_path)

    return text_data, image_data
def main():
    """Main function to train and evaluate the models."""
    data_dir = 'data/processed'  # Define the data directory
    models_dir = 'models' # define model directory
    os.makedirs(models_dir, exist_ok=True) #create the directory if it does not exists.

    # Load data
    text_data, image_data = load_data(data_dir)
    num_labels = len(set([0, 1])) #dynamic labels
    labels = [0, 1] * (len(text_data) // num_labels)
    labels_image = [0, 1] * (len(image_data) // num_labels)
    # Feature Extraction
    feature_extractor = FeatureExtractor()
    X_text,_,_ = feature_extractor.extract_features(text_data)
    X_image = image_data

    # Train models
    text_model,X_text_test,y_text_test = train_text_model(X_text, labels, model_type='bert', tuning_method='optuna', n_trials=10)
    image_model,X_image_test,y_image_test = train_image_model(X_image, labels_image,model_type = 'resnet', tuning_method='optuna', n_trials=5)

    # Save models
    torch.save(text_model.state_dict(), os.path.join(models_dir, 'text_model.pth'))
    torch.save(image_model.state_dict(), os.path.join(models_dir, 'image_model.pth'))
    # Evaluate models
    print("\nText Model Evaluation:")
    evaluate_model(text_model, X_text_test, y_text_test,model_type = 'text')
    print("\nImage Model Evaluation:")
    evaluate_model(image_model, X_image_test, y_image_test, model_type = 'image')

if __name__ == '__main__':
    main()