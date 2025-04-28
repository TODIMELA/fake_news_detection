import pytest
import os
import shutil
from unittest.mock import patch, MagicMock
from src.models import train, evaluate
from src.models.text_model import BertClassifier, RobertaClassifier, DistilBertClassifier
from src.models.image_model import ResNetClassifier, VGGClassifier
import numpy as np

# Mock model classes for testing
class MockBertClassifier(BertClassifier):
    def __init__(self):
        super().__init__()
        self.fit = MagicMock()
        self.predict = MagicMock(return_value=np.random.randint(0, 2, size=50))
        self.predict_proba = MagicMock(return_value=np.random.rand(50, 2))

class MockRobertaClassifier(RobertaClassifier):
    def __init__(self):
        super().__init__()
        self.fit = MagicMock()
        self.predict = MagicMock(return_value=np.random.randint(0, 2, size=50))
        self.predict_proba = MagicMock(return_value=np.random.rand(50, 2))

class MockDistilBertClassifier(DistilBertClassifier):
    def __init__(self):
        super().__init__()
        self.fit = MagicMock()
        self.predict = MagicMock(return_value=np.random.randint(0, 2, size=50))
        self.predict_proba = MagicMock(return_value=np.random.rand(50, 2))

class MockResNetClassifier(ResNetClassifier):
    def __init__(self):
        super().__init__()
        self.fit = MagicMock()
        self.predict = MagicMock(return_value=np.random.randint(0, 2, size=50))
        self.predict_proba = MagicMock(return_value=np.random.rand(50, 2))

class MockVGGClassifier(VGGClassifier):
    def __init__(self):
        super().__init__()
        self.fit = MagicMock()
        self.predict = MagicMock(return_value=np.random.randint(0, 2, size=50))
        self.predict_proba = MagicMock(return_value=np.random.rand(50, 2))

@pytest.fixture(scope="function")
def setup_teardown():    
    # Create temporary directory for testing
    test_dir = 'test_data'
    if not os.path.exists(test_dir):
      os.makedirs(test_dir)
    
    yield test_dir

    # Remove the test directory and its contents
    shutil.rmtree(test_dir, ignore_errors=True)
    
@pytest.mark.parametrize("model_type, mock_model_class", [
    ("bert", MockBertClassifier),
    ("roberta", MockRobertaClassifier),
    ("distilbert", MockDistilBertClassifier),
    ("resnet", MockResNetClassifier),
    ("vgg", MockVGGClassifier)
])
@patch("src.models.train.evaluate_model")
def test_train_model(mock_evaluate_model, model_type, mock_model_class, setup_teardown):
    """Tests the text and image model training process."""
    test_dir = setup_teardown
    
    X_train = np.random.rand(100, 10)
    y_train = np.random.randint(0, 2, size=100)
    X_val = np.random.rand(50, 10)
    y_val = np.random.randint(0, 2, size=50)

    if model_type in ["bert", "roberta", "distilbert"]:
      with patch("src.models.train." + model_type.capitalize() + "Classifier", new=mock_model_class):
        model = train.train_text_model(X_train, y_train, model_type=model_type, tuning_method="none", n_trials=1)
    elif model_type in ["resnet", "vgg"]:
      with patch("src.models.train." + model_type.capitalize() + "Classifier", new=mock_model_class):
        model = train.train_image_model(X_train, y_train, model_type=model_type, tuning_method="none", n_trials=1)

    # Check if train_text_model or train_image_model were called
    assert hasattr(model, "fit")
    assert model.fit.called

    #Check if evaluate_model was called
    assert mock_evaluate_model.called


@patch("src.models.evaluate.accuracy_score", return_value=0.9)
@patch("src.models.evaluate.precision_score", return_value=0.8)
@patch("src.models.evaluate.recall_score", return_value=0.7)
@patch("src.models.evaluate.f1_score", return_value=0.85)
@patch("src.models.evaluate.roc_auc_score", return_value=0.95)
def test_evaluate_model(mock_roc_auc_score, mock_f1_score, mock_recall_score, mock_precision_score, mock_accuracy_score, setup_teardown):
    """Tests the model evaluation process."""
    test_dir = setup_teardown
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, size=100)
    model = MagicMock()
    metrics = evaluate.evaluate_model(model, X, y)
    
    # Check if the functions were called
    assert mock_accuracy_score.called
    assert mock_precision_score.called
    assert mock_recall_score.called
    assert mock_f1_score.called
    assert mock_roc_auc_score.called