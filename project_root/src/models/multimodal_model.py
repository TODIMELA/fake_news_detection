import torch
import torch.nn as nn
from src.models.text_model import BertClassifier, RobertaClassifier, DistilBertClassifier
from src.models.image_model import ResNetClassifier, VGGClassifier

class MultimodalClassifier(nn.Module):
        """
        Initializes the MultimodalClassifier.

        Args:
            text_model_type (str): Type of the text model ('bert', 'roberta', 'distilbert').
            image_model_type (str): Type of the image model ('resnet', 'vgg').
            num_classes (int): Number of output classes.
        """
        super(MultimodalClassifier, self).__init__()

        # Initialize text model based on text_model_type
        if text_model_type == 'bert':
            self.text_model = BertClassifier()
        elif text_model_type == 'roberta':
            self.text_model = RobertaClassifier()
        elif text_model_type == 'distilbert':
            self.text_model = DistilBertClassifier()
        else:
            raise ValueError(f"Invalid text_model_type: {text_model_type}")

        # Freeze text model weights
        for param in self.text_model.parameters():
            param.requires_grad = False

        # Initialize image model based on image_model_type
        if image_model_type == 'resnet':
            self.image_model = ResNetClassifier()
        elif image_model_type == 'vgg':
            self.image_model = VGGClassifier()
        else:
            raise ValueError(f"Invalid image_model_type: {image_model_type}")

        # Freeze image model weights
        for param in self.image_model.parameters():
            param.requires_grad = False

        # Classifier for combining text and image features
        text_feature_size = self.text_model.classifier.in_features if hasattr(self.text_model, 'classifier') else self.text_model.classifier.out_features
        image_feature_size = self.image_model.resnet.fc.out_features if hasattr(self.image_model, 'resnet') else self.image_model.vgg.classifier[6].out_features
        combined_feature_size = text_feature_size + image_feature_size

        # Define the classifier network
        self.classifier = nn.Sequential(
            nn.Linear(combined_feature_size, combined_feature_size // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(combined_feature_size // 2, num_classes)
        )

    def forward(self, text_input_ids, text_attention_mask, image_input):
        # Get text features
        text_features, _ = self.text_model(text_input_ids, text_attention_mask)

        # Get image features
        image_features = self.image_model(image_input)

        # Concatenate text and image features
        combined_features = torch.cat((text_features, image_features), dim=1)

        # Classify combined features
        output = self.classifier(combined_features)
        return output

        super(MultimodalClassifier, self).__init__()

        # Load pre-trained text model
        self.text_model = TextClassifier()
        self.text_model.load_state_dict(torch.load(text_model_path))
        for param in self.text_model.parameters():
            param.requires_grad = False  # Freeze text model weights

        # Load pre-trained image model
        self.image_model = ImageClassifier()
        self.image_model.load_state_dict(torch.load(image_model_path))
        for param in self.image_model.parameters():
            param.requires_grad = False  # Freeze image model weights

        # Classifier for combining text and image features
        text_feature_size = self.text_model.classifier.in_features
        image_feature_size = self.image_model.fc.in_features
        combined_feature_size = text_feature_size + image_feature_size

        self.classifier = nn.Sequential(
            nn.Linear(combined_feature_size, combined_feature_size // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(combined_feature_size // 2, num_classes)
        )

    def forward(self, text_input, image_input):
        # Get text features
        text_features = self.text_model.bert(text_input)[1]  # Assuming BERT output
        text_features = self.text_model.dropout(text_features)
        text_features = self.text_model.classifier(text_features)

        # Get image features
        image_features = self.image_model(image_input)

        # Concatenate text and image features
        combined_features = torch.cat((text_features, image_features), dim=1)

        # Classify combined features
        output = self.classifier(combined_features)
        return output

# Example usage (assuming you have data and model paths)
# text_model_path = 'models/text_model.pth'
# image_model_path = 'models/image_model.pth'
# multimodal_model = MultimodalClassifier(text_model_path, image_model_path)