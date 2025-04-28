import torch
import torch.nn as nn
from transformers import BertModel, RobertaModel, DistilBertModel
# Define a classifier for BERT models
class BertClassifier(nn.Module):
    def __init__(self, num_labels=2, dropout_rate=0.3, use_attention=True):
        """
        Initializes the BERT classifier model.
        Args:
            num_labels (int): Number of output labels for classification.
            dropout_rate (float): Dropout probability.
            use_attention (bool): Whether to use attention mechanism.
        """
        super(BertClassifier, self).__init__()
        # Load the pretrained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)
        # Classification layer
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        # Flag to determine if attention mechanism should be used
        self.use_attention = use_attention
        # If using attention, define the attention mechanism layers
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
                nn.Tanh(),
                nn.Linear(self.bert.config.hidden_size, 1)
            )
    # Forward pass of the model
    def forward(self, input_ids, attention_mask):
        """
        Defines the forward pass for the BERT classifier.
        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
        Returns:
            torch.Tensor: Output logits.
            torch.Tensor or None: Attention weights (if available).
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if self.use_attention:
            hidden_states = outputs.last_hidden_state # Get the hidden states
            attention_scores = self.attention(hidden_states).squeeze(-1) # Compute attention scores
            attention_weights = torch.softmax(attention_scores, dim=1) # Apply softmax to get attention weights
            context_vector = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1) # Compute context vector
            pooled_output = context_vector # Use context vector as the pooled output
        else:
            pooled_output = outputs.pooler_output # Use the pooler output from BERT
        pooled_output = self.dropout(pooled_output) # Apply dropout for regularization
        logits = self.classifier(pooled_output) # Pass through the classification layer
        return logits, outputs.attentions if hasattr(outputs, 'attentions') else None


class RobertaClassifier(nn.Module):
    def __init__(self, num_labels=2, dropout_rate=0.3, use_attention=True):
        super(RobertaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)
        # Classification layer
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
        # Flag to determine if attention mechanism should be used
        self.use_attention = use_attention
        # If using attention, define the attention mechanism layers
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(self.roberta.config.hidden_size, self.roberta.config.hidden_size),
                nn.Tanh(),
                nn.Linear(self.roberta.config.hidden_size, 1)
            )
    # Forward pass of the model
    def forward(self, input_ids, attention_mask):
        """
        Defines the forward pass for the RoBERTa classifier.
        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
        Returns:
            torch.Tensor: Output logits.
            torch.Tensor or None: Attention weights (if available).
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        if self.use_attention:
            hidden_states = outputs.last_hidden_state # Get the hidden states
            attention_scores = self.attention(hidden_states).squeeze(-1) # Compute attention scores
            attention_weights = torch.softmax(attention_scores, dim=1) # Apply softmax to get attention weights
            context_vector = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1) # Compute context vector
            pooled_output = context_vector # Use context vector as the pooled output
        else:
            pooled_output = outputs.pooler_output # Use the pooler output from RoBERTa
        pooled_output = self.dropout(pooled_output) # Apply dropout for regularization
        logits = self.classifier(pooled_output) # Pass through the classification layer
        return logits, outputs.attentions if hasattr(outputs, 'attentions') else None


class DistilBertClassifier(nn.Module):
    def __init__(self, num_labels=2, dropout_rate=0.3, use_attention=True):
        super(DistilBertClassifier, self).__init__()
        # Load the pretrained DistilBERT model
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        # Dropout layer for regularization
        self.dropout = nn.Dropout(dropout_rate)
        # Classification layer
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)
        # Flag to determine if attention mechanism should be used
        self.use_attention = use_attention
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(self.distilbert.config.hidden_size, self.distilbert.config.hidden_size),
                nn.Tanh(),
                nn.Linear(self.distilbert.config.hidden_size, 1)
            )
    # Forward pass of the model
    def forward(self, input_ids, attention_mask):
        """
        Defines the forward pass for the DistilBERT classifier.
        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for the input.
        Returns:
            torch.Tensor: Output logits.
            torch.Tensor or None: Attention weights (if available).
        """
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        if self.use_attention:
            hidden_states = outputs.last_hidden_state # Get the hidden states
            attention_scores = self.attention(hidden_states).squeeze(-1) # Compute attention scores
            attention_weights = torch.softmax(attention_scores, dim=1) # Apply softmax to get attention weights
            context_vector = torch.bmm(attention_weights.unsqueeze(1), hidden_states).squeeze(1) # Compute context vector
            pooled_output = context_vector # Use context vector as the pooled output
        else:
            pooled_output = outputs.last_hidden_state[:, 0, :] # Use the CLS token representation
        pooled_output = self.dropout(pooled_output) # Apply dropout for regularization
        logits = self.classifier(pooled_output) # Pass through the classification layer
        return logits, outputs.attentions if hasattr(outputs, 'attentions') else None