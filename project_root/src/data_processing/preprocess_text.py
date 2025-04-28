import nltk
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import re
import random

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class TextPreprocessor:
    """
    A class for preprocessing text data, including cleaning, tokenization,
    stop-word removal, lemmatization, and data augmentation.
    """
    def __init__(self, language='en'):
        """
        Initializes the TextPreprocessor with specified language settings.
        Args:
            language (str): The language of the text data (default: 'en').
        """
        self.nlp = spacy.load(language, disable=['parser', 'ner'])
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()

    def tokenize(self, text):
        """
        Tokenizes the input text into individual words.
        Args:
            text (str): The input text.
        Returns:
            list: A list of tokens (words).
        """
        if not isinstance(text, str):
            return []
        return word_tokenize(text)

    def remove_stopwords(self, tokens):
        """
        Removes stop words from a list of tokens.
        Args:
            tokens (list): A list of tokens.
        Returns:
            list: A list of tokens without stop words.
        """
        return [token for token in tokens if token.lower() not in self.stop_words]

    def lemmatize(self, tokens):
        """
        Lemmatizes a list of tokens using WordNetLemmatizer.
        Args:
            tokens (list): A list of tokens.
        Returns:
            list: A list of lemmatized tokens.
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]

    def clean_text(self, text):
        """
        Cleans the input text by removing URLs, mentions, hashtags, and special characters,
        and converting it to lowercase.
        Args:
            text (str): The input text.
        Returns:
            str: The cleaned text.
        """
        if not isinstance(text, str) or not text:
            return ""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Remove URLs
        text = re.sub(r'\@\w+|\#','', text) # Remove mentions and hashtags
        text = re.sub(r'[^\w\s]', '', text) # Remove special characters
        text = text.lower() # Convert to lowercase
        return text

    def preprocess(self, text):
        """
        Preprocesses the input text by cleaning, tokenizing, removing stop words,
        and lemmatizing.
        Args:
            text (str): The input text.
        Returns:
            str: The preprocessed text.
        """
        if not isinstance(text, str) or not text:
            return ""
        cleaned_text= self.clean_text(text) # Clean the text
        tokens = self.tokenize(cleaned_text)
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return " ".join(tokens)

    def get_synonyms(self, word):
        """Gets synonyms for a given word using WordNet."""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ").lower()
                if synonym != word.lower():
                    synonyms.add(synonym)
    
    def synonym_replacement(self, tokens, n=1):  
        """Replaces n words in a list of tokens with their synonyms."""
        new_tokens = tokens[:] # Create a copy to avoid modifying the original list
        random_token_list = list(set([token for token in tokens if token not in self.stop_words])) # Get a list of unique words (excluding stop words)
        random.shuffle(random_token_list) # Shuffle the list to randomize the replacement order
        num_replaced = 0 # Counter to track how many words have been replaced
        for random_token in random_token_list:
            synonyms = self.get_synonyms(random_token) # Get synonyms for the word
            if len(synonyms) >= 1:
                synonym = random.choice(list(synonyms)) # Choose a random synonym
                new_tokens = [synonym if token == random_token else token for token in new_tokens] # Replace the word with the synonym
                num_replaced += 1
            if num_replaced >= n: # Stop when n words have been replaced
                break
        return new_tokens

    def random_insertion(self, tokens, n=1):
        """Inserts n random synonyms into the list of tokens."""
        new_tokens = tokens[:] # Create a copy of the list
        for _ in range(n):
            random_synonym = random.choice(list(self.get_synonyms(random.choice(tokens)))) # Choose a random synonym
            new_tokens.insert(random.randint(0, len(new_tokens)), random_synonym) # Insert the synonym at a random position
        return new_tokens
    def handle_missing_data(self, text_list, strategy='fill_empty'):
        """
        Handles missing data in a list of text by either filling empty strings or excluding them.
        Args:
            text_list (list): A list of text strings.
            strategy (str): The strategy to handle missing data ('fill_empty' or 'exclude').
        Returns:
            list: A list of processed text strings.
        """
        processed_text = [] # Initialize an empty list to store processed text
        for text in text_list:
            if text is None or text.strip() == '': # Check if the text is missing
                if strategy == 'fill_empty': # If strategy is to fill empty strings
                    processed_text.append('') # Append an empty string
                elif strategy == 'exclude': # If strategy is to exclude missing data
                  continue # Skip this iteration
            else:
                processed_text.append(text) # Append the non-missing text

        return processed_text