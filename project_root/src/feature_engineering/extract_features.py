import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.data_processing.preprocess_text import TextPreprocessor
from src.data_processing.preprocess_image import ImageProcessor
from gensim.models import Word2Vec, KeyedVectors
import spacy


class FeatureExtractor:
    """
    A class for extracting various text features from a corpus of documents.
    """

    def __init__(self):
        """
        Initializes the FeatureExtractor, downloads necessary NLTK data,
        and loads the spaCy language model.
        """
        nltk.download('punkt', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        # Load the spaCy English language model, disabling unnecessary components for efficiency.
        self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    def tfidf(self, corpus):
        """
        Computes the TF-IDF (Term Frequency-Inverse Document Frequency) matrix for the corpus.

        Args:
            corpus (list): A list of text documents.

        Returns:
            tuple: A tuple containing the TF-IDF matrix and the vectorizer.
        """
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(corpus)
        return tfidf_matrix, vectorizer

    def word2vec(self, corpus, vector_size=100, window=5, min_count=1, workers=4):
        """
        Trains a Word2Vec model on the corpus and computes document vectors.

        Args:
            corpus (list): A list of text documents.
            vector_size (int): Dimensionality of the word vectors.
            window (int): Maximum distance between the current and predicted word within a sentence.
            min_count (int): Ignores all words with total frequency lower than this.
            workers (int): Number of worker threads to train the model.

        Returns:
            tuple: A tuple containing the document vectors and the trained Word2Vec model.
        """
        tokenized_corpus = [nltk.word_tokenize(doc.lower()) for doc in corpus]
        model = Word2Vec(sentences=tokenized_corpus, vector_size=vector_size,
                         window=window, min_count=min_count, workers=workers)

        vectors = []
        for tokens in tokenized_corpus:
            doc_vector = np.zeros(vector_size)
            valid_word_count = 0
            for word in tokens:
                if word in model.wv:
                    doc_vector += model.wv[word]  # Sum the word vectors
                    valid_word_count += 1
            if valid_word_count > 0:
                doc_vector /= valid_word_count  # Average the word vectors
            vectors.append(doc_vector)
        return np.array(vectors), model

    def glove(self, corpus, vector_size=100):
        """
        Indicates that GloVe is not implemented and suggests using Word2Vec.

        Args:
            corpus (list): A list of text documents.
            vector_size (int): The size of the vectors (not used).

        Returns:
            tuple: An empty list and None, indicating GloVe is not implemented.
        """
        print('GloVe not implemented, please use Word2Vec.')  # Inform user about non-implementation
        return [], None

    def pos_tagging(self, corpus):
        """
        Performs Part-of-Speech (POS) tagging on the corpus.
        """
        pos_tags = []
        for doc in corpus:
            doc_spacy = self.nlp(doc)
            tags = [token.pos_ for token in doc_spacy]  # Extract POS tags for each token
            pos_tags.append(tags)  # Append the tags for the current document
        return pos_tags
    
def extract_text_features(text_list):
    """
    Extracts features from a list of text using TF-IDF, Word2Vec, and POS tagging.

    Args:
        text_list (list): A list of text documents.

    Returns:
        tuple: A tuple containing the TF-IDF matrix, Word2Vec vectors, and POS tags.
    """
    # Initialize the FeatureExtractor
    feature_extractor = FeatureExtractor()
    # Preprocess the text
    preprocessor = TextPreprocessor()
    processed_text_list = [preprocessor.preprocess(text) for text in text_list]
    # Extract features
    tfidf_matrix, word2vec_vectors, pos_tags = feature_extractor.extract_features(processed_text_list)
    return tfidf_matrix, word2vec_vectors, pos_tags

def extract_image_features(image_paths):
    """
    Extracts image features from a list of image paths.

    Args:
        image_paths (list): A list of paths to image files.

    Returns:
        list: A list of preprocessed and augmented images.
    """
    # Initialize the ImageProcessor
    image_processor = ImageProcessor()
    # Process and augment the images
    processed_images = [image_processor.preprocess_and_augment(image_path) for image_path in image_paths]
    return processed_images

    def extract_features(self, corpus):
        tfidf_matrix, word2vec_vectors, pos_tags = self.tfidf(corpus)
        return tfidf_matrix, word2vec_vectors, pos_tags

    def extract_features(self, corpus):
        tfidf_matrix, tfidf_vectorizer = self.tfidf(corpus)
        word2vec_vectors, word2vec_model = self.word2vec(corpus)
        pos_tags = self.pos_tagging(corpus)
        return tfidf_matrix, word2vec_vectors, pos_tags