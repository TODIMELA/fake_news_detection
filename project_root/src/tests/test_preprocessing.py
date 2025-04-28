import pytest
from src.data_processing.preprocess_text import TextPreprocessor
from src.data_processing.preprocess_image import ImageProcessor
from src.data_processing.preprocess_video import extract_frames
import os
import cv2
import numpy as np
import importlib

def test_preprocess_text():
    """Test the TextPreprocessor class."""
    preprocessor = TextPreprocessor()
    text = (
        "This is a sample text with some stopwords like the, a, and. "
        "It has also some capitalized words Like This. "
        "Testing punctuation,!"
    )
    processed_text = preprocessor.preprocess(text)

    # Basic type check
    assert isinstance(processed_text, str)

    # Check if stopwords are removed
    assert "the" not in processed_text.lower()
    assert "a" not in processed_text.lower()
    assert "and" not in processed_text.lower()
    
    #check if it keeps important words.
    assert "like" in processed_text.lower() 
    assert "testing" in processed_text.lower() 

    #Check if it removes punctuation.
    assert "," not in processed_text
    assert "!" not in processed_text

def test_preprocess_text_empty():
    """Test TextPreprocessor with an empty string."""
    preprocessor = TextPreprocessor()
    processed_text = preprocessor.preprocess("")
    assert processed_text == ""  # Expected an empty string
    
def test_preprocess_text_none():
    """Test TextPreprocessor with None."""
    preprocessor = TextPreprocessor()
    processed_text = preprocessor.preprocess(None)
    assert processed_text == ""  # Expected an empty string

def create_dummy_image(filepath, size=(100, 100)):
    """Create a dummy image for testing."""
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    cv2.imwrite(filepath, img)

def test_preprocess_image():
    """Test the ImageProcessor class."""
    dummy_image_path = "test_image.jpg"
    create_dummy_image(dummy_image_path)
    
    processor = ImageProcessor(target_size=(64, 64))
    processed_image = processor.preprocess_and_augment(dummy_image_path, augment=False)
    
    #Basic shape check.
    assert processed_image.shape == (64, 64, 3) 
    # Check if values are between 0 and 1 after normalization
    assert np.all(processed_image >= 0) and np.all(processed_image <= 1)

    os.remove(dummy_image_path)

def test_preprocess_image_nonexistent():
    """Test ImageProcessor with a nonexistent image."""
    processor = ImageProcessor()
    with pytest.raises(FileNotFoundError):
        processor.preprocess_and_augment("nonexistent.jpg")

def create_dummy_video(filepath, duration=1):
    """Create a dummy video for testing."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filepath, fourcc, 20.0, (100, 100))
    for _ in range(20 * duration):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out.write(frame)
    out.release()
    
def test_preprocess_video_moviepy_not_installed():
    """Test if the correct error is thrown if moviepy is not installed."""
    with pytest.raises(ImportError, match="moviepy is not installed"):
        # Remove moviepy from the modules.
        import sys
        sys.modules["moviepy"] = None
        importlib.reload(sys.modules["src.data_processing.preprocess_video"])

def test_preprocess_video():
    """Test the extract_frames function."""
    dummy_video_path = "test_video.mp4"
    create_dummy_video(dummy_video_path)

    frames = preprocess_video.preprocess(dummy_video_path)
    assert isinstance(frames, list)
    assert len(frames) > 0
    
    os.remove(dummy_video_path)

def test_preprocess_video_nonexistent():
    """Test extract_frames with a nonexistent video."""
    with pytest.raises(FileNotFoundError):
        extract_frames("nonexistent.mp4", "test_output")