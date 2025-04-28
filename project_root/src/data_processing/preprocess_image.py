import cv2
import numpy as np
import random

class ImageProcessor:
    def __init__(self, target_size=(224, 224)):
        """
        Initializes the ImageProcessor with a target size for resizing.

        Args:
            target_size (tuple): The desired size (width, height) for resizing images.
        """
        self.target_size = target_size

    def resize_image(self, image_path):
        """
        Resizes the image to the target size.

        Args:
            image_path (str): The path to the image file.

        Returns:
            numpy.ndarray: The resized image as a NumPy array.

        Raises:
            FileNotFoundError: If the image is not found at the specified path.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        image = cv2.resize(image, self.target_size)
        return image

    def normalize_image(self, image):
        """
        Normalizes the image pixel values to the range [0, 1].

        Args:
            image (numpy.ndarray): The input image as a NumPy array.

        Returns:
            numpy.ndarray: The normalized image.
        """
        image = image.astype(np.float32) / 255.0
        return image

    def random_flip(self, image, flip_prob=0.5):
        """
        Randomly flips the image horizontally with a given probability.

        Args:
            image (numpy.ndarray): The input image.
            flip_prob (float): The probability of flipping the image.

        Returns:
            numpy.ndarray: The potentially flipped image.
        """
        if random.random() < flip_prob:
            image = cv2.flip(image, 1)  # 1 indicates horizontal flip
        return image

    def random_rotation(self, image, max_angle=15):
        """
        Randomly rotates the image within the specified maximum angle.

        Args:
            image (numpy.ndarray): The input image.
            max_angle (int): The maximum angle (in degrees) for rotation.

        Returns:
            numpy.ndarray: The rotated image.
        """
        rows, cols = image.shape[:2]
        angle = random.uniform(-max_angle, max_angle)
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
        return rotated_image

    def random_brightness(self, image, max_delta=30):
        """Randomly adjusts the brightness of the image.""" 
        delta = random.randint(-max_delta, max_delta)
        image = np.clip(image + delta, 0, 255).astype(np.uint8)
        return image

    def random_contrast(self, image, alpha_range=(0.8, 1.2)):
        """Randomly adjusts the contrast of the image.""" 
        alpha = random.uniform(alpha_range[0], alpha_range[1])
        image = np.clip(alpha * image, 0, 255).astype(np.uint8)
        return image

    def augment_image(self, image):
        """Applies a sequence of random augmentations to the image.""" 
        image = self.random_flip(image)
        image = self.random_rotation(image)
        image = self.random_brightness(image)
        image = self.random_contrast(image)
        return image

    def preprocess_and_augment(self, image_path, augment=True):
        """
        Combines resizing, normalization, and optional augmentation.

        Args:
            image_path (str): The path to the image file.
            augment (bool): Whether to apply data augmentation.

        Returns:
            numpy.ndarray: The processed image.
        """
        image = self.resize_image(image_path)
        if augment:
            image = self.augment_image(image)
        image = self.normalize_image(image)
        return image