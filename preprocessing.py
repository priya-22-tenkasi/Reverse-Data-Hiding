import cv2
import numpy as np
def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )
    return img

def preprocess_image(img):
    if img.ndim != 2:
        raise ValueError("Input image must be grayscale.")
    img = img.astype(np.int32)
    return img