import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


def save_image(image, filename):
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(filename, image)

def normalize_error(error):
    error_display = cv2.normalize(
        error.astype(np.float32),
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )
    return error_display.astype(np.uint8)

def save_prediction_error_image(error):
    error_display = normalize_error(error)
    save_image(
        error_display,
        "outputs/prediction_error.png"
    )

def save_histogram(error):
    os.makedirs("outputs", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(
        error.flatten(),
        bins=100
    )
    plt.title("Prediction Error Histogram")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        "outputs/error_histogram.png",
        dpi=300
    )
    plt.close()

def save_all_images(
        original,
        predicted,
        error,
        checkerboard
):
    os.makedirs("outputs", exist_ok=True)
    cv2.imwrite(
        "outputs/original.png",
        original.astype(np.uint8)
    )
    cv2.imwrite(
        "outputs/predicted.png",
        np.clip(predicted, 0, 255).astype(np.uint8)
    )
    save_prediction_error_image(error)
    cv2.imwrite(
        "outputs/checkerboard.png",
        checkerboard.astype(np.uint8)
    )
    save_histogram(error)