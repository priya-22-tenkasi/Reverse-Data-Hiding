import numpy as np

class RhombusPredictor:

    @staticmethod
    def predict(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.int32)
        h, w = image.shape
        predicted = np.zeros((h, w), dtype=np.int32)
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                top = image[i - 1, j]
                left = image[i, j - 1]
                right = image[i, j + 1]
                bottom = image[i + 1, j]
                predicted[i, j] = (top + left + right + bottom) // 4
        return predicted

    @staticmethod
    def prediction_error(image: np.ndarray):
        predicted = RhombusPredictor.predict(image)
        error = image.astype(np.int32) - predicted
        return predicted, error