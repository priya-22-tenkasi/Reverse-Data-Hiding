import numpy as np
import matplotlib.pyplot as plt
class ErrorHistogram:

    @staticmethod
    def calculate(error, mask=None):
        if mask is not None:
            values = error[mask]
        else:
            values = error.flatten()
        values = values.astype(np.int32)
        unique_values, counts = np.unique(
            values,
            return_counts=True
        )
        histogram = dict(
            zip(unique_values, counts)
        )
        return histogram

    @staticmethod
    def plot(error, mask=None, save_path=None):
        if mask is not None:
            values = error[mask]
        else:
            values = error.flatten()
        values = values.astype(np.int32)
        plt.figure(figsize=(10, 5))
        plt.hist(
            values,
            bins=np.arange(
                values.min() - 0.5,
                values.max() + 1.5,
                1
            )
        )
        plt.xlabel("Prediction Error")
        plt.ylabel("Frequency")
        plt.title("Prediction Error Histogram")
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()