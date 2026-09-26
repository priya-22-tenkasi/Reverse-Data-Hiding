import numpy as np
def calculate_error_statistics(error):
    minimum = np.min(error)
    maximum = np.max(error)
    mean = np.mean(error)
    std = np.std(error)
    zero_error_pixels = np.sum(error == 0)
    total_pixels = error.size
    zero_error_percentage = (
        zero_error_pixels /
        total_pixels
    ) * 100
    return {
        "Minimum Error": minimum,
        "Maximum Error": maximum,
        "Mean Error": mean,
        "Standard Deviation": std,
        "Zero Error Pixels": zero_error_pixels,
        "Zero Error Percentage": zero_error_percentage
    }

def save_statistics(stats):
    with open(
        "outputs/results.txt",
        "w"
    ) as f:
        f.write("S+PEE Prediction Results\n")
        f.write("========================\n\n")
        for key, value in stats.items():
            if isinstance(value, float):
                f.write(
                    f"{key}: {value:.4f}\n"
                )
            else:
                f.write(
                    f"{key}: {value}\n"
                )