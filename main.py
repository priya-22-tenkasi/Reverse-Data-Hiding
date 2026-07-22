import cv2
import numpy as np
from predictor import RhombusPredictor

# Read grayscale image
img = cv2.imread("quotes.bmp", cv2.IMREAD_GRAYSCALE)

# Generate predicted image and prediction error
predicted, error = RhombusPredictor.prediction_error(img)

# Print matrices (optional)
print("Predicted Pixel Matrix:\n", predicted)
print("Prediction Error Matrix:\n", error)

print("Minimum Error:", np.min(error))
print("Maximum Error:", np.max(error))

# -------------------------------
# Save Predicted Image
# -------------------------------
predicted_img = np.clip(predicted, 0, 255).astype(np.uint8)
cv2.imwrite("predicted_image.png", predicted_img)

error_display = np.abs(error)
error_display = cv2.normalize(error_display, None, 0, 255, cv2.NORM_MINMAX)
error_display = error_display.astype(np.uint8)

# Increase contrast
error_display = cv2.equalizeHist(error_display)

cv2.imwrite("prediction_error_map.png", error_display)

print("Images saved successfully!")