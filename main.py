import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from preprocessing import (
    load_image,
    preprocess_image
)

from predictor import RhombusPredictor

from checkerboard import (
    create_checkerboard
)

from visualization import (
    save_all_images
)

from metrics import (
    calculate_error_statistics,
    save_statistics
)


# ============================================================
# 1. CREATE OUTPUT FOLDER
# ============================================================

os.makedirs("outputs", exist_ok=True)


# ============================================================
# 2. LOAD IMAGE
# ============================================================

image_path = "quotes.bmp"

img = load_image(image_path)
img = preprocess_image(img)

print("==============================================")
print("       REVERSIBLE DATA HIDING PROJECT")
print("==============================================")

print("\nImage loaded successfully.")
print("Image shape:", img.shape)
print("Image data type:", img.dtype)


# ============================================================
# 3. ORIGINAL IMAGE INFORMATION
# ============================================================

print("\n==============================================")
print("       ORIGINAL IMAGE INFORMATION")
print("==============================================")

print("Image Height       :", img.shape[0])
print("Image Width        :", img.shape[1])
print("Minimum Pixel Value:", np.min(img))
print("Maximum Pixel Value:", np.max(img))
print("Mean Pixel Value   :", round(np.mean(img), 4))


# ============================================================
# 4. ORIGINAL PIXEL MATRIX
# ============================================================

print("\n==============================================")
print("       ORIGINAL PIXEL MATRIX (10 x 10)")
print("==============================================")

print(img[:10, :10])

np.savetxt(
    "outputs/original_matrix.txt",
    img,
    fmt="%d"
)


# ============================================================
# 5. RHOMBUS PREDICTION
# ============================================================

predicted, error = RhombusPredictor.prediction_error(img)

print("\n==============================================")
print("       RHOMBUS PREDICTION COMPLETED")
print("==============================================")


# ============================================================
# 6. PREDICTED IMAGE INFORMATION
# ============================================================

print("\n==============================================")
print("       PREDICTED IMAGE INFORMATION")
print("==============================================")

print("Minimum Predicted Value:", np.min(predicted))
print("Maximum Predicted Value:", np.max(predicted))
print("Mean Predicted Value   :", round(np.mean(predicted), 4))


# ============================================================
# 7. PREDICTED PIXEL MATRIX
# ============================================================

print("\n==============================================")
print("       PREDICTED PIXEL MATRIX (10 x 10)")
print("==============================================")

print(predicted[:10, :10])

np.savetxt(
    "outputs/predicted_matrix.txt",
    predicted,
    fmt="%d"
)


# ============================================================
# 8. PREDICTION ERROR INFORMATION
# ============================================================

print("\n==============================================")
print("       PREDICTION ERROR INFORMATION")
print("==============================================")

print("Minimum Error:", np.min(error))
print("Maximum Error:", np.max(error))
print("Mean Error   :", round(np.mean(error), 4))


# ============================================================
# 9. PREDICTION ERROR MATRIX
# ============================================================

print("\n==============================================")
print("       PREDICTION ERROR MATRIX (10 x 10)")
print("==============================================")

print(error[:10, :10])

np.savetxt(
    "outputs/prediction_error_matrix.txt",
    error,
    fmt="%d"
)


# ============================================================
# 10. PIXEL CLASSIFICATION
# ============================================================
#
# NOTE:
# The following four classes are used for VISUALIZATION:
#
# 0 -> Zero Error      : error == 0
# 1 -> Low Error       : |error| = 1 or 2
# 2 -> Medium Error    : |error| = 3 to 5
# 3 -> High Error      : |error| > 5
#
# This gives a presentation-friendly classification map
# similar to the example provided.
#
# This is a visualization classification. The actual S+PEE
# embedding thresholds Tn and Tp are handled separately.
# ============================================================

classification = np.zeros_like(error, dtype=np.uint8)

absolute_error = np.abs(error)


# ------------------------------------------------------------
# Class 0: Zero Error
# ------------------------------------------------------------

classification[absolute_error == 0] = 0


# ------------------------------------------------------------
# Class 1: Low Error
# ------------------------------------------------------------

classification[
    (absolute_error >= 1) &
    (absolute_error <= 2)
] = 1


# ------------------------------------------------------------
# Class 2: Medium Error
# ------------------------------------------------------------

classification[
    (absolute_error >= 3) &
    (absolute_error <= 5)
] = 2


# ------------------------------------------------------------
# Class 3: High Error
# ------------------------------------------------------------

classification[
    absolute_error > 5
] = 3


# ============================================================
# 11. PRINT CLASSIFICATION RESULTS
# ============================================================

print("\n==============================================")
print("       PIXEL CLASSIFICATION")
print("==============================================")

print("Zero Error pixels   :", np.sum(classification == 0))
print("Low Error pixels    :", np.sum(classification == 1))
print("Medium Error pixels :", np.sum(classification == 2))
print("High Error pixels   :", np.sum(classification == 3))


# ============================================================
# 12. CLASSIFICATION MATRIX
# ============================================================

print("\n==============================================")
print("       CLASSIFICATION MATRIX (10 x 10)")
print("==============================================")

print(classification[:10, :10])

np.savetxt(
    "outputs/pixel_classification.txt",
    classification,
    fmt="%d"
)


# ============================================================
# 13. CREATE FULL PIXEL CLASSIFICATION IMAGE
# ============================================================

# RGB image
classification_image = np.zeros(
    (classification.shape[0],
     classification.shape[1],
     3),
    dtype=np.uint8
)


# Zero Error -> White
classification_image[classification == 0] = [255, 255, 255]

# Low Error -> Green
classification_image[classification == 1] = [150, 210, 120]

# Medium Error -> Yellow
classification_image[classification == 2] = [255, 210, 100]

# High Error -> Red
classification_image[classification == 3] = [240, 90, 80]


cv2.imwrite(
    "outputs/pixel_classification.png",
    cv2.cvtColor(classification_image, cv2.COLOR_RGB2BGR)
)

print("\nPixel classification image saved:")
print("outputs/pixel_classification.png")


# ============================================================
# 14. CREATE PRESENTATION-STYLE CLASSIFICATION MAP
# ============================================================
#
# We display a small region of the image with the actual
# prediction-error values written inside each cell.
#
# This produces an output similar to your reference image.
# ============================================================

display_size = 12

error_small = error[:display_size, :display_size]
class_small = classification[:display_size, :display_size]


# Colors:
# 0 = White
# 1 = Green
# 2 = Yellow
# 3 = Red

cmap = ListedColormap([
    "white",
    "lightgreen",
    "khaki",
    "salmon"
])


fig, ax = plt.subplots(figsize=(10, 8))

ax.imshow(
    class_small,
    cmap=cmap,
    vmin=0,
    vmax=3
)


# ------------------------------------------------------------
# Add prediction-error value inside every cell
# ------------------------------------------------------------

for i in range(display_size):
    for j in range(display_size):

        value = error_small[i, j]

        ax.text(
            j,
            i,
            str(int(value)),
            ha="center",
            va="center",
            fontsize=9,
            color="black"
        )


# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------

ax.set_xticks(np.arange(-0.5, display_size, 1), minor=True)
ax.set_yticks(np.arange(-0.5, display_size, 1), minor=True)

ax.grid(
    which="minor",
    color="gray",
    linestyle="-",
    linewidth=0.8
)

ax.tick_params(
    which="minor",
    bottom=False,
    left=False
)


# ------------------------------------------------------------
# Remove normal axis ticks
# ------------------------------------------------------------

ax.set_xticks([])
ax.set_yticks([])


# ------------------------------------------------------------
# Title
# ------------------------------------------------------------

ax.set_title(
    "Pixel Classification Map",
    fontsize=16,
    fontweight="bold"
)


# ============================================================
# 15. CREATE LEGEND
# ============================================================

legend_elements = [

    Patch(
        facecolor="lightgreen",
        edgecolor="gray",
        label="Low Error (Suitable)"
    ),

    Patch(
        facecolor="khaki",
        edgecolor="gray",
        label="Medium Error (Marginal)"
    ),

    Patch(
        facecolor="salmon",
        edgecolor="gray",
        label="High Error (Avoid)"
    ),

    Patch(
        facecolor="white",
        edgecolor="gray",
        label="Zero Error (Exact Prediction)"
    )
]


ax.legend(
    handles=legend_elements,
    loc="upper left",
    bbox_to_anchor=(1.02, 1),
    frameon=True
)


plt.tight_layout()

plt.savefig(
    "outputs/pixel_classification_example.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Presentation-style classification map saved:")
print("outputs/pixel_classification_example.png")


# ============================================================
# 16. CHECKERBOARD PARTITION
# ============================================================

checkerboard = create_checkerboard(
    img,
    layer=0
)

print("\n==============================================")
print("       CHECKERBOARD PARTITION")
print("==============================================")

print("Checkerboard partition completed.")


# ============================================================
# 17. PREDICTION ERROR STATISTICS
# ============================================================

stats = calculate_error_statistics(error)

print("\n==============================================")
print("       PREDICTION ERROR STATISTICS")
print("==============================================")

for key, value in stats.items():
    print(f"{key}: {value}")


# ============================================================
# 18. SAVE VISUAL OUTPUTS
# ============================================================

save_all_images(
    img,
    predicted,
    error,
    checkerboard
)

save_statistics(stats)


# ============================================================
# 19. FINAL OUTPUT SUMMARY
# ============================================================

print("\n==============================================")
print("       ALL OUTPUTS GENERATED SUCCESSFULLY")
print("==============================================")

print("\nGenerated files:")

print("1.  outputs/original.png")
print("2.  outputs/predicted.png")
print("3.  outputs/prediction_error.png")
print("4.  outputs/checkerboard.png")
print("5.  outputs/error_histogram.png")
print("6.  outputs/results.txt")
print("7.  outputs/original_matrix.txt")
print("8.  outputs/predicted_matrix.txt")
print("9.  outputs/prediction_error_matrix.txt")
print("10. outputs/pixel_classification.txt")
print("11. outputs/pixel_classification.png")
print("12. outputs/pixel_classification_example.png")

print("\n==============================================")
print("       PROCESS COMPLETED")
print("==============================================")