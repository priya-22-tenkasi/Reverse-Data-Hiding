import numpy as np
import cv2
def create_checkerboard(image, layer=0):
    h, w = image.shape
    checkerboard = np.zeros_like(image)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if (i + j) % 2 == layer:
                checkerboard[i, j] = image[i, j]
    return checkerboard

def create_checkerboard_mask(image, layer=0):
    h, w = image.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            if (i + j) % 2 == layer:
                mask[i, j] = 255
    return mask
