import numpy as np
class LocalVariance:

    @staticmethod
    def calculate(image, layer=0):
        image = image.astype(np.int32)
        h, w = image.shape
        variance_map = np.zeros((h, w), dtype=np.int32)
        for i in range(1, h - 2):
            for j in range(1, w - 2):
                if (i + j) % 2 != layer:
                    continue
                v1 = image[i - 1, j]
                v2 = image[i, j - 1]
                v4 = image[i, j + 1]
                p = image[i, j]
                v3 = image[i + 1, j]
                u1 = image[i + 1, j - 1]
                u4 = image[i + 1, j + 1]
                u2 = image[i + 2, j - 1]
                u3 = image[i + 2, j]
                u5 = image[i + 2, j + 1]
                u6 = image[i - 1, j + 2]
                u7 = image[i, j + 2]
                u8 = image[i + 1, j + 2]
                u9 = image[i + 2, j + 2]
                value = (
                    abs(v2 - u1)
                    + abs(u1 - u2)
                    + abs(v1 - p)
                    + abs(p - v3)
                    + abs(v3 - u3)
                    + abs(v4 - u4)
                    + abs(u4 - u5)
                    + abs(u6 - u7)
                    + abs(u7 - u8)
                    + abs(u8 - u9)
                    + abs(v2 - p)
                    + abs(p - v4)
                    + abs(v4 - u7)
                    + abs(u1 - v3)
                    + abs(v3 - u4)
                    + abs(u4 - u8)
                    + abs(u2 - u3)
                    + abs(u3 - u5)
                    + abs(u5 - u9)
                )
                variance_map[i, j] = value
        return variance_map

    @staticmethod
    def get_selected_pixels(image, variance_map, layer=0, number_of_pixels=None):
        mask = np.zeros(image.shape, dtype=bool)
        h, w = image.shape
        candidates = []
        for i in range(1, h - 2):
            for j in range(1, w - 2):
                if (i + j) % 2 == layer:
                    candidates.append(
                        (variance_map[i, j], i, j)
                    )

        candidates.sort(key=lambda x: x[0])
        if number_of_pixels is not None:
            candidates = candidates[:number_of_pixels]
        for variance, i, j in candidates:
            mask[i, j] = True
        return mask