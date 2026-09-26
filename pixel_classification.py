import numpy as np
class PixelClassifier:
    def __init__(self, Tn=-1, Tp=2):
        self.Tn = Tn
        self.Tp = Tp
    def classify(self, error, mask=None):
        if mask is None:
            mask = np.ones(error.shape, dtype=bool)
        expansion_mask = (
            ((error == self.Tn) | (error == self.Tp))
            & mask
        )
        shifted_mask = (
            ((error < self.Tn) | (error > self.Tp))
            & mask
        )
        inner_mask = (
            (error > self.Tn)
            & (error < self.Tp)
            & mask
        )
        return {
            "expansion": expansion_mask,
            "shifted": shifted_mask,
            "inner": inner_mask
        }
    def statistics(self, error, mask=None):
        result = self.classify(error, mask)
        expansion_count = np.sum(result["expansion"])
        shifted_count = np.sum(result["shifted"])
        inner_count = np.sum(result["inner"])
        return {
            "expansion_pixels": int(expansion_count),
            "shifted_pixels": int(shifted_count),
            "inner_pixels": int(inner_count)
        }