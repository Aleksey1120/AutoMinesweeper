from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from utils import preprocess, find_positions


@dataclass
class PredictionResult:
    probabilities: torch.Tensor
    suggested_move: tuple[int, int]


class Predictor:
    def __init__(self, model):
        self.model = model

    def predict(self, field: np.ndarray) -> PredictionResult:
        field = np.expand_dims(field, axis=0)
        one_hot_field = torch.from_numpy(preprocess(field))
        with torch.no_grad():
            pred_proba = nn.functional.sigmoid(self.model(one_hot_field))
        mask = torch.from_numpy((field >= 0) & (field <= 8))
        positions = find_positions(pred_proba, mask)
        return PredictionResult(
            probabilities=pred_proba.squeeze(),
            suggested_move=positions[0]
        )
