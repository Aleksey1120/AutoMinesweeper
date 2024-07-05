from unittest import TestCase
import numpy as np
import torch

from minesweeper_app.predictor import Predictor, PredictionResult
from deep_learning.model import MinesweeperModel
from constants import IN_CHANNELS


class TestPredictor(TestCase):

    def setUp(self):
        self.model = MinesweeperModel(IN_CHANNELS, 1, 0)
        self.predictor = Predictor(self.model)

    def test_predict(self):
        field_shape = (8, 8)

        field = np.random.randint(IN_CHANNELS + 1, size=field_shape)
        result = self.predictor.predict(field)
        self.assertIsInstance(result, PredictionResult)

        self.assertTrue(result.probabilities.shape == field_shape)
        self.assertTrue(result.probabilities.max() <= 1.0)
        self.assertTrue(result.probabilities.min() >= 0.0)
