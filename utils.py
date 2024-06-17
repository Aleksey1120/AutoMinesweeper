import random
import numpy as np
import torch
from constants import IN_CHANNELS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def preprocess(fields: np.ndarray):
    fields = np.where(fields == 10, 9, fields)
    games, rows, cols = np.indices(fields.shape)
    one_hot_field = np.zeros((fields.shape[0], IN_CHANNELS, fields.shape[1], fields.shape[2]), dtype=np.float32)
    one_hot_field[games, fields, rows, cols] = 1.0
    return one_hot_field


def position_to_idx(i, j, k, row_count, column_count):
    return i * row_count * column_count + j * column_count + k


def fild_positions(probabilities, mask):
    masked_probabilities = probabilities.clone()
    masked_probabilities[mask] = torch.inf

    positions = torch.argmin(masked_probabilities.reshape(masked_probabilities.shape[0], -1), dim=1).tolist()
    column_count = masked_probabilities.shape[2]
    return [(position // column_count, position % column_count) for position in positions]
