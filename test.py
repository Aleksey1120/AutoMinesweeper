import random
import time

import numpy as np
import torch
from data import GamesManager
from model import MinesweeperModel
from torch import nn

from options.test_options import TestOptions

IN_CHANNELS = 11


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def preprocess(field: torch.tensor):
    one_hot_field = torch.zeros(IN_CHANNELS, field.size(0), field.size(1))
    return one_hot_field.scatter_(0, field.unsqueeze(0).to(torch.int64), 1.0)


def position_to_idx(i, j, k, row_count, column_count):
    return i * row_count * column_count + j * column_count + k


def test(opt, model, games_manager: GamesManager, device):
    start_time = time.time()

    while games_manager.total_played_games < opt.game_count:
        fields = games_manager.get_fields()
        one_hot_fields = torch.stack([preprocess(field) for field in fields]).to(device)
        with torch.no_grad():
            pred = model(one_hot_fields)
        pred_proba = nn.functional.sigmoid(pred)

        mask = torch.stack([(field >= 0) & (field <= 8) for field in fields]).to(device)
        pred_proba[mask] = torch.inf

        positions = torch.argmin(pred_proba.reshape(pred_proba.shape[0], -1), dim=1).tolist()
        positions = [(position // opt.column_count, position % opt.column_count) for position in positions]

        games_manager.step(positions)
    print(f'Games played: {games_manager.total_played_games}.')
    print(f'Time: {time.time() - start_time:.2f}.')
    print(f'Winrate: {games_manager.get_winrate():.2%}.')
    print(f'Average step count: {games_manager.get_average_step_count():.1f}.')


if __name__ == '__main__':
    test_options = TestOptions()
    opt = test_options.get_options()
    if opt.seed is not None:
        set_seed(opt.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and opt.cuda else 'cpu')

    games_manager = GamesManager(opt.batch_size, opt.row_count, opt.column_count, opt.mine_count, None)
    model = MinesweeperModel(IN_CHANNELS, opt.hidden_channels, opt.block_count).to(device)
    if opt.checkpoint_path:
        model.load_state_dict(torch.load(opt.checkpoint_path))
    model.eval()
    test(opt, model, games_manager, device)
