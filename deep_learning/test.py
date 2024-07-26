import time

import numpy as np
import torch
from torch import nn

from deep_learning.options.test_options import TestOptions
from deep_learning.data import GamesManager
from deep_learning.model import MinesweeperModel
from constants import IN_CHANNELS
from utils import preprocess, set_seed, find_positions


def test(opt, model, games_manager: GamesManager, device):
    start_time = time.time()

    while games_manager.total_played_games < opt.game_count:
        fields = np.stack(games_manager.get_fields())
        one_hot_fields = torch.from_numpy(preprocess(fields)).to(device)
        with torch.no_grad():
            pred = model(one_hot_fields)
        pred_proba = nn.functional.sigmoid(pred)

        mask = torch.from_numpy((fields >= 0) & (fields <= 8)).to(device)
        positions = find_positions(pred_proba, mask)

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
