import numpy as np
import torch
from collections import deque

from minesweeper import Minesweeper


class GamesManager:

    def __init__(self, game_count, row_count, column_count, mine_count, n_games_for_metrics):
        self.game_count = game_count
        self.row_count = row_count
        self.column_count = column_count
        self.mine_count = mine_count
        self.n_games_for_metrics = n_games_for_metrics
        self.games = [Minesweeper(self.row_count, self.column_count, self.mine_count) for _ in range(game_count)]

        self.game_result_deque = deque(maxlen=n_games_for_metrics)
        self.step_count_deque = deque(maxlen=n_games_for_metrics)
        self.step_count = np.zeros(game_count, dtype=np.int32)
        self.total_played_games = 0

    def get_winrate(self):
        if self.game_result_deque:
            return np.mean(self.game_result_deque)
        return 0.0

    def get_average_step_count(self):
        if self.step_count_deque:
            return np.mean(self.step_count_deque)
        return 0.0

    def get_fields(self):
        return [torch.from_numpy(game.get_field()).to(torch.float32) for game in self.games]

    def step(self, positions: list[tuple[int, int]]):
        str_to_int_result = {
            'win': 0,
            'lose': 1,
            'continue': 0,
            'checked': 0
        }
        results = []
        for i in range(len(self.games)):
            result, _ = self.games[i].step(positions[i][0], positions[i][1])
            results.append(str_to_int_result[result])
            match result:
                case 'win':
                    self.games[i] = Minesweeper(self.row_count, self.column_count, self.mine_count)
                    self.game_result_deque.append(1)
                    self.step_count_deque.append(self.step_count[i] + 1)
                    self.step_count[i] = 0
                    self.total_played_games += 1
                case 'lose':
                    self.games[i] = Minesweeper(self.row_count, self.column_count, self.mine_count)
                    self.game_result_deque.append(0)
                    self.step_count_deque.append(self.step_count[i] + 1)
                    self.step_count[i] = 0
                    self.total_played_games += 1
                case 'continue':
                    self.step_count[i] += 1
                case 'checked':
                    print('Checked cell was chosen.')
                    self.games[i] = Minesweeper(self.row_count, self.column_count, self.mine_count)
                    self.step_count[i] = 0
        return results
