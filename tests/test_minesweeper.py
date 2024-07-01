from unittest import TestCase
from game import Minesweeper
import numpy as np


def games_generator(n_games, first_step=True):
    for _ in range(n_games):
        row_count, column_count = np.random.randint(8, 60, 2)
        mine_count = int(row_count * column_count * (0.15 + np.random.random() * 0.05))
        yield Minesweeper(row_count, column_count, mine_count, first_step=first_step)


class TestMinesweeper(TestCase):
    def test_win(self):
        for game in games_generator(1000):
            field = game._get_hidden_field()
            for i in range(game.row_count):
                for j in range(game.column_count):
                    if field[i, j] != 9:
                        res, _ = game.step(i, j)
                        if res == 'win':
                            continue
                        elif res == 'lose':
                            self.fail()

    def test_lose(self):
        for game in games_generator(1000):
            field = game._get_hidden_field()
            for i in range(game.row_count):
                for j in range(game.column_count):
                    if field[i, j] == 9:
                        res, _ = game.step(i, j)
                        self.assertTrue(res == 'lose')

    def test_first_step(self):
        for game in games_generator(1000, False):
            res, _ = game.step(np.random.randint(0, game.row_count), np.random.randint(0, game.column_count))
            self.assertTrue(res == 'continue')

    def test_reveal(self):
        for game in games_generator(1000, False):
            while True:
                x, y = np.random.randint(0, game.row_count), np.random.randint(0, game.column_count)
                res, _ = game.step(x, y)
                if res == 'lose':
                    break
                mask = game._field_mask
                self.assertTrue(mask[x, y] == 1)

    def test_generation(self):
        for game in games_generator(1000):
            field = game._get_hidden_field()
            self.assertTrue(field.shape == (game.row_count, game.column_count))
            self.assertTrue(np.sum(field == 9) == game.mine_count)
            for i in range(game.row_count):
                for j in range(game.column_count):
                    if field[i, j] != 9:
                        min_x = max(0, i - 1)
                        min_y = max(0, j - 1)
                        max_x = min(game.row_count, i + 2)
                        max_y = min(game.column_count, j + 2)
                        self.assertTrue(np.sum(field[min_x: max_x, min_y: max_y] == 9) == field[i, j])

    def test_get_field(self):
        for game in games_generator(1000):
            field = game.get_field()
            hidden_field = game._get_hidden_field()
            mask = game._field_mask
            self.assertTrue(np.all(field == np.where(mask, hidden_field, 10)))
