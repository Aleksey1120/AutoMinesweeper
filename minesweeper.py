import time

import numpy as np

from numba import njit


# 0-8 – mine count;
# 9 – mine.
# 10 – unrevealed.


class Minesweeper:

    def __init__(self, row_count=13, column_count=17, mine_count=36, first_step=True):
        self.row_count = row_count
        self.column_count = column_count
        self.mine_count = mine_count

        if row_count * column_count < mine_count - 4:
            raise ValueError('Too many mines.')

        self._field = None
        self.score = None
        self._field_mask = np.zeros((self.row_count, self.column_count))
        if first_step:
            self.step(np.random.randint(0, self.row_count), np.random.randint(0, self.column_count))

    def create_field(self, empty_cell_x, empty_cell_y):
        field = np.zeros(self.row_count * self.column_count)

        empty_idx = empty_cell_x * self.column_count + empty_cell_y

        mine_idx = np.random.choice(self.row_count * self.column_count, self.mine_count, replace=False)
        while empty_idx in mine_idx:
            mine_idx = np.random.choice(self.row_count * self.column_count, self.mine_count, replace=False)

        field[mine_idx] = 9

        field = field.reshape(self.row_count, self.column_count)

        update_neighbor_counts(field)
        return field

    def get_field(self):
        return np.where(self._field_mask, self._field, 10)

    def _get_hidden_field(self):
        return self._field

    def step(self, x, y):
        if self._field_mask[x, y] == 1:
            return 'checked', self.get_field()
        if self._field is None:
            self._field = self.create_field(x, y)
            self.score = 0

        if self._field[x, y] == 9:
            return 'lose', self._get_hidden_field()

        open_cells(self._field, self._field_mask, x, y)
        if self.score == self.row_count * self.column_count - self.mine_count:
            return 'win', self._get_hidden_field()
        return 'continue', self.get_field()


@njit(fastmath=True)
def update_neighbor_counts(field):
    row_count, column_count = field.shape
    for i in range(row_count):
        for j in range(column_count):
            if field[i, j] == 9:
                for k in (-1, 0, 1):
                    for l in (-1, 0, 1):
                        if 0 <= i + k < row_count and 0 <= j + l < column_count:
                            if field[i + k, j + l] != 9:
                                field[i + k, j + l] += 1


@njit(fastmath=True)
def open_cells(field, field_mask, x, y):
    coords = [(x, y)]
    row_count, column_count = field.shape
    score = 0
    for cur_x, cur_y in coords:
        if cur_x < 0 or cur_x >= row_count or cur_y < 0 or cur_y >= column_count:
            continue
        if field_mask[cur_x, cur_y] == 0:
            field_mask[cur_x, cur_y] = 1
            score += 1
            if field[cur_x, cur_y] == 0:
                for i, j in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                    coords.append((cur_x + i, cur_y + j))
    return score


def main():
    m = Minesweeper()
    print(m.get_field())
    while True:
        a = input('Point: ')
        x, y = map(int, a.split())
        res = m.step(x, y)
        print(res[0])
        print(res[1])
        if res[0] == 'lose' or res[0] == 'win':
            break


def play_game(p_misstep):
    m = Minesweeper(13, 17, 36, first_step=True)
    field = m._get_hidden_field()
    for i in range(13):
        for j in range(17):
            if field[i, j] != 9:
                res, _ = m.step(i, j)
                if res == 'win':
                    return
            elif np.random.random() < p_misstep:
                res, _ = m.step(i, j)
                return


def speed_test(n_iter, p_misstep):
    total_time = 0
    for _ in range(n_iter):
        start = time.time()
        play_game(p_misstep)
        total_time += time.time() - start
    return total_time / n_iter


if __name__ == '__main__':
    # main()
    print(speed_test(5000, 0.01))
