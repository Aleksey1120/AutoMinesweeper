import numpy as np


# 0-8 – mine count;
# 9 – mine.
# 10 – unrevealed.


class Minesweeper:

    def __init__(self, row_count=13, column_count=17, mine_count=36):
        self.row_count = row_count
        self.column_count = column_count
        self.mine_count = mine_count

        if row_count * column_count < mine_count - 4:
            raise ValueError('Too many mines.')

        self._field = None
        self.score = None
        self._field_mask = np.zeros((self.row_count, self.column_count))

    def create_field(self, empty_cell_x, empty_cell_y):
        field = np.zeros(self.row_count * self.column_count)

        empty_idx = empty_cell_x * self.column_count + empty_cell_y

        mine_idx = np.random.choice(self.row_count * self.column_count, self.mine_count, replace=False)
        while empty_idx in mine_idx:
            mine_idx = np.random.choice(self.row_count * self.column_count, self.mine_count, replace=False)

        field[mine_idx] = 9

        field = field.reshape(self.row_count, self.column_count)
        for i in range(self.row_count):
            for j in range(self.column_count):
                if field[i, j] == 9:
                    for k in (-1, 0, 1):
                        for l in (-1, 0, 1):
                            if 0 <= i + k < self.row_count and 0 <= j + l < self.column_count:
                                if field[i + k, j + l] != 9:
                                    field[i + k, j + l] += 1

        return field

    def get_field(self):
        visible = np.ones((self.row_count, self.column_count)) * 10
        visible[self._field_mask.nonzero()] = self._field[self._field_mask.nonzero()]
        return visible

    def _get_hidden_field(self):
        return self._field

    def recursive_open(self, x, y):
        if x < 0 or x >= self.row_count or y < 0 or y >= self.column_count:
            return
        if self._field_mask[x, y] == 0:
            self._field_mask[x, y] = 1
            self.score += 1
            if self._field[x, y] == 0:
                for i, j in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                    self.recursive_open(x + i, y + j)

    def step(self, x, y):
        if self._field_mask[x, y] == 1:
            return 'checked', self.get_field()
        if self._field is None:
            self._field = self.create_field(x, y)
            self.score = 0

        if self._field[x, y] == 9:
            return 'lose', self._get_hidden_field()

        self.recursive_open(x, y)
        if self.score == self.row_count * self.column_count - self.mine_count:
            return 'win', self._get_hidden_field()
        return 'continue', self.get_field()


def main():
    m = Minesweeper()
    while True:
        a = input('Point: ')
        x, y = map(int, a.split())
        res = m.step(x, y)
        print(res[0])
        print(res[1])
        if res[0] == 'lose' or res[0] == 'win':
            break


if __name__ == '__main__':
    main()
