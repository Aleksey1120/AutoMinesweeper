import sys

import torch
from PyQt6.QtCore import QTimer, Qt
from PyQt6 import QtCore
from PyQt6.QtWidgets import QWidget, QPushButton, QLineEdit, QApplication, QTableWidget, QTableWidgetItem
from PyQt6 import QtWidgets, QtGui

from game import Minesweeper
from constants import IN_CHANNELS
from deep_learning.model import MinesweeperModel
import minesweeper_app.config as cfg
from minesweeper_app.predictor import Predictor


def get_gray_to_red_gradient(value):
    if not 0 <= value <= 1:
        raise ValueError('Value must be in the range [0, 1]')

    r = int(255 * 0.5 * (1 + value))
    g = int(255 * 0.5 * (1 - value))
    b = int(255 * 0.5 * (1 - value))
    return r, g, b


def show_game_result(title, text):
    msg = QtWidgets.QMessageBox()
    msg.setWindowTitle(title)
    msg.setText(text)
    msg.setIcon(QtWidgets.QMessageBox.Icon.Information)
    msg.exec()


class MinesweeperGUI(QWidget):
    def __init__(self, row_count, column_count, mine_count, predictor: Predictor):
        super().__init__()

        self.row_count = row_count
        self.column_count = column_count
        self.mine_count = mine_count
        self.cell_size = 60
        self.game = None
        self.predictor = predictor
        self.is_running = False

        self.init_ui()

    def init_ui(self):
        self.setFont(QtGui.QFont('Segoe UI', 14))
        self.vertical_layout = QtWidgets.QVBoxLayout()
        self.horizontal_layout = QtWidgets.QHBoxLayout()

        self.setWindowTitle('AutoMinesweeper')

        self.tableWidget = QTableWidget(self)
        self.tableWidget.setRowCount(self.row_count)
        self.tableWidget.setColumnCount(self.column_count)

        self.tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tableWidget.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.tableWidget.setFixedSize(min(10, self.column_count) * self.cell_size + 2,
                                      min(10, self.row_count) * self.cell_size + 2)

        self.tableWidget.horizontalHeader().setVisible(False)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(self.cell_size)
        self.tableWidget.horizontalHeader().setMinimumSectionSize(self.cell_size)
        self.tableWidget.horizontalHeader().setMaximumSectionSize(self.cell_size)
        self.tableWidget.verticalHeader().setVisible(False)
        self.tableWidget.verticalHeader().setDefaultSectionSize(self.cell_size)
        self.tableWidget.verticalHeader().setMinimumSectionSize(self.cell_size)
        self.tableWidget.verticalHeader().setMaximumSectionSize(self.cell_size)
        self.tableWidget.setFont(QtGui.QFont('Segoe UI', 40))

        self.horizontal_layout.addWidget(self.tableWidget)
        self.vertical_layout.addLayout(self.horizontal_layout)

        self.interval_input = QLineEdit(self)
        self.interval_input.setPlaceholderText('Interval(sec)')
        self.vertical_layout.addWidget(self.interval_input)

        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.start_game)
        self.vertical_layout.addWidget(self.start_button)

        self.setLayout(self.vertical_layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_game)

    def start_game(self):
        try:
            if self.game is None:
                self.game = Minesweeper(row_count=self.row_count,
                                        column_count=self.column_count,
                                        mine_count=self.mine_count)
            self.interval = int(float(self.interval_input.text()) * 1000)
            self.timer.start(self.interval)

            self.start_button.setText('Pause')
            self.start_button.clicked.disconnect(self.start_game)
            self.start_button.clicked.connect(self.pause_game)

            self.update_game()
        except ValueError:
            print('Invalid format')
            return

    def pause_game(self):
        if self.timer.isActive():
            self.timer.stop()
            self.start_button.setText('Continue')
            self.start_button.clicked.disconnect(self.pause_game)
            self.start_button.clicked.connect(self.start_game)

    def update_game(self):
        field = self.game.get_field()
        predict = self.predictor.predict(field)
        self.update_gui(field, predict.probabilities)

        res, _ = self.game.step(*predict.suggested_move)

        if res in ['lose', 'win']:
            self.update_gui(self.game._get_hidden_field(), predict.probabilities)
            self.timer.stop()
            self.game = None
            self.start_button.setText('Start')
            self.start_button.clicked.disconnect(self.pause_game)
            self.start_button.clicked.connect(self.start_game)
            title = 'Lose' if res == 'lose' else 'Win'
            text = 'That was close! (Or maybe not...)' if res == 'lose' \
                else 'The AI cleared the board without detonating any mines'
            show_game_result(title, text)

    def update_gui(self, field, probabilities):
        for row in range(self.row_count):
            for col in range(self.column_count):
                cell_value = field[row, col]
                if cell_value in range(1, 8):
                    item = QTableWidgetItem(str(cell_value))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    font = QtGui.QFont()
                    font.setBold(True)
                    font.setKerning(True)
                    item.setFont(font)
                    brush = QtGui.QBrush(QtGui.QColor(*cfg.COLORS[cell_value]))
                    item.setForeground(brush)
                    self.tableWidget.setItem(row, col, item)
                else:
                    item = QTableWidgetItem('')
                    color = get_gray_to_red_gradient(probabilities[row, col]) if cell_value == 10 else cfg.COLORS[
                        cell_value]
                    brush = QtGui.QBrush(QtGui.QColor(*color))
                    brush.setStyle(QtCore.Qt.BrushStyle.SolidPattern)
                    item.setBackground(brush)
                    self.tableWidget.setItem(row, col, item)


if __name__ == '__main__':
    model = MinesweeperModel(IN_CHANNELS, cfg.MODEL_HIDDEN_CHANNELS, cfg.MODEL_BLOCK_COUNT)
    model.load_state_dict(torch.load(cfg.MODEL_PATH))
    predictor = Predictor(model)

    app = QApplication(sys.argv)
    ex = MinesweeperGUI(cfg.GAME_ROW_COUNT, cfg.GAME_COLUMN_COUNT, cfg.GAME_MINE_COUNT, predictor)
    ex.show()
    sys.exit(app.exec())
