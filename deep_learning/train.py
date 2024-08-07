import os.path
import time

import numpy as np
import torch
from torch import nn

from deep_learning.data import GamesManager
from deep_learning.model import MinesweeperModel
from torch.utils.tensorboard import SummaryWriter
from deep_learning.early_stopping import EarlyStopping
from deep_learning.options.train_options import TrainOptions
from constants import IN_CHANNELS
from utils import position_to_idx, preprocess, set_seed, find_positions


def print_epoch_metrics(iter_number, elapsed_time, winrate, average_step_count):
    print(f'|{iter_number:^10}|{elapsed_time:^10.2f}|{winrate:^9.2%}|{average_step_count:^20.1f}|')


def print_result_table_headers():
    print('|   Iter   |   Time   | Winrate | Average step count |')


def log_metrics(iter_number, writer, winrate, average_step_count):
    writer.add_scalar('Winrate', winrate, iter_number)
    writer.add_scalar('Average step count', average_step_count, iter_number)


def train(opt, model, games_manager, loss_fn, optimizer, scheduler, device):
    if opt.verbose >= 2:
        print_result_table_headers()

    if opt.tb_dir is not None:
        writer = SummaryWriter(log_dir=os.path.join(opt.tb_dir, opt.tb_comment))
    early_stopping = EarlyStopping(model, opt)

    t = time.time()
    for it in range(opt.niter):
        optimizer.zero_grad()
        fields = np.stack(games_manager.get_fields())
        one_hot_fields = torch.from_numpy(preprocess(fields)).to(device)

        pred = model(one_hot_fields)
        pred_proba = nn.functional.sigmoid(pred)

        mask = torch.from_numpy((fields >= 0) & (fields <= 8))
        positions = find_positions(pred_proba.detach(), mask)

        results = torch.tensor(games_manager.step(positions), dtype=torch.float32).to(device)
        target = pred_proba.detach().clone()
        target[mask] = 0.0

        target_pos = []
        for i, (j, k) in enumerate(positions):
            target_pos.append(position_to_idx(i, j, k, opt.row_count, opt.column_count))
        target.put_(torch.tensor(target_pos).to(device), results)

        loss = loss_fn(pred_proba, target.to(device))
        loss.backward()
        optimizer.step()

        if opt.checkpoint_frequency and (it + 1) % opt.checkpoint_frequency == 0:
            torch.save(model.state_dict(), os.path.join(opt.output_dir, f'{opt.model_comment}_iter_{it}.bin'))

        if (it + 1) % opt.log_frequency == 0:
            print_epoch_metrics(it + 1,
                                time.time() - t,
                                games_manager.get_winrate(),
                                games_manager.get_average_step_count())
            if opt.tb_dir is not None:
                log_metrics(it + 1,
                            writer,
                            games_manager.get_winrate(),
                            games_manager.get_average_step_count())
            t = time.time()
            if early_stopping(games_manager.get_winrate()):
                break

        if (it + 1) % opt.lr_scheduler_frequency == 0:
            scheduler.step()
    early_stopping.rename_file()


if __name__ == '__main__':
    train_options = TrainOptions()
    opt = train_options.get_options()
    if opt.seed is not None:
        set_seed(opt.seed)
    if opt.verbose >= 1:
        train_options.print_options()

    device = torch.device('cuda' if torch.cuda.is_available() and opt.cuda else 'cpu')

    games_manager = GamesManager(opt.batch_size, opt.row_count, opt.column_count, opt.mine_count, opt.games_for_metrics)
    model = MinesweeperModel(IN_CHANNELS, opt.hidden_channels, opt.block_count).to(device)
    if opt.checkpoint_path:
        model.load_state_dict(torch.load(opt.checkpoint_path))
    model.train()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt.gamma)

    train(opt, model, games_manager, loss_fn, optimizer, scheduler, device)
