import time

import torch
from data import GamesManager
from model import Model
from torch import nn

IN_CHANNELS = 11


def preprocess(field: torch.tensor):
    one_hot_field = torch.zeros(IN_CHANNELS, field.size(0), field.size(1))
    return one_hot_field.scatter_(0, field.unsqueeze(0).to(torch.int64), 1.0)


def position_to_idx(i, j, k, row_count, column_count):
    return i * row_count * column_count + j * column_count + k


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

row_count = 8
column_count = 8
mine_count = 10

lr = 1e-3
n_iter = 5000
n_games = 512
n_games_for_metrics = n_games * 5

print_every = 100

gm = GamesManager(n_games, row_count, column_count, mine_count, n_games_for_metrics)
m = Model(IN_CHANNELS, 64 * 3).to(device)
m.train()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(m.parameters(), lr=lr)
t = time.time()
for it in range(n_iter):
    optimizer.zero_grad()
    fields = gm.get_fields()
    one_hot_fields = torch.stack([preprocess(field) for field in fields]).to(device)

    pred = m(one_hot_fields)
    pred_proba = nn.functional.sigmoid(pred)

    mask = torch.stack([(field >= 0) & (field <= 8) for field in fields]).to(device)
    tmp = pred_proba.detach().clone()
    tmp[mask] = torch.inf

    positions = torch.argmin(tmp.reshape(tmp.shape[0], -1), dim=1).tolist()
    positions = [(position // column_count, position % column_count) for position in positions]

    results = torch.tensor(gm.step(positions), dtype=torch.float32).to(device)

    target = pred_proba.detach().clone()
    target[mask] = 0.0

    target_pos = []
    for i, (j, k) in enumerate(positions):
        target_pos.append(position_to_idx(i, j, k, row_count, column_count))
    target.put_(torch.tensor(target_pos).to(device), results)
    loss = loss_fn(pred_proba, target.to(device))
    loss.backward()
    optimizer.step()

    if (it + 1) % print_every == 0:
        print(
            f'{it + 1}: '
            f'Time: {time.time() - t:.1f}, '
            f'WR = {gm.get_winrate():.3f}, '
            f'Average Step Count = {gm.get_average_step_count():.1f}, '
            f'Loss = {loss.item():.3f}, '
            f'Total played games = {gm.total_played_games}')
        t = time.time()
