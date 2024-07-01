import torch
import os


class EarlyStopping:

    def __init__(self, model, options):
        self.trigger_times = 0
        self.best_winrate = -torch.inf
        self.save_best_model = options.save_best_model
        self.model = model
        self.early_stop = options.early_stop

        self.output_dir = options.output_dir
        self.model_comment = options.model_comment

    def __call__(self, winrate):
        if self.best_winrate < winrate:
            self.best_winrate = winrate
            self.trigger_times = 0
            if self.save_best_model:
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)
                torch.save(self.model.state_dict(),
                           os.path.join(self.output_dir, f'{self.model_comment}.bin.train'))
            return False
        if self.early_stop is None:
            return False
        self.trigger_times += 1
        if self.trigger_times >= self.early_stop:
            return True

    def rename_file(self):
        if self.save_best_model:
            path = os.path.join(self.output_dir, f'{self.model_comment}.bin')
            if os.path.exists(path):
                os.remove(path)
            os.rename(f'{path}.train', path)
