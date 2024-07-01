import argparse


class BaseOptions:

    def __init__(self):
        self.options = self.initialize(argparse.ArgumentParser()).parse_args()
        self.check_options_is_correct()
        self.modify_options()

    def initialize(self, parser):
        parser.add_argument('--row_count', default=None, type=int, required=True,
                            help='Row count for Minesweeper field')
        parser.add_argument('--column_count', default=None, type=int, required=True,
                            help='Column count for Minesweeper field')
        parser.add_argument('--mine_count', default=None, type=int, required=True,
                            help='Mine count for Minesweeper')
        parser.add_argument('--checkpoint_path', default=None, type=str,
                            help='Checkpoint path')
        parser.add_argument('--batch_size', default=None, type=int, required=True, help='Batch size')

        parser.add_argument('--block_count', default=None, type=int, required=True,
                            help=' Number of residual blocks in the model.')
        parser.add_argument('--hidden_channels', default=None, type=int, required=True,
                            help='Number of channels in the hidden convolutional layers.')

        parser.add_argument('--seed', type=int, default=None,
                            help='Enables repeatable experiments by setting the seed for the random')
        parser.add_argument('--cuda', action='store_true', help='Use cuda if available')

        return parser

    def check_options_is_correct(self):
        pass

    def modify_options(self):
        pass

    def get_options(self):
        return self.options

    def print_options(self):
        print('       Option               Value        ')
        for k, v in self.options.__dict__.items():
            print(f'{k:<26}{str(v):<21}')
