import datetime
from string import Template

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def check_options_is_correct(self):
        assert self.options.save_best_model or self.options.save_every, \
            'Use --checkpoint_frequency and/or --save_best_model to save the training result'
        assert self.options.niter >= 1, 'niter must be >= 1'

    def modify_options(self):
        opt_dict = self.options.__dict__.copy()
        opt_dict['lr'] = format(opt_dict['lr'], '.1e')
        opt_dict['time'] = datetime.datetime.now().strftime('%H.%M')
        opt_dict['date'] = datetime.datetime.now().strftime('%d-%m-%y')
        new_model_comment = Template(self.options.model_comment.replace('%', '$')).safe_substitute(**opt_dict)
        opt_dict['model_comment'] = new_model_comment
        self.options.model_comment = new_model_comment
        if self.options.tb_dir is not None:
            if self.options.tb_comment is None:
                self.options.tb_comment = '%model_comment'
            new_tb_comment = Template(self.options.tb_comment.replace('%', '$')).safe_substitute(**opt_dict)
            self.options.tb_comment = new_tb_comment

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument(
            '--output_dir',
            type=str,
            required=True,
            help='Directory for saving the model and its checkpoints',
        )
        parser.add_argument('--niter', type=int, default=5000,
                            help='Maximum number of iters')

        parser.add_argument('--checkpoint_frequency', type=int, default=None,
                            help='Save checkpoint every k iters')

        parser.add_argument('--log_frequency', type=int, default=100,
                            help='Log every k iters')
        parser.add_argument('--early_stop', type=int, default=None,
                            help='Stop training after n logs without improvement')
        parser.add_argument('--games_for_metrics', type=int, default=1000,
                            help='Save checkpoint every k iters')
        parser.add_argument('--model_comment', type=str, required=True, help='Model comment')
        parser.add_argument('--tb_dir', type=str, default=None,
                            help='TensorBoard dir path')
        parser.add_argument('--tb_comment', type=str, default=None,
                            help='TensorBoard comment.')

        parser.add_argument('--lr_scheduler_frequency', type=int, default=100,
                            help='Log every k iters')
        parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
        parser.add_argument('--gamma', default=1.0, type=float, help='Gamma for exponential scheduler')

        parser.add_argument('--verbose', type=int, default=0, help='Controls the verbosity')
        parser.add_argument('--save_best_model', action='store_true',
                            help='Save best model')

        return parser
