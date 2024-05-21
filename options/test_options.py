from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument(
            '--game_count', type=int, required=True, help='Game count for calc metrics.'
        )
        return parser
