from unittest import TestCase
import numpy as np
from utils import preprocess, position_to_idx, IN_CHANNELS


class TestUtils(TestCase):
    def test_preprocess(self):
        fields = np.arange(1, 10).reshape(1, 3, 3)
        expected_output = np.zeros((1, IN_CHANNELS, 3, 3), dtype=np.int32)
        expected_output[0, 1, 0, 0] = 1.0
        expected_output[0, 2, 0, 1] = 1.0
        expected_output[0, 3, 0, 2] = 1.0
        expected_output[0, 4, 1, 0] = 1.0
        expected_output[0, 5, 1, 1] = 1.0
        expected_output[0, 6, 1, 2] = 1.0
        expected_output[0, 7, 2, 0] = 1.0
        expected_output[0, 8, 2, 1] = 1.0
        expected_output[0, 9, 2, 2] = 1.0
        actual_output = preprocess(fields)
        self.assertTrue(np.all(expected_output == actual_output))

        fields = np.stack([np.zeros((4, 4), dtype=np.int32), np.ones((4, 4), dtype=np.int32)])
        output = preprocess(fields)
        self.assertTrue(np.sum(output[0, 0]) == 4 * 4)
        self.assertTrue(np.sum(output[0, 1:]) == 0)
        self.assertTrue(np.sum(output[1, 1]) == 4 * 4)
        self.assertTrue(np.sum(output[1, [0, 2, 3, 4, 5, 6, 7, 8, 9]]) == 0)

    def test_position_to_idx(self):
        i = 1
        j = 2
        k = 3
        row_count = 4
        column_count = 5
        expected_output = 33
        actual_output = position_to_idx(i, j, k, row_count, column_count)
        self.assertEqual(actual_output, expected_output)

        i = 0
        j = 0
        k = 0
        row_count = 10
        column_count = 10
        expected_output = 0
        actual_output = position_to_idx(i, j, k, row_count, column_count)
        self.assertEqual(actual_output, expected_output)

        i = 2
        j = 9
        k = 9
        row_count = 10
        column_count = 10
        expected_output = 299
        actual_output = position_to_idx(i, j, k, row_count, column_count)
        self.assertEqual(actual_output, expected_output)
