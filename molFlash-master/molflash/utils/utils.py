# Consists basic functions that can be used throughout
from typing import List


def split_data(len_data: int, splits: List = [0.8, 0.1, 0.1]):
    r"""
    Return the length of train, test, and val data
    Args:
        len_data: length of the data
        splits: a list of sizes of splits
    """
    train_len = int(splits[0] * len_data)
    test_len = int(splits[1] * len_data)
    val_len = len_data - train_len - test_len
    return train_len, test_len, val_len
