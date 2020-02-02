import string
import torch

SOS = '1'
EOS = '2'
PAD = '3'
ALL_CHARS = string.ascii_letters + "'-" + SOS + EOS + PAD
LETTERS_COUNT = len(ALL_CHARS)
MAX_LENGTH = 15


def char_to_index(char: str) -> int:
    return ALL_CHARS.find(char)


def string_to_tensor(string: str) -> list:
    tensor = torch.zeros(len(string), 1, LETTERS_COUNT)
    for i, char in enumerate(string):
        tensor[i, 0, char_to_index(char)] = 1
    return tensor


def int_to_tensor(index: int) -> list:
    tensor = torch.zeros([1, LETTERS_COUNT], dtype=torch.long)
    tensor[:, index] = 1
    return tensor


def pad_string(original: str, desired_len: int = MAX_LENGTH, pad_character: str = PAD, pre_pad: bool = True):
    """
    Returns the padded version of the original string to length: desired_len
    original: The string to be padded
    desired_len: The length of the new string
    pad_character: The character used to pad the original
    """
    if pre_pad:
        return (str(pad_character) * (desired_len - len(original))) + original
    else:
        return original + (str(pad_character) * (desired_len - len(original)))
