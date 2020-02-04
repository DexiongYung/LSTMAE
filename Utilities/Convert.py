import torch


def char_to_index(char: str, allowed_chars: str) -> int:
    return allowed_chars.find(char)


def string_to_tensor(string: str, allowed_chars: str) -> list:
    tensor = torch.zeros(len(string), 1, len(allowed_chars))
    for i, char in enumerate(string):
        tensor[i, 0, char_to_index(char, allowed_chars)] = 1
    return tensor


def int_to_tensor(index: int, allowed_chars: str) -> list:
    tensor = torch.zeros([1, len(allowed_chars)], dtype=torch.long)
    tensor[:, index] = 1
    return tensor


def pad_string(original: str, desired_len: int, pad_character: str, pre_pad: bool = True):
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
