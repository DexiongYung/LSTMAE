import math
from random import randint

from Convert import ALL_CHARS, MAX_LENGTH, LETTERS_COUNT

MAX = 2

def noise_name(x: str, allowed_chars: str):
    noise_type = randint(0, 3)

    if noise_type == 0:
        return add_chars(x, allowed_chars)
    elif noise_type == 1:
        return switch_chars(x, allowed_chars)
    elif 2:
        return remove_chars(x)
    else:
        x = remove_chars(x)
        return add_chars(x, allowed_chars)


def add_chars(x: str, allowed_chars: str, max_add: int = MAX):
    if max_add + len(x) > MAX_LENGTH:
        raise Exception(f"{max_add + len(x)} is greater than max length:{MAX_LENGTH}")

    ret = x
    num_to_add = randint(0, max_add)

    for i in range(num_to_add):
        random_char = allowed_chars[randint(0, len(allowed_chars) - 1)]
        pos = randint(0, len(ret) - 1)
        ret = "".join((ret[:pos], random_char, ret[pos:]))

    return ret


def switch_chars(x: str, allowed_chars: str, max_switch: int = MAX):
    ret = x
    num_to_switch = randint(0, min(math.floor(len(x) / 2), max_switch))

    for i in range(num_to_switch):
        random_char = allowed_chars[randint(0, len(allowed_chars) - 1)]
        pos = randint(0, len(ret) - 1)
        ret = "".join((ret[:pos], random_char, ret[pos + 1:]))

    return ret


def remove_chars(x: str, max_remove: int = MAX):
    ret = x
    num_to_remove = randint(0, min(math.floor(len(x) / 2), max_remove))

    for i in range(num_to_remove):
        pos = randint(0, len(ret) - 1)
        ret = "".join((ret[:pos], ret[pos + 1:]))

    return ret
