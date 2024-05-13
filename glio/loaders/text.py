import random, torch

def get_char_dict(text:str, mode='random'):
    unique = list(set(text))
    if mode == 'random': random.shuffle(unique)
    elif mode == 'sorted': unique = sorted(unique)
    elif mode == 'common': unique = sorted(unique, key=text.count, reverse=True)
    return {char: i for i, char in enumerate(unique)}


def str_to_tensor(text:str, vmin=0, vmax=1, mode='random'):
    """Converts a string to a 1d tesor, all unique characters are spread in `min` to `max` range according to `mode`"""
    chardict = get_char_dict(text, mode)
    tensor = torch.tensor([chardict[char] for char in text])
    return (tensor / (len(chardict)-1)) * (vmax-vmin) + vmin

from ..python_tools import reduce_dim
def str_to_2d_tensor(text:str, vmin = 0, vmax = 1, mode = 'random', term = ['.', '\n'], remove_empty = True, padding = ' '):
    """Converts string to 2d tensor where next line begins after a string from `term` occurs. To achieve a constant size, all strings are padded with `padding` to the same length"""
    if isinstance(term, str): term = [term]

    chardict = get_char_dict(text, mode)

    text_list = [text]
    for i in term:
        text_list = reduce_dim([t.split(i) for t in text_list])

    if remove_empty: text_list = [t for t in text_list if len(t) > 0]

    lengths = [len(t) for t in text_list]
    max_length = max(lengths)

    for i in range(len(lengths)):
        text_list[i] = text_list[i] + padding*(max_length - lengths[i])

    tensor = torch.stack([torch.tensor([chardict[char] for char in t]) for t in text_list])
    return (tensor / (len(chardict)-1)) * (vmax-vmin) + vmin



