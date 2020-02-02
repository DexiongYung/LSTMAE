import datetime
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import string
import time
import torch
import torch.nn as nn
from io import open

from Decoder import Decoder
from Encoder import Encoder

SOS = '1'
EOS = '2'
PAD = '3'
ALL_CHARS = string.ascii_letters + "'-" + SOS + EOS + PAD
LETTERS_COUNT = len(ALL_CHARS)


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


def randomName(data, column: str):
    return data.iloc[random.randint(0, len(data) - 1)][column]


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def train(x):
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    loss = 0.
    x = string_to_tensor(x)
    encoder_hidden = encoder.init_hidden()
    for i in range(x.shape[0]):
        # LSTM requires 3 dimensional inputs
        _, encoder_hidden = encoder(x[i].unsqueeze(0), encoder_hidden)

    decoder_input = torch.zeros(1, 1, LETTERS_COUNT)
    decoder_input[0, 0, -1] = 1.
    decoder_hidden = encoder_hidden
    name = ''

    for i in range(x.shape[0]):
        decoder_probs, decoder_hidden = decoder(decoder_input, decoder_hidden)
        _, nonzero_indexes = x[i].topk(1)
        best_index = torch.argmax(decoder_probs, dim=2).item()
        loss += criterion(decoder_probs[0], nonzero_indexes[0])
        name += ALL_CHARS[best_index]
        decoder_input = torch.zeros(1, 1, LETTERS_COUNT)
        decoder_input[0, 0, best_index] = 1.

    loss.backward()
    encoder_optim.step()
    decoder_optim.step()
    return name, decoder_probs, loss.item()


def run_iter(n_iters: int, column: str, path: str = "Checkpoints/", print_every: int = 5000, plot_every: int = 500):
    all_losses = []
    total_loss = 0  # Reset every plot_every iters
    start = time.time()
    for iter in range(1, n_iters + 1):
        input = randomName(df, column)
        name, output, loss = train(input)
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))
            print('input: %s, output: %s' % (input, name))

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    current_DT = datetime.datetime.now()
    date_time = current_DT.strftime("%Y-%m-%d_%Hhr%Mm")
    torch.save({'weights': decoder.state_dict()}, os.path.join(f"{path}{date_time}"))


def iter_entire_data(column: str, df: pd.DataFrame, path: str = "Checkpoints/", print_every: int = 5000,
                     plot_every: int = 500):
    all_losses = []
    total_loss = 0  # Reset every plot_every iters
    start = time.time()
    n_iters = len(df)
    for iter in range(n_iters):
        input = df.iloc[iter][column]
        name, output, loss = train(input)
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))
            print('input: %s, output: %s' % (input, name))

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    current_DT = datetime.datetime.now()
    date_time = current_DT.strftime("%Y-%m-%d_%Hhr%Mm")
    torch.save({'weights': decoder.state_dict()}, os.path.join(f"{path}{date_time}"))


df = pd.read_csv("Data/Train.csv")
hidden_layer_sz = 256
encoder = Encoder(LETTERS_COUNT, hidden_layer_sz, 1)
decoder = Decoder(LETTERS_COUNT, hidden_layer_sz, LETTERS_COUNT)
criterion = nn.NLLLoss()

learning_rate = 0.0005

encoder_optim = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optim = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

iter_entire_data("name", df)
