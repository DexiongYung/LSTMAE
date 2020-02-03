import datetime
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import random
import time
import torch
import torch.nn as nn
from io import open

from Convert import string_to_tensor, pad_string, int_to_tensor, MAX_LENGTH, ALL_CHARS, LETTERS_COUNT, char_to_index, \
    SOS, EOS, PAD
from Decoder import Decoder
from Encoder import Encoder
from Noiser import noise_name

PRINTS = 10000


def init_decoder_input():
    decoder_input = torch.zeros(1, 1, LETTERS_COUNT)
    decoder_input[0, 0, char_to_index(SOS)] = 1.
    return decoder_input


def plot_losses(loss: list, folder: str = "Result", filename: str = None):
    x = list(range(len(loss)))
    plt.plot(x, loss, 'r--', label="Loss")
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper left')
    plt.savefig(f"{folder}/{date_time}")
    plt.close()


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def denoise_train(x: str):
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    loss = 0.

    noisy_x = noise_name(x)
    x = string_to_tensor(x + EOS)
    noised_x = string_to_tensor(noisy_x + EOS)

    encoder_hidden = encoder.init_hidden()

    for i in range(noised_x.shape[0]):
        # LSTM requires 3 dimensional inputs
        _, encoder_hidden = encoder(noised_x[i].unsqueeze(0), encoder_hidden)

    decoder_input = init_decoder_input()
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
    return name, noisy_x, loss.item()


def test(x: str):
    noisy_x = noise_name(x)
    noised_x = string_to_tensor(noisy_x + SOS)
    x = string_to_tensor(x + EOS)

    encoder_hidden = encoder.init_hidden()
    for i in range(noised_x.shape[0]):
        # LSTM requires 3 dimensional inputs
        _, encoder_hidden = encoder(noised_x[i].unsqueeze(0), encoder_hidden)

    decoder_input = init_decoder_input()
    decoder_hidden = encoder_hidden
    output_char = SOS
    name = ''

    while output_char is not EOS and len(name) <= MAX_LENGTH:
        decoder_probs, decoder_hidden = decoder(decoder_input, decoder_hidden)
        best_index = torch.argmax(decoder_probs, dim=2).item()
        output_char = ALL_CHARS[best_index]
        name += output_char
        decoder_input = torch.zeros(1, 1, LETTERS_COUNT)
        decoder_input[0, 0, best_index] = 1.

    return name, noisy_x


def test_no_noise(x: str):
    x = string_to_tensor(x + EOS)

    encoder_hidden = encoder.init_hidden()
    for i in range(x.shape[0]):
        # LSTM requires 3 dimensional inputs
        _, encoder_hidden = encoder(x[i].unsqueeze(0), encoder_hidden)

    decoder_input = init_decoder_input()
    decoder_hidden = encoder_hidden
    output_char = SOS
    name = ''

    while output_char is not EOS and len(name) <= MAX_LENGTH:
        decoder_probs, decoder_hidden = decoder(decoder_input, decoder_hidden)
        best_index = torch.argmax(decoder_probs, dim=2).item()
        output_char = ALL_CHARS[best_index]
        name += output_char
        decoder_input = torch.zeros(1, 1, LETTERS_COUNT)
        decoder_input[0, 0, best_index] = 1.

    return name


def iter_test(column: str, df: pd.DataFrame, print_every: int = PRINTS):
    start = time.time()
    n_iters = len(df)
    total = 0
    correct = 0
    for iter in range(n_iters):
        input = df.iloc[iter][column]
        name, noised_x = test(input)

        total += 1

        name = name.replace(EOS, '')

        if input == name:
            correct += 1

        if iter % print_every == 0:
            print(f"Total: {total}, Correct: {correct}, Input: {noised_x}, Name:{name}, Original:{input}")

    print(f"Total: {total}, Correct: {correct}")
    return total, correct


def iter_test_no_noise(column: str, df: pd.DataFrame, print_every: int = PRINTS):
    start = time.time()
    n_iters = len(df)
    total = 0
    correct = 0
    for iter in range(n_iters):
        input = df.iloc[iter][column]
        name = test_no_noise(input)

        total += 1

        name = name.replace(EOS, '')

        if input == name:
            correct += 1

        if iter % print_every == 0:
            print(f"Total: {total}, Correct: {correct}, Input: {input}, Output:{name}")

    print(f"Total: {total}, Correct: {correct}")
    return total, correct


def iter_train(column: str, df: pd.DataFrame, epochs: int = 2000, path: str = "Checkpoints/", print_every: int = PRINTS,
               plot_every: int = PRINTS):
    all_losses = []
    total_loss = 0  # Reset every plot_every iters
    start = time.time()
    n_iters = len(df)

    for e in range(epochs):
        for iter in range(n_iters):
            input = df.iloc[iter][column]
            name, noisy_name, loss = denoise_train(input)
            total_loss += loss

            if iter % print_every == 0:
                print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))
                print('input: %s, output: %s, original: %s' % (noisy_name, name, input))

            if iter % plot_every == 0:
                all_losses.append(total_loss / plot_every)
                total_loss = 0
                plot_losses(all_losses)
                torch.save({'weights': encoder.state_dict()}, os.path.join(f"{path}encoder_{date_time}.path.tar"))
                torch.save({'weights': decoder.state_dict()}, os.path.join(f"{path}decoder_{date_time}.path.tar"))


test_df = pd.read_csv("Data/Train.csv")
train_df = pd.read_csv("Data/Test.csv")

hidden_layer_sz = 256
encoder = Encoder(LETTERS_COUNT, hidden_layer_sz)
decoder = Decoder(LETTERS_COUNT, hidden_layer_sz, LETTERS_COUNT)
criterion = nn.NLLLoss()

current_DT = datetime.datetime.now()
date_time = current_DT.strftime("%Y-%m-%d_%Hhr")

learning_rate = 0.0005
encoder_optim = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optim = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

iter_train("name", train_df)
iter_test("name", test_df)
