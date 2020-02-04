import datetime
import argparse
import math
import os
import pandas as pd
import random
import torch
import torch.nn as nn
from io import open
import string
import time

from Utilities.Convert import string_to_tensor, pad_string, int_to_tensor, char_to_index
from Utilities.Train_Util import plot_losses, timeSince
from Utilities.Noiser import noise_name

from Models.Decoder import Decoder
from Models.Encoder import Encoder


parser = argparse.ArgumentParser()
parser.add_argument('--sess_nm', help='Session name', nargs='?', default="No_Name", type=str)
parser.add_argument('--hidden_sz', help='Size of the hidden layer of LSTM', nargs='?', default=256, type=int)
parser.add_argument('--lr', help='Learning rate', nargs='?', default=0.0005, type=float)
parser.add_argument('--batch_sz', help='Size of the batch training on', nargs='?', default=2048, type=int)
parser.add_argument('--epochs', help='Number of epochs', nargs='?', default=1000, type=int)
parser.add_argument('--prints', help='Number of iterations to count', nargs="?", default=5000, type=int)
parser.add_argument('--train_csv', help="Path of the train csv file", nargs="?", default="Data/LN_Train.csv", type=str)
parser.add_argument('--test_csv', help="Path of the test csv file", nargs="?", default="Data/Test.csv", type=str)

args = parser.parse_args()
PRINTS = args.prints
LR = args.lr
HIDD_LAYER_SZ = args.hidden_sz
TRAIN_PATH = args.train_csv
TEST_PATH = args.test_csv

SOS = '0'
PAD = '1'
EOS = '2'
ALL_CHARS = string.ascii_letters + "\'."
LETTERS_COUNT = len(ALL_CHARS)
MAX_LENGTH = 20


def init_decoder_input():
    decoder_input = torch.zeros(1, 1, LETTERS_COUNT)
    decoder_input[0, 0, char_to_index(SOS, ALL_CHARS)] = 1.
    return decoder_input

def denoise_train(x: str):
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    loss = 0.

    noisy_x = noise_name(x, ALL_CHARS, MAX_LENGTH)
    x = string_to_tensor(x + EOS, ALL_CHARS)
    noised_x = string_to_tensor(noisy_x + EOS, ALL_CHARS)

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
    noisy_x = noise_name(x, ALL_CHARS, MAX_LENGTH)
    noised_x = string_to_tensor(noisy_x + SOS, ALL_CHARS)
    x = string_to_tensor(x + EOS, ALL_CHARS)

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
    x = string_to_tensor(x + EOS, ALL_CHARS, ALL_CHARS)

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
                plot_losses(all_losses, date_time)
                torch.save({'weights': encoder.state_dict()}, os.path.join(f"{path}encoder_{date_time}.path.tar"))
                torch.save({'weights': decoder.state_dict()}, os.path.join(f"{path}decoder_{date_time}.path.tar"))


train_df = pd.read_csv(TRAIN_PATH)

encoder = Encoder(LETTERS_COUNT, HIDD_LAYER_SZ)
decoder = Decoder(LETTERS_COUNT, HIDD_LAYER_SZ, LETTERS_COUNT)
criterion = nn.NLLLoss()

current_DT = datetime.datetime.now()
date_time = current_DT.strftime("%Y-%m-%d")

encoder_optim = torch.optim.Adam(encoder.parameters(), lr=LR)
decoder_optim = torch.optim.Adam(decoder.parameters(), lr=LR)

iter_train("name", train_df)