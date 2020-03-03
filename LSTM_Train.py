import argparse
import datetime
import json
import math
import os
import pandas as pd
import random
import string
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from io import open
from torch.utils.data import DataLoader

from DataSetUtils.NameDS import NameDataset
from Models.Decoder import Decoder
from Utilities.Convert import string_to_tensor, pad_string, int_to_tensor, char_to_index
from Utilities.Noiser import noise_name
from Utilities.Train_Util import plot_losses, timeSince

# Optional command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Name of the Session', nargs='?', default='first_lstm', type=str)
parser.add_argument('--hidden_size', help='Size of the hidden layer of LSTM', nargs='?', default=256, type=int)
parser.add_argument('--lr', help='Learning rate', nargs='?', default=0.005, type=float)
parser.add_argument('--batch_size', help='Size of the batch training on', nargs='?', default=500, type=int)
parser.add_argument('--num_epochs', help='Number of epochs', nargs='?', default=1000, type=int)
parser.add_argument('--num_layers', help='Number of layers', nargs='?', default=5, type=int)
parser.add_argument('--train_file', help='File to train on', nargs='?', default='Data/FirstNames.csv', type=str)
parser.add_argument('--column', help='Column header of data', nargs='?', default='name', type=str)
parser.add_argument('--print', help='Print every', nargs='?', default=500, type=int)
parser.add_argument('--continue_training', help='Boolean whether to continue training an existing model', nargs='?',
                    default=False, type=bool)

# Parse optional args from command line and save the configurations into a JSON file
args = parser.parse_args()
NAME = args.name
EPOCH = args.num_epochs
NUM_LAYERS = args.num_layers
LR = args.lr
HIDDEN_SZ = args.hidden_size
TRAIN_FILE = args.train_file
COLUMN = args.column
PRINTS = args.print
CLIP = 1

SOS = '0'
PAD = '1'
EOS = '2'
ALL_CHARS = string.ascii_lowercase + "\'." + EOS + SOS
LETTERS_COUNT = len(ALL_CHARS)
MAX_LENGTH = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_lstm_input():
    lstm_input = torch.zeros(1, 1, LETTERS_COUNT)
    lstm_input[0, 0, char_to_index(SOS, ALL_CHARS)] = 1.
    return lstm_input.to(DEVICE)


def train(x: str):
    lstm_optim.zero_grad()

    loss = 0.

    x = string_to_tensor(x + EOS, ALL_CHARS)
    lstm_input = init_lstm_input()
    lstm_hidden = lstm.initHidden()
    lstm_hidden = (lstm_hidden[0].to(DEVICE), lstm_hidden[1].to(DEVICE))
    name = ''

    for i in range(x.shape[0]):
        lstm_probs, lstm_hidden = lstm(lstm_input, lstm_hidden)
        _, nonzero_indexes = x[i].topk(1)
        best_index = torch.argmax(lstm_probs, dim=2).item()
        loss += criterion(lstm_probs[0], nonzero_indexes[0])
        name += ALL_CHARS[best_index]
        lstm_input = torch.zeros(1, 1, LETTERS_COUNT).to(DEVICE)
        lstm_input[0, 0, best_index] = 1.

    loss.backward()
    lstm_optim.step()
    return name, loss.item()


def iter_train(column: str, dl: DataLoader, epochs: int = EPOCH, path: str = "Checkpoints/", print_every: int = PRINTS):
    all_losses = []
    total_loss = 0  # Reset every plot_every iters
    start = time.time()
    iter = 0

    for e in range(epochs):
        for x in dl:
            iter += 1
            input = x[0]
            name, loss = train(input)
            total_loss += loss

            if iter % print_every == 0:
                all_losses.append(total_loss / print_every)
                total_loss = 0
                plot_losses(all_losses, filename=NAME)
                torch.save({'weights': lstm.state_dict()}, os.path.join(f"{path}{NAME}.path.tar"))


def sample():
    lstm.no_grad()

    lstm_input = init_lstm_input()
    lstm_hidden = lstm.initHidden()
    name = ''
    char = SOS
    iter = 0

    while char is not EOS and iter < MAX_LENGTH:
        iter += 1
        lstm_probs, lstm_hidden = lstm(lstm_input, lstm_hidden)
        best_index = torch.argmax(lstm_probs, dim=2).item()
        char = ALL_CHARS[best_index]
        name += char
        lstm_input = torch.zeros(1, 1, LETTERS_COUNT)
        lstm_input[0, 0, best_index] = 1.

    return name


def load_json(jsonpath: str) -> dict:
    with open(jsonpath) as jsonfile:
        return json.load(jsonfile, object_pairs_hook=OrderedDict)


def save_json(jsonpath: str, content):
    with open(jsonpath, 'w') as jsonfile:
        json.dump(content, jsonfile)


to_save = {
    'session_name': NAME,
    'hidden_size': args.hidden_size,
    'batch_size': args.batch_size,
    'num_layers': NUM_LAYERS,
    'input_size': len(ALL_CHARS),
    'output_size': len(ALL_CHARS),
    'input': ALL_CHARS,
    'output': ALL_CHARS,
}

save_json(f'Config/{NAME}.json', to_save)

train_df = pd.read_csv(TRAIN_FILE)
train_ds = NameDataset(train_df, "name")
dl = DataLoader(train_ds, batch_size=1, shuffle=True)

lstm = Decoder(LETTERS_COUNT, HIDDEN_SZ, LETTERS_COUNT, num_layers=NUM_LAYERS)
criterion = nn.NLLLoss()

current_DT = datetime.datetime.now()
date_time = current_DT.strftime("%Y-%m-%d")

lstm_optim = torch.optim.Adam(lstm.parameters(), lr=LR)
lstm.to(DEVICE)

iter_train("name", dl)
