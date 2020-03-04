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
from Utilities.Convert import string_to_tensor, pad_string, int_to_tensor, char_to_index, targetTensor
from Utilities.Noiser import noise_name
from Utilities.Train_Util import plot_losses, timeSince

# Optional command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Name of the Session', nargs='?', default='first_lstm', type=str)
parser.add_argument('--hidden_size', help='Size of the hidden layer of LSTM', nargs='?', default=256, type=int)
parser.add_argument('--lr', help='Learning rate', nargs='?', default=0.005, type=float)
parser.add_argument('--num_epochs', help='Number of epochs', nargs='?', default=1000, type=int)
parser.add_argument('--num_layers', help='Number of layers', nargs='?', default=5, type=int)
parser.add_argument('--train_file', help='File to train on', nargs='?', default='Data/FirstNames.csv', type=str)
parser.add_argument('--column', help='Column header of data', nargs='?', default='name', type=str)
parser.add_argument('--print', help='Print every', nargs='?', default=500, type=int)
parser.add_argument('--continue_training', help='Boolean whether to continue training an existing model', nargs='?',
                    default=True, type=bool)

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

    loss = 0

    src = string_to_tensor(SOS + x, ALL_CHARS).to(DEVICE)
    trg = targetTensor(x + EOS, ALL_CHARS).to(DEVICE)
    lstm_input = init_lstm_input()
    lstm_hidden = lstm.initHidden()
    lstm_hidden = (lstm_hidden[0].to(DEVICE), lstm_hidden[1].to(DEVICE))
    
    name = ''

    for i in range(src.shape[0]):
        lstm_probs, lstm_hidden = lstm(lstm_input, lstm_hidden)
        loss += criterion(lstm_probs[0], trg[i].unsqueeze(0))
        best_index = torch.argmax(lstm_probs, dim=2).item()
        name += ALL_CHARS[best_index]
        lstm_input = src[i].unsqueeze(0)

    loss.backward()
    lstm_optim.step()
    return name, loss.item()


def iter_train(column: str, dl: DataLoader, epochs: int = EPOCH, path: str = "Checkpoints/", print_every: int = PRINTS):
    all_losses = []
    total_loss = 0
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
    with torch.no_grad():
        lstm_input = init_lstm_input()
        lstm_hidden = lstm.initHidden()
        lstm_hidden = (lstm_hidden[0].to(DEVICE), lstm_hidden[1].to(DEVICE))
        name = ''
        char = SOS
        iter = 0

        while char is not EOS and iter < MAX_LENGTH:
            iter += 1
            lstm_probs, lstm_hidden = lstm(lstm_input, lstm_hidden)
            best_index = torch.argmax(lstm_probs, dim=2).item()
            char = ALL_CHARS[best_index]
            name += char
            lstm_input = torch.zeros(1, 1, LETTERS_COUNT).to(DEVICE)
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
    'hidden_size': HIDDEN_SZ,
    'num_layers': NUM_LAYERS,
    'input_size/output': LETTERS_COUNT,
    'input/output': ALL_CHARS,
    'EOS_idx': EOS,
    'SOS_idx': SOS,
    'PAD_idx': PAD
}

save_json(f'Config/{NAME}.json', to_save)

df = pd.read_csv(TRAIN_FILE)
ds = NameDataset(df, COLUMN)
dl = DataLoader(ds, batch_size=1, shuffle=True)

lstm = Decoder(LETTERS_COUNT, HIDDEN_SZ, LETTERS_COUNT, num_layers=NUM_LAYERS)
criterion = nn.NLLLoss()

if args.continue_training:
    lstm.load_state_dict(torch.load(f'Checkpoints/{NAME}.path.tar')['weights'])

lstm_optim = torch.optim.Adam(lstm.parameters(), lr=LR)
lstm.to(DEVICE)

iter_train(COLUMN, dl)