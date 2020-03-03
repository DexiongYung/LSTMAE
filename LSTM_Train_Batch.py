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
from Utilities.Convert import string_to_tensor, pad_string, int_to_tensor, char_to_index, strings_to_tensor
from Utilities.Noiser import noise_name
from Utilities.Train_Util import plot_losses, timeSince

# Optional command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Name of the Session', nargs='?', default='batch_first_lstm', type=str)
parser.add_argument('--hidden_size', help='Size of the hidden layer of LSTM', nargs='?', default=256, type=int)
parser.add_argument('--lr', help='Learning rate', nargs='?', default=0.005, type=float)
parser.add_argument('--num_epochs', help='Number of epochs', nargs='?', default=1000, type=int)
parser.add_argument('--num_layers', help='Number of layers', nargs='?', default=5, type=int)
parser.add_argument('--train_file', help='File to train on', nargs='?', default='Data/FirstNames.csv', type=str)
parser.add_argument('--column', help='Column header of data', nargs='?', default='name', type=str)
parser.add_argument('--print', help='Print every', nargs='?', default=5000, type=int)
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
ALL_CHARS = string.ascii_lowercase + "\'." + EOS + SOS + PAD
LETTERS_COUNT = len(ALL_CHARS)
MAX_LENGTH = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_lstm_input(batch_sz: int):
    lstm_input = torch.zeros(1, batch_sz, LETTERS_COUNT)
    
    for idx in range(batch_sz):
        lstm_input[0, idx, char_to_index(SOS, ALL_CHARS)] = 1.
    
    return lstm_input.to(DEVICE)


def train(x: str):
    batch_sz = len(x)
    max_len = len(max(x, key=len)) + 2 # +2 for EOS and SOS

    x = list(map(lambda s: (PAD * ((max_len - len(s)) - 2)) + SOS + s + EOS, x))

    lstm_optim.zero_grad()
    loss = 0.

    x = strings_to_tensor(x, max_len, ALL_CHARS).to(DEVICE)
    lstm_input = x[0]
    lstm_hidden = lstm.initHidden(batch_sz)
    lstm_hidden = (lstm_hidden[0].to(DEVICE), lstm_hidden[1].to(DEVICE))
    names = [''] * batch_sz

    for i in range(x.shape[0]):
        lstm_probs, lstm_hidden = lstm(lstm_input.unsqueeze(0), lstm_hidden)
        _, nonzero_indexes = x[i].topk(1)
        best_index = torch.argmax(lstm_probs, dim=2)
        # If not learning properly probably because of these transposes being the wrong dimension
        loss += criterion(lstm_probs.transpose(1,2), nonzero_indexes.transpose(0,1).to(DEVICE))
        letters = [''] * batch_sz

        for i in range(len(letters)):
            letters[i] = ALL_CHARS[best_index[0][i].item()]

        lstm_input = strings_to_tensor(letters, 1, ALL_CHARS)[0].to(DEVICE)

    loss.backward()
    lstm_optim.step()
    return names, loss.item()


def iter_train(column: str, dl: DataLoader, epochs: int = EPOCH, path: str = "Checkpoints/", print_every: int = PRINTS):
    all_losses = []
    total_loss = 0
    iter = 0

    for e in range(epochs):
        for x in dl:
            iter += 1
            name, loss = train(x)
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
dl = DataLoader(ds, batch_size=256, shuffle=True)

lstm = Decoder(LETTERS_COUNT, HIDDEN_SZ, LETTERS_COUNT, num_layers=NUM_LAYERS)
criterion = nn.NLLLoss(ignore_index= ALL_CHARS.find(PAD))

if args.continue_training:
    lstm.load_state_dict(torch.load(f'Checkpoints/{NAME}.path.tar')['weights'])

lstm_optim = torch.optim.Adam(lstm.parameters(), lr=LR)
lstm.to(DEVICE)

iter_train(COLUMN, dl)
