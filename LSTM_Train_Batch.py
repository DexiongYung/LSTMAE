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

from DataSetUtils.NameCateDL import NameCategoricalDataLoader
from DataSetUtils.NameDS import NameDataset
from Models.Decoder import Decoder
from Utilities.Convert import *
from Utilities.Noiser import noise_name
from Utilities.Train_Util import plot_losses, timeSince

# Optional command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help='Name of the Session', nargs='?', default='first', type=str)
parser.add_argument('--hidden_size', help='Size of the hidden layer of LSTM', nargs='?', default=256, type=int)
parser.add_argument('--lr', help='Learning rate', nargs='?', default=0.005, type=float)
parser.add_argument('--num_iter', help='Number of iterations', nargs='?', default=100000, type=int)
parser.add_argument('--num_layers', help='Number of layers', nargs='?', default=5, type=int)
parser.add_argument('--train_file', help='File to train on', nargs='?', default='Data/FirstNames.csv', type=str)
parser.add_argument('--column', help='Column header of data', nargs='?', default='name', type=str)
parser.add_argument('--print', help='Print every', nargs='?', default=100, type=int)
parser.add_argument('--batch', help='Batch size', nargs='?', default=5000, type=int)
parser.add_argument('--continue_training', help='Boolean whether to continue training an existing model', nargs='?',
                    default=False, type=bool)

# Parse optional args from command line and save the configurations into a JSON file
args = parser.parse_args()
NAME = args.name
ITER = args.num_iter
NUM_LAYERS = args.num_layers
LR = args.lr
HIDDEN_SZ = args.hidden_size
TRAIN_FILE = args.train_file
BATCH_SZ = args.batch
COLUMN = args.column
PRINTS = args.print
CLIP = 1

# Global variables
SOS = '0'
PAD = '1'
EOS = '2'
IN_CHARS = string.ascii_letters + "\'-" + EOS + SOS + PAD
IN_COUNT = len(IN_CHARS)
OUT_CHARS = string.ascii_letters + "\'-" + EOS + PAD
OUT_COUNT = len(OUT_CHARS)
MAX_LENGTH = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(x: str):
    optimizer.zero_grad()
    loss = 0.
    
    batch_sz = len(x)
    max_len = len(max(x, key=len)) + 1  # +1 for EOS xor SOS

    src_x = list(map(lambda s: SOS + s + (PAD * ((max_len - len(s)) - 1)), x))
    trg_x = list(map(lambda s: s + EOS + (PAD * ((max_len - len(s)) - 1)), x))

    trg = targetsTensor(trg_x, max_len, OUT_CHARS).to(DEVICE)
    src = indexTensor(src_x, max_len, IN_CHARS).to(DEVICE)
    lstm_hidden = lstm.initHidden(batch_sz)
    lstm_hidden = (lstm_hidden[0].to(DEVICE), lstm_hidden[1].to(DEVICE))

    names = [''] * batch_sz

    for i in range(src.shape[0]):
        lstm_input = src[i]
        lstm_probs, lstm_hidden = lstm(lstm_input, lstm_hidden)
        best_index = torch.argmax(lstm_probs, dim=2)

        loss += criterion(lstm_probs[0], trg[i])

        for idx in range(len(names)):
            names[idx] += OUT_CHARS[best_index[0][idx].item()]

    loss.backward()
    optimizer.step()
    
    for p in lstm.parameters():
        p.data.add_(-LR, p.grad.data)

    return names, loss.item()


def iter_train(dl: NameCategoricalDataLoader, iterations: int = ITER, path: str = "Checkpoints/",
               print_every: int = PRINTS):
    all_losses = []
    total_loss = 0

    for iter in range(1, iterations + 1):
        x = dl.sample()
        name, loss = train(x)
        total_loss += loss

        if iter % print_every == 0:
            all_losses.append(total_loss / print_every)
            total_loss = 0
            plot_losses(all_losses, x_label=f"Iteration of Batch Size: {BATCH_SZ}", y_label="NLLosss", filename=NAME)
            torch.save({'weights': lstm.state_dict()}, os.path.join(f"{path}{NAME}.path.tar"))

def iter_train_dl(dl: DataLoader, epochs: int = ITER, path: str = "Checkpoints/", print_every: int = PRINTS):
    all_losses = []
    total_loss = 0

    for iter in range(1, epochs + 1):
        for x in dl:
            name, loss = train(x)
            total_loss += loss

            if iter % print_every == 0:
                all_losses.append(total_loss / print_every)
                total_loss = 0
                plot_losses(all_losses, x_label=f"Iteration of Batch Size: {BATCH_SZ}", y_label="NLLosss", filename=NAME)
                torch.save({'weights': lstm.state_dict()}, os.path.join(f"{path}{NAME}.path.tar"))

def sample():
    with torch.no_grad():
        lstm_input = indexTensor([SOS], 1 ,IN_CHARS).to(DEVICE)
        lstm_hidden = lstm.initHidden(1)
        lstm_hidden = (lstm_hidden[0].to(DEVICE), lstm_hidden[1].to(DEVICE))
        name = ''
        char = SOS

        for i in range(MAX_LENGTH):
            lstm_probs, lstm_hidden = lstm(lstm_input, lstm_hidden)
            lstm_probs = torch.softmax(lstm_probs, dim=2)
            sample = torch.distributions.categorical.Categorical(lstm_probs).sample()
            sample = sample[0]
            char = OUT_CHARS[sample]

            if char is EOS:
                break

            name += char
            lstm_input = torch.zeros(1, 1, IN_COUNT).to(DEVICE)
            lstm_input[0, 0, sample] = 1.

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
    'input': IN_CHARS,
    'output': OUT_CHARS,
    'input_sz': IN_COUNT,
    'output_sz': OUT_COUNT,
    'EOS': EOS,
    'SOS': SOS,
    'PAD': PAD
}

save_json(f'Config/{NAME}.json', to_save)

df = pd.read_csv(TRAIN_FILE)
dl = NameCategoricalDataLoader(df, batch_sz=BATCH_SZ)

lstm = Decoder(IN_COUNT, HIDDEN_SZ, OUT_COUNT, padding_idx=IN_CHARS.find(PAD), num_layers=NUM_LAYERS)
lstm.to(DEVICE)
criterion = nn.NLLLoss(ignore_index=OUT_CHARS.find(PAD))
optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)

if args.continue_training is True:
     lstm.load_state_dict(torch.load(f'Checkpoints/{NAME}.path.tar')['weights'])

iter_train(dl)
