import random
import torch
import torch.nn as nn
from io import open
import string
import matplotlib.pyplot as plt
import time
import math
import os
import pandas as pd

SOS = '1'
EOS = '2'
PAD = '3'
DROP_OUT = '4'
CHARS = string.ascii_lowercase + "'-" + SOS + EOS + PAD + DROP_OUT
LETTERS_COUNT = len(CHARS)

df = pd.read_csv('cleaned.csv')

class Encoder(nn.Module):
    """
    Takes in an one-hot tensor of names and produces hidden state and cell state
    for decoder LSTM to use.

    input_size: N_LETTER
    hidden_size: Size of the hidden dimension
    """
    def __init__(self, input_size, hidden_size, num_layers=2):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Initialize LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, input, hidden):
        """
        Run LSTM through 1 time step.

        SHAPE REQUIREMENT
        - input: <1 x batch_size x N_LETTER>
        - hidden: (<num_layer x batch_size x hidden_size>, <num_layer x batch_size x hidden_size>)
        """
        #input = input.view(1, self.batch_size, -1)
        lstm_out, hidden = self.lstm(input, hidden)
        return lstm_out, hidden
    
    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.num_layers,batch_size,self.hidden_size),
                torch.zeros(self.num_layers,batch_size,self.hidden_size))

class Decoder(nn.Module):
    """
    Accept hidden layers as an argument <num_layer x batch_size x hidden_size> for each hidden and cell state.
    At every forward call, output probability vector of <batch_size x output_size>.
    input_size: N_LETTER
    hidden_size: Size of the hidden dimension
    output_size: N_LETTER
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # Initialize LSTM - Notice it does not accept output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input, hidden):
        """
        Run LSTM through 1 time step
        SHAPE REQUIREMENT
        - input: <1 x batch_size x N_LETTER>
        - hidden: (<num_layer x batch_size x hidden_size>, <num_layer x batch_size x hidden_size>)
        - lstm_out: <1 x batch_size x N_LETTER>
        """
        lstm_out, hidden = self.lstm(input, hidden)
        lstm_out = self.fc1(lstm_out)
        lstm_out = self.softmax(lstm_out)
        return lstm_out, hidden

    def initHidden(self, batch_size=1):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

def char_to_index(char: str) -> int:
    return CHARS.find(char)

def string_to_tensor(string: str) -> list:
    tensor = torch.zeros(len(string),1,LETTERS_COUNT)
    for i,char in enumerate(string):
        tensor[i,0,char_to_index(char)] = 1
    return tensor

def int_to_tensor(index: int) -> list:
    tensor = torch.zeros([1, LETTERS_COUNT],dtype=torch.long)
    tensor[:,index] = 1
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
    enc_optim.zero_grad()
    dec_optim.zero_grad()

    loss = 0.
    x = x.lower()
    x = string_to_tensor(x)
    enc_hidden = enc.init_hidden()
    for i in range(x.shape[0]):
        # RNN requires 3 dimensional inputs
        _, enc_hidden = enc(x[i].unsqueeze(0), enc_hidden)

    dec_input = torch.zeros(1, 1, LETTERS_COUNT)
    dec_input[0, 0, -1] = 1.
    dec_hidden = enc_hidden
    name = ''
    
    for i in range(x.shape[0]):
        dec_probs, dec_hidden = dec(dec_input, dec_hidden)
        _, nonzero_indexes = x[i].topk(1)
        best_index = torch.argmax(dec_probs, dim=2).item()
        loss += criterion(dec_probs[0], nonzero_indexes[0])
        name += CHARS[best_index]
        dec_input = torch.zeros(1, 1, LETTERS_COUNT)
        dec_input[0, 0, best_index] = 1.

    loss.backward()
    enc_optim.step()
    dec_optim.step()
    return name, dec_probs, loss.item()

def run_iter(n_iters, column: str, chckpt):
    print_every = 5000
    plot_every = 500
    all_losses = []
    total_loss = 0 # Reset every plot_every iters
    start = time.time()
    for iter in range(1, n_iters + 1):
        input = randomName(df, column)
        if not isinstance(input,str):
            continue
        name, output, loss = train(input)
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))
            print('input: %s, output: %s' % (input, name))

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

    torch.save({'weights':dec.state_dict()}, os.path.join(f"{chckpt}_checkpt.pth.tar"))

hidden_layer_sz = 64
enc = Encoder(LETTERS_COUNT, hidden_layer_sz, 1)
dec = Decoder(LETTERS_COUNT, hidden_layer_sz, LETTERS_COUNT)
criterion = nn.NLLLoss()

learning_rate = 0.0005

enc_optim = torch.optim.Adam(enc.parameters(),lr=0.001)
dec_optim = torch.optim.Adam(dec.parameters(),lr=0.001)

run_iter(10000000, "first", "first")
run_iter(10000000, "last", "last")
run_iter(10000000, "middle", "middle")
