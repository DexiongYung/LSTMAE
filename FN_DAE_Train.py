import argparse
import datetime
import os
import pandas as pd
import string
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from DataSetUtils.NameDS import NameDataset
from Models.Decoder import Decoder
from Models.Encoder import Encoder
from Utilities.Convert import string_to_tensor, pad_string, char_to_index, strings_to_index_tensor, \
    to_rnn_tensor, strings_to_tensor, index_to_char
from Utilities.Noiser import noise_name
from Utilities.Train_Util import plot_losses

parser = argparse.ArgumentParser()
parser.add_argument('--sess_nm', help='Session name', nargs='?', default="No_Name", type=str)
parser.add_argument('--hidden_sz', help='Size of the hidden layer of LSTM', nargs='?', default=256, type=int)
parser.add_argument('--lr', help='Learning rate', nargs='?', default=0.0005, type=float)
parser.add_argument('--batch_sz', help='Size of the batch training on', nargs='?', default=5000, type=int)
parser.add_argument('--epochs', help='Number of epochs', nargs='?', default=2000, type=int)
parser.add_argument('--prints', help='Number of batch iterations before print and plot/weight save', nargs="?",
                    default=5, type=int)
parser.add_argument('--noise_num', help='Number characters to noise', nargs="?", default=2, type=int)
parser.add_argument('--train_csv', help="Path of the train csv file", nargs="?", default="Data/FN_Train.csv", type=str)
parser.add_argument('--test_csv', help="Path of the test csv file", nargs="?", default="Data/FN_Test.csv", type=str)
parser.add_argument('--chck_pt_name', help="Name of checkpoint file", nargs="?", default="", type=str)

args = parser.parse_args()
PRINTS = args.prints
LR = args.lr
HIDD_SZ = args.hidden_sz
TRAIN_PTH = args.train_csv
TEST_PTH = args.test_csv
BATCH_SZ = args.batch_sz
NOISE_CNT = args.noise_num
EPOCH = args.epochs
CHCK_PT_NAME = args.chck_pt_name

PAD = '1'
SOS = '0'
ENCODER_CHARS = string.printable
DECODER_CHARS = string.ascii_letters + "\'.,- " + PAD + SOS
ENC_CHAR_CNT = len(ENCODER_CHARS)
DEC_CHAR_CNT = len(DECODER_CHARS)
MAX_LEN = 50


def denoise_train(x: DataLoader):
    encoder_optim.zero_grad()
    decoder_optim.zero_grad()

    loss = 0

    padded_x = list(map(lambda s: pad_string(s, MAX_LEN, PAD), x))
    x_idx_tensor = strings_to_index_tensor(padded_x, MAX_LEN, DECODER_CHARS, char_to_index)

    noisy_x = list(map(lambda s: noise_name(s, ENCODER_CHARS, MAX_LEN, NOISE_CNT), x))
    padded_noisy_x = list(map(lambda s: pad_string(s, MAX_LEN, PAD), noisy_x))
    noisy_x_idx_tensor = strings_to_index_tensor(padded_noisy_x, MAX_LEN, ENCODER_CHARS, char_to_index)

    noisy_rnn_tensor = to_rnn_tensor(noisy_x_idx_tensor, ENC_CHAR_CNT)

    batch_sz = len(x)

    encoder_hidden = encoder.init_hidden(batch_size=batch_sz)

    for i in range(noisy_rnn_tensor.shape[0]):
        # LSTM requires 3 dimensional inputs
        _, encoder_hidden = encoder(noisy_rnn_tensor[i].unsqueeze(0), encoder_hidden)

    decoder_input = strings_to_tensor([SOS] * batch_sz, max_name_len=1, allowed_chars=DECODER_CHARS,
                                      index_func=char_to_index)
    decoder_hidden = encoder_hidden
    names = [''] * batch_sz

    for i in range(x_idx_tensor.shape[0]):
        decoder_probs, decoder_hidden = decoder(decoder_input, decoder_hidden)
        nonzero_indexes = x_idx_tensor[i]
        best_indexes = torch.squeeze(torch.argmax(decoder_probs, dim=2), dim=0)
        decoder_probs = torch.squeeze(decoder_probs, dim=0)
        best_chars = list(map(lambda idx: index_to_char(int(idx), DECODER_CHARS), best_indexes))
        loss += criterion(decoder_probs, nonzero_indexes.type(torch.LongTensor))

        for i, char in enumerate(best_chars):
            names[i] += char

        decoder_input = strings_to_tensor(best_chars, 1, DECODER_CHARS, char_to_index)

    loss.backward()
    encoder_optim.step()
    decoder_optim.step()

    return names, noisy_x, loss.item()


def iterate_train(dl: DataLoader, epochs: int = EPOCH, path: str = "Checkpoints/", print_every: int = PRINTS,
                  plot_every: int = PRINTS):
    all_losses = []
    total_loss = 0  # Reset every plot_every iters
    start = time.time()
    iteration = 0

    for e in range(epochs):
        for x in dl:
            _, _, loss = denoise_train(x)
            iteration += 1
            total_loss += loss

            if iteration % plot_every == 0:
                all_losses.append(total_loss / plot_every)
                total_loss = 0
                plot_losses(all_losses, filename=f"{CHCK_PT_NAME}{date_time}")
                torch.save({'weights': encoder.state_dict()},
                           os.path.join(f"{path}{CHCK_PT_NAME}encoder_{date_time}.path.tar"))
                torch.save({'weights': decoder.state_dict()},
                           os.path.join(f"{path}{CHCK_PT_NAME}decoder_{date_time}.path.tar"))


def test_w_noise(x: str):
    noisy_x = noise_name(x, ALL_CHARS, MAX_LEN)
    noised_x = string_to_tensor(noisy_x + SOS, ALL_CHARS)
    x = string_to_tensor(x + EOS, ALL_CHARS)

    encoder_hidden = encoder.init_hidden()
    for i in range(noised_x.shape[0]):
        # LSTM requires 3 dimensional inputs
        _, encoder_hidden = encoder(noised_x[i].unsqueeze(0), encoder_hidden)

    decoder_input = strings_to_tensor([SOS], 1, DECODER_CHARS, char_to_index)
    decoder_hidden = encoder_hidden
    output_char = SOS
    name = ''

    while output_char is not EOS and len(name) <= MAX_LEN:
        decoder_probs, decoder_hidden = decoder(decoder_input, decoder_hidden)
        best_indexes = torch.argmax(decoder_probs, dim=2).item()
        output_char = ALL_CHARS[best_indexes]
        name += output_char
        decoder_input = torch.zeros(1, 1, LETTERS_COUNT)
        decoder_input[0, 0, best_indexes] = 1.

    return name, noisy_x


def test_wo_noise(x: str):
    padded_x = pad_string(x, MAX_LEN, PAD)
    x = string_to_tensor(padded_x, ENCODER_CHARS)

    encoder_hidden = encoder.init_hidden()
    for i in range(x.shape[0]):
        # LSTM requires 3 dimensional inputs
        _, encoder_hidden = encoder(x[i].unsqueeze(0), encoder_hidden)

    decoder_input = strings_to_tensor([SOS] * 1, max_name_len=1, allowed_chars=DECODER_CHARS,
                                      index_func=char_to_index)
    decoder_hidden = encoder_hidden
    name = ''

    while len(name) <= MAX_LEN:
        decoder_probs, decoder_hidden = decoder(decoder_input, decoder_hidden)
        best_indexes = torch.argmax(decoder_probs, dim=2).item()
        output_char = DECODER_CHARS[best_indexes]
        name += output_char
        decoder_input = torch.zeros(1, 1, DEC_CHAR_CNT)
        decoder_input[0, 0, best_indexes] = 1.

    return name


def iterate_test(column: str, df: pd.DataFrame, print_every: int = PRINTS):
    start = time.time()
    n_iters = len(df)
    total = 0
    correct = 0
    for iter in range(n_iters):
        input = df.iloc[iter][column]
        name, noised_x = test(input)

        total += 1

        name = name.replace(PAD, '')

        if input == name:
            correct += 1

        if iter % print_every == 0:
            print(f"Total: {total}, Correct: {correct}, Input: {noised_x}, Name:{name}, Original:{input}")

    print(f"Total: {total}, Correct: {correct}")
    return total, correct


def iterate_test_wo_noise(column: str, df: pd.DataFrame, print_every: int = PRINTS):
    start = time.time()
    n_iters = len(df)
    total = 0
    correct = 0
    for iter in range(n_iters):
        input = df.iloc[iter][column]
        name = test_wo_noise(input)

        total += 1

        name = name.replace(PAD, '')

        if input == name:
            correct += 1

        if iter % print_every == 0:
            print(f"Total: {total}, Correct: {correct}, Input: {input}, Output:{name}")

    print(f"Total: {total}, Correct: {correct}")
    return total, correct


train_df = pd.read_csv(TRAIN_PTH)

dataset = NameDataset(train_df, 'name')
dataloader = DataLoader(dataset, batch_size=BATCH_SZ, shuffle=True)

encoder = Encoder(ENC_CHAR_CNT, HIDD_SZ)
decoder = Decoder(DEC_CHAR_CNT, HIDD_SZ, DEC_CHAR_CNT)

# criterion = nn.NLLLoss()

# current_DT = datetime.datetime.now()
# date_time = current_DT.strftime("%Y-%m-%d")

# encoder_optim = torch.optim.Adam(encoder.parameters(), lr=LR)
# decoder_optim = torch.optim.Adam(decoder.parameters(), lr=LR)

# iterate_train(dataloader)


encoder.load_state_dict(torch.load("Checkpoints/encoder_2020-02-11.path.tar")['weights'])
decoder.load_state_dict(torch.load("Checkpoints/decoder_2020-02-11.path.tar")['weights'])

print(test_wo_noise("Frank Wood"))
