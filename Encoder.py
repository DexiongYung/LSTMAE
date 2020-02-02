import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Takes in an one-hot tensor of names and produces hidden state and cell state
    for decoder LSTM to use.

    input_size: N_LETTER
    hidden_size: Size of the hidden dimension
    """

    def __init__(self, input_size, hidden_size, num_layers=4):
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
        # input = input.view(1, self.batch_size, -1)
        lstm_out, hidden = self.lstm(input, hidden)
        return lstm_out, hidden

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))
