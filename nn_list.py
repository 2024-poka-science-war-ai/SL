"""
The file of actor and critic architectures.
"""

import torch
from torch import nn
import torch.nn.functional as F

SEQ_LEN=30

class ActionHead(nn.Module):
    def __init__(self, hidden_size, output_size, activation="tanh"):
        super(ActionHead, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.linear1 = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ELU(),
        )
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        if activation == "sigmoid":
            self.linear2 = nn.Sequential(
                nn.Linear(self.hidden_size, self.output_size),
                nn.Sigmoid()
            )
        elif activation == "tanh":
            self.linear2 = nn.Sequential(
                nn.Linear(self.hidden_size, self.output_size),
                nn.Tanh()
            )
        elif activation == "relu":
            self.linear2 = nn.Sequential(
                nn.Linear(self.hidden_size, self.output_size),
                nn.ReLU()
            )
        else:
            self.linear2 = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        out = self.linear1(x)
        out = self.batchnorm(out.permute(1, 2, 0)).permute(2, 0, 1)
        return self.linear2(out)


class ResLSTMBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ResLSTMBlock, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.batchnorm = nn.BatchNorm1d(hidden_size)

    def forward(self, x, states):
        lstm_out, states = self.lstm(x, states)
        linear_out = self.linear(lstm_out)
        out = F.elu(x + linear_out)
        out = self.batchnorm(out.permute(1, 2, 0)).permute(2, 0, 1)
        return out, states


class DeepResLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(DeepResLSTM, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.deep_rnn = nn.ModuleList(
            [ResLSTMBlock(hidden_size) for _ in range(num_layers)]
        )
        self.batchnorm = nn.BatchNorm1d(hidden_size)

    def initial_state(self, batch_size, device):
        return [
            (
                torch.zeros(1, batch_size, block.hidden_size).to(device),
                torch.zeros(1, batch_size, block.hidden_size).to(device),
            )
            for block in self.deep_rnn
        ]

    def forward(self, inputs, prev_state):
        inputs = F.elu(self.encoder(inputs))
        inputs = self.batchnorm(inputs.permute(1, 2, 0)).permute(2, 0, 1)
        new_states = []
        for block, state in zip(self.deep_rnn, prev_state):
            inputs, new_state = block(inputs, state)
            new_states.append(new_state)
        return inputs, new_states


class DeepResLSTMActor(nn.Module):
    """
    Actor network consisted of ResLSTMBlocks
    """
    def __init__(self, hidden_size, input_size, num_layers):
        super(DeepResLSTMActor, self).__init__()

        # DeepResLSTM
        self.deep_res_lstm = DeepResLSTM(
            input_size, hidden_size, num_layers
        )

        # Additional layers for autoregressive action head
        self.action_head = ActionHead(hidden_size, 34, activation="none")

    def initial_state(self, batch_size, device):
        """
        Get initial state for LSTM module

        Args:
        - batch_size: int, batch_size of the initial state you want
        - device: torch.cuda.device, device where intial state tensors loaded

        Returns:
        - initial_state: a list of shape [(h, c), ...] \
            containing current states of LSTM module, \
                each tensor's shape is (1, batch_size, hidden_size)
        """

        return self.deep_res_lstm.initial_state(batch_size, device)

    def forward(self, inputs, prev_state):
        """
        Get policy of each state by neural net

        Args:
        - inputs: tensor, a tensor of shape (seq_len, batch_size, input_size)
        - prev_state: list, a list of shape [(h, c), ...] \
            containing previous states of LSTM module

        Returns:
        - values: tensor, a tensor of shape (seq_len, batch_size, 34) \
            containing policy of each state
        - state: list, a list of shape [(h, c), ...] \
            containing current states of LSTM module
        """

        outputs, state = self.deep_res_lstm(inputs, prev_state)
        action_output = self.action_head(outputs)
        return action_output, state


class DeepResLSTMCritic(nn.Module):
    """
    Critic network consisted of ResLSTMBlocks
    """
    def __init__(self, hidden_size, input_size, num_layers):
        super(DeepResLSTMCritic, self).__init__()

        # DeepResLSTM
        self.deep_res_lstm = DeepResLSTM(
            input_size, hidden_size, num_layers
        )

        # value head
        self.value_head = nn.Linear(hidden_size, 1)

    def initial_state(self, batch_size, device):
        """
        Get initial state for LSTM module

        Args:
        - batch_size: int, batch_size of the initial state you want
        - device: torch.cuda.device, device where intial state tensors loaded

        Returns:
        - initial_state: a list of shape [(h, c), ...] \
            containing current states of LSTM module, \
                each tensor's shape is (1, batch_size, hidden_size)
        """

        return self.deep_res_lstm.initial_state(batch_size, device)

    def forward(self, inputs, prev_state):
        """
        Get estimated value of each state by neural net

        Args:
        - inputs: tensor, a tensor of shape (seq_len, batch_size, input_size)
        - prev_state: list, a list of shape [(h, c), ...] \
            containing previous states of LSTM module

        Returns:
        - values: tensor, a tensor of shape (seq_len, batch_size, 1) \
            containing estimated value of each state
        - state: list, a list of shape [(h, c), ...] \
            containing current states of LSTM module
        """

        outputs, state = self.deep_res_lstm(inputs, prev_state)
        values = self.value_head(outputs)
        return values, state
