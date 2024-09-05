"""
The file of actor and critic architectures.
"""

import torch
from torch import nn
import torch.nn.functional as F

SEQ_LEN = 64

class AutoregressiveActionHead(nn.Module):
    def __init__(self, embedding_size, output_size, activation="tanh"):
        super(AutoregressiveActionHead, self).__init__()
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = embedding_size # autoregressive embedding is also in this size
        self.decode = nn.Sequential(
            nn.Linear(embedding_size+self.hidden_size, self.hidden_size*2),
            nn.ReLU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.ReLU(),
        )
        if activation == "sigmoid":
            self.mapping = nn.Sequential(
                nn.Linear(self.hidden_size, self.output_size),
                nn.Sigmoid()
            )
        elif activation == "tanh":
            self.mapping = nn.Sequential(
                nn.Linear(self.hidden_size, self.output_size),
                nn.Tanh()
            )
        elif activation == "relu":
            self.mapping = nn.Sequential(
                nn.Linear(self.hidden_size, self.output_size),
                nn.ReLU()
            )
        else:
            self.mapping = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, input, prev_embedding=None):
        if prev_embedding == None:
            prev_embedding = torch.zeros_like(input).to(device=input.device)
        embed = self.decode(torch.concat([input, prev_embedding], axis=2))
        return self.mapping(embed), embed

class ResLSTMBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ResLSTMBlock, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, states):
        lstm_out, states = self.lstm(x, states)
        linear_out = self.linear(lstm_out)
        return F.relu(x + linear_out), states


class DeepResLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(DeepResLSTM, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.deep_rnn = nn.ModuleList(
            [ResLSTMBlock(hidden_size) for _ in range(num_layers)]
        )

    def initial_state(self, batch_size, device):
        return [
            (
                torch.zeros(1, batch_size, block.hidden_size).to(device),
                torch.zeros(1, batch_size, block.hidden_size).to(device),
            )
            for block in self.deep_rnn
        ]

    def forward(self, inputs, prev_state):
        inputs = self.encoder(inputs)
        new_states = []
        for block, state in zip(self.deep_rnn, prev_state):
            inputs, new_state = block(inputs, state)
            new_states.append(new_state)
        return inputs, new_states

class DeepResLSTMNetwork(nn.Module):
    def __init__(self, hidden_size, input_size, output_size, num_layers, device):
        super(DeepResLSTMNetwork, self).__init__()
        self._hidden_size = hidden_size
        self._input_embed_size = 32
        self._button_num = output_size - 2

        # input embedding
        self._input_embed = nn.Linear(input_size, self._input_embed_size)

        # DeepResLSTM
        self.deep_res_lstm = DeepResLSTM(self._input_embed_size, hidden_size, num_layers).to(device)

        # Additional layers for autoregressive action head
        self.button_heads = [AutoregressiveActionHead(hidden_size, 1, "sigmoid").to(device) for _ in range(5)]  # ABXYZ
        self.main_stick_head = AutoregressiveActionHead(hidden_size, 17, None).to(device)
        self.c_stick_head = AutoregressiveActionHead(hidden_size, 17, None).to(device)
        self.trigger_head = AutoregressiveActionHead(hidden_size, 5, None).to(device)

    def initial_state(self, batch_size):
        return self.deep_res_lstm.initial_state(batch_size)

    def forward(self, inputs, prev_state):
        # inputs should have shape (seq_len, batch_size, input_size)
        # prev_state should have shape [(h, c), ...] with length num_layers
        inputs = self._input_embed(inputs)
        outputs, state = self.deep_res_lstm(inputs, prev_state)

        buttons = []
        embed = None
        for button_head in self.button_heads:
            button, embed = button_head(outputs, embed)
            buttons.append(button)

        main_stick_output, embed = self.main_stick_head(outputs, embed)
        c_stick_output, embed = self.c_stick_head(outputs, embed)
        trigger_output, embed = self.trigger_head(outputs, embed)

        return buttons, main_stick_output, c_stick_output, trigger_output, state
    
class DeepResLSTMActor(nn.Module):
    def __init__(self, hidden_size, input_size, num_layers):
        super(DeepResLSTMActor, self).__init__()
        self._hidden_size = hidden_size
        self._input_embed_size = 32

        # input embedding
        self._input_embed = nn.Linear(input_size, self._input_embed_size)

        # DeepResLSTM
        self.deep_res_lstm = DeepResLSTM(
            self._input_embed_size, hidden_size, num_layers
        )

        # Additional layers for autoregressive action head
        self.button_heads = nn.ModuleList(
            AutoregressiveActionHead(hidden_size, 1, "sigmoid")
            for _ in range(5)
        )  # ABXYZ
        self.main_stick_head = AutoregressiveActionHead(hidden_size, 17, None)
        self.c_stick_head = AutoregressiveActionHead(hidden_size, 17, None)
        self.trigger_head = AutoregressiveActionHead(hidden_size, 5, None)

    def initial_state(self, batch_size, device):
        return self.deep_res_lstm.initial_state(batch_size, device)

    def forward(self, inputs, prev_state):
        # inputs should have shape (seq_len, batch_size, input_size)
        # prev_state should have shape [(h, c), ...] with length num_layers
        inputs = self._input_embed(inputs)
        outputs, state = self.deep_res_lstm(inputs, prev_state)

        buttons = []
        embed = None
        for button_head in self.button_heads:
            button, embed = button_head(outputs, embed)
            buttons.append(button)

        main_stick_output, embed = self.main_stick_head(outputs, embed)
        c_stick_output, embed = self.c_stick_head(outputs, embed)
        trigger_output, embed = self.trigger_head(outputs, embed)

        return buttons, main_stick_output, c_stick_output, trigger_output, state


class DeepResLSTMCritic(nn.Module):
    def __init__(self, hidden_size, input_size, num_layers):
        super(DeepResLSTMCritic, self).__init__()
        self._hidden_size = hidden_size
        self._input_embed_size = 32

        # input embedding
        self._input_embed = nn.Linear(input_size, self._input_embed_size)

        # DeepResLSTM
        self.deep_res_lstm = DeepResLSTM(
            self._input_embed_size, hidden_size, num_layers
        )

        # value head
        self.value_head = nn.Linear(hidden_size, 1)

    def initial_state(self, batch_size, device):
        return self.deep_res_lstm.initial_state(batch_size, device)

    def forward(self, inputs, prev_state):
        # inputs should have shape (seq_len, batch_size, input_size)
        # prev_state should have shape [(h, c), ...] with length num_layers
        inputs = self._input_embed(inputs)
        outputs, state = self.deep_res_lstm(inputs, prev_state)
        values = self.value_head(outputs)

        return values, state