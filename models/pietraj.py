"""PIETraj (Rasouli et al., ICCV 2019), ported from the Keras reference."""

import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.dense = nn.Linear(seq_len, seq_len)

    def forward(self, x):
        a = torch.sigmoid(self.dense(x.transpose(1, 2)))
        return x * a.transpose(1, 2)


class ElementAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dense = nn.Linear(dim, dim)

    def forward(self, x):
        return x * torch.sigmoid(self.dense(x))


class SoftsignLSTM(nn.Module):
    """LSTM with a softsign cell activation, as Keras ``LSTM(activation="softsign")``."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)
        self.act = nn.Softsign()

    def forward(self, x, state=None):
        b, t, _ = x.shape
        h, c = state if state is not None else (x.new_zeros(b, self.hidden_size), x.new_zeros(b, self.hidden_size))
        outputs = []
        for step in range(t):
            i, f, g, o = (self.x2h(x[:, step]) + self.h2h(h)).chunk(4, dim=1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * self.act(g)
            h = torch.sigmoid(o) * self.act(c)
            outputs.append(h)
        return torch.stack(outputs, dim=1), (h, c)


class PIETraj(nn.Module):
    def __init__(self, observe_length=15, predict_length=45, feature_size=4, hidden_size=256, embed_size=64):
        super().__init__()
        self.predict_length = predict_length
        self.feature_size = feature_size
        self.temporal_attention = TemporalAttention(observe_length)
        self.encoder = SoftsignLSTM(feature_size, hidden_size)
        self.embed = nn.Sequential(nn.Linear(hidden_size, embed_size), nn.ReLU(), nn.Dropout(0.0))
        self.element_attention = ElementAttention(embed_size + feature_size)
        self.decoder = SoftsignLSTM(embed_size + feature_size, hidden_size)
        self.output = nn.Linear(hidden_size, feature_size)

    def forward(self, x):
        _, (h, c) = self.encoder(self.temporal_attention(x))
        embedded = self.embed(h.unsqueeze(1).expand(-1, self.predict_length, -1))
        # the reference decoder input is all zeros
        dec_input = x.new_zeros(x.shape[0], self.predict_length, self.feature_size)
        out, _ = self.decoder(self.element_attention(torch.cat([embedded, dec_input], dim=-1)), state=(h, c))
        return self.output(out)
