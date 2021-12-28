import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import TransformerEncoderLayer, TransformerEncoder, Transformer, LayerNorm
from torch.autograd.grad_mode import F
from torch.nn.init import xavier_uniform_
from torch.nn.utils import weight_norm


class EncoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, in_channel, encoding_size, num_layers=4, device='cuda:0', dropout=0, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.in_channel = in_channel
        self.num_layers = num_layers
        self.encoding_size = encoding_size
        self.bidirectional = bidirectional
        self.device = device

        self.rnn = torch.nn.GRU(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=False, dropout=dropout,
                                bidirectional=bidirectional).to(self.device)
        self.nn = torch.nn.Sequential(torch.nn.Linear(self.hidden_size * (int(self.bidirectional) + 1), self.encoding_size)).to(self.device)

    def forward(self, x):
        x = x.permute(1, 0, 2).to(self.device).float()
        past = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), x.shape[1], self.hidden_size).to(self.device)
        out, _ = self.rnn(x, past)  # out shape = [seq_len, batch_size, num_directions*hidden_size]
        encodings = self.nn(out[-1])
        return encodings


class DecoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, in_channel, encoding_size, num_layers=4, device='cuda:0', dropout=0, bidirectional=True, window=0):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.in_channel = in_channel
        self.num_layers = num_layers
        self.encoding_size = encoding_size
        self.bidirectional = bidirectional
        self.window = window
        self.device = device

        self.rnn = torch.nn.GRU(input_size=self.encoding_size, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=False, dropout=dropout,
                                bidirectional=bidirectional).to(self.device)
        self.nn = torch.nn.Sequential(torch.nn.Linear(self.hidden_size * (int(self.bidirectional) + 1), self.in_channel)).to(self.device)

    def forward(self, x):
        # in shape = [batch_size, hidden_size] --> [seq_len, batch_size, hidden_size]
        x = x.reshape(1, -1, self.encoding_size).repeat(self.window, 1, 1).to(self.device).float()
        out, _ = self.rnn(x)  # out shape = [seq_len, batch_size, num_directions*hidden_size]
        out = self.nn(out)
        out = out.permute(1, 0, 2).to(self.device).float()
        return out


class Transformation(torch.nn.Module):
    def __init__(self, input_size):
        super(Transformation, self).__init__()
        self.input_size = input_size
        self.linear1 = torch.nn.Linear(self.input_size, self.input_size)
        self.linear2 = torch.nn.Linear(self.input_size, self.input_size)
        self.linear3 = torch.nn.Linear(self.input_size, self.input_size)
        self.linear4 = torch.nn.Linear(self.input_size, self.input_size)
        self.leaky_relu = nn.LeakyReLU(0.3)

    def forward(self, x):
        """
        Transforms input x with a mask M(x) followed by multiplication with x.
        """
        h = self.linear1(x.float())
        h = self.leaky_relu(h)

        h = self.linear2(h)
        h = self.leaky_relu(h)

        h = self.linear3(h)
        h = self.leaky_relu(h)

        h = self.linear4(h)
        m = torch.sigmoid(h)

        t_x = m * x.float()
        return t_x  # T(x) = M(x) * x (element-wise multiplication)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, device='cpu'):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, device=device))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, device='cpu'):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout, device=device)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCNEncoder(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, device='cpu'):
        super(TCNEncoder, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout, device=device)
        self.linear = nn.Linear(num_channels[-1], output_size, device=device)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y = self.tcn(inputs.transpose(2, 1))  # input should have dimension (N, C, L)
        out = self.linear(y[:, :, -1])  # last hidden value
        return out, y


class TCNDecoder(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout, device='cpu'):
        super(TCNDecoder, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout, device=device)
        self.linear = nn.Linear(num_channels[-1], output_size, device=device)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y = self.tcn(inputs)  # input should have dimension (N, C, L)
        out = self.linear(y.transpose(2, 1))
        return out


class MyTransformer(nn.Module):
    def __init__(self, feature_size, device, dropout, d_model):
        super(MyTransformer, self).__init__()
        self.linear = nn.Linear(feature_size, d_model, device=device)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=2*d_model, dropout=dropout,
                                                batch_first=True, device=device)
        encoder_norm = LayerNorm(d_model, device=device)
        self.encoder = TransformerEncoder(encoder_layer, 6, encoder_norm)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = 8

        self.batch_first = True
        # self.transformer = Transformer(nhead=8, num_encoder_layers=6, num_decoder_layers=6, d_model=128, dim_feedforward=512, batch_first=True, device=device)
        self.linear2 = nn.Linear(d_model, feature_size, device=device)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src):
        src = self.linear(src)
        # tgt = self.linear(tgt)
        # out = self.transformer(src, tgt)
        out = self.encoder(src)
        out = self.linear2(out)
        return out


class GraphAttentionNetwork(nn.Module):
    def __init__(self, features, window_size, dropout, alpha, d_embed=None, device='cpu'):
        super(GraphAttentionNetwork, self).__init__()
        self.features = features
        self.window_size = window_size
        self.dropout = dropout
        self.d_embed = d_embed
        self.nodes = features
        self.device = device

        # GATv2
        self.d_embed *= 2
        d_input = 2 * window_size
        d_w_input = self.d_embed

        self.linear = nn.Linear(d_input, self.d_embed, device=device)
        self.w = nn.Parameter(torch.empty((d_w_input, 1))).to(device)
        xavier_uniform_(self.w)

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # input shape (batch, window_size, n_feature):
        input = input.permute(0, 2, 1)

        # Dynamic GAT attention: proposed by Brody et. al., 2021
        a_input = self._make_attention_matrix(input)
        a_input = self.leakyrelu(self.linear(a_input))
        e = torch.matmul(a_input, self.w).squeeze(3)

        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        h = self.sigmoid(torch.matmul(attention, input))

        return h.permute(0, 2, 1)

    def _make_attention_matrix(self, x):
        K = self.nodes
        left = x.repeat_interleave(K, dim=1)
        right = x.repeat(1, K, 1)
        combined = torch.cat((left, right), dim=2)  # batch, K*K, 2*window_size
        return combined.view(x.size(0), K, K, 2 * self.window_size)


class FeatureExtractor(nn.Module):
    def __init__(self, features, kernel_size=7, device='cpu'):
        super(FeatureExtractor, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=features, out_channels=features, kernel_size=kernel_size, device=device)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.conv(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        return x


class MyModel(nn.Module):
    def __init__(self, feature, window, kernel_size=5, dropout=0.2, d_model=128, device='cpu'):
        super(MyModel, self).__init__()
        self.extractor = FeatureExtractor(features=feature, kernel_size=kernel_size, device=device)
        self.correlation = GraphAttentionNetwork(features=feature, window_size=window, dropout=dropout, alpha=0.2, d_embed=window, device=device)
        self.temporality = MyTransformer(feature_size=feature, device=device, dropout=dropout, d_model=d_model)
        self.encoder = TCNEncoder(input_size=2 * feature, output_size=2 * d_model, num_channels=[2 * d_model] * 4, kernel_size=kernel_size, dropout=dropout, device=device).to(device)
        self.decoder = TCNDecoder(input_size=2 * d_model, output_size=feature, num_channels=[2 * d_model] * 4, kernel_size=kernel_size, dropout=dropout, device=device).to(device)
        self.device = device

    def forward(self, x):
        v = self.extractor(x).to(self.device)
        h_temp = self.temporality(v).to(self.device)
        h_feat = self.correlation(v).to(self.device)
        h_cat = torch.cat([h_temp, h_feat], dim=2)
        h_last, h = self.encoder(h_cat)  # anchor
        x_hat = self.decoder(h).to(self.device)
        return h_last, x_hat