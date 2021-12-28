import math
import random
from pandas import DataFrame
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import TransformerEncoderLayer, TransformerEncoder, Transformer, LayerNorm
from torch.nn.init import xavier_uniform_
from torch.nn.utils import weight_norm


class EncoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, in_channel, encoding_size, num_layers=4, dropout=0, bidirectional=True, device='cuda'):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.in_channel = in_channel
        self.num_layers = num_layers
        self.encoding_size = encoding_size
        self.bidirectional = bidirectional
        self.device = device

        self.rnn = torch.nn.GRU(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=False, dropout=dropout,
                                bidirectional=bidirectional, device=self.device)
        self.nn = torch.nn.Sequential(torch.nn.Linear(self.hidden_size * (int(self.bidirectional) + 1), self.encoding_size, device=self.device))

    def forward(self, x):
        x = x.permute(1, 0, 2).float()
        past = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), x.shape[1], self.hidden_size).to(self.device)
        out, _ = self.rnn(x, past)  # out shape = [seq_len, batch_size, num_directions*hidden_size]
        encodings = self.nn(out[-1])
        return encodings


class DecoderRNN(torch.nn.Module):
    def __init__(self, hidden_size, in_channel, encoding_size, num_layers=4, dropout=0, bidirectional=True, window=0):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.in_channel = in_channel
        self.num_layers = num_layers
        self.encoding_size = encoding_size
        self.bidirectional = bidirectional
        self.window = window

        self.rnn = torch.nn.GRU(input_size=self.encoding_size, hidden_size=self.hidden_size, num_layers=num_layers, batch_first=False, dropout=dropout,
                                bidirectional=bidirectional)
        self.nn = torch.nn.Sequential(torch.nn.Linear(self.hidden_size * (int(self.bidirectional) + 1), self.in_channel))

    def forward(self, x):
        # in shape = [batch_size, hidden_size] --> [seq_len, batch_size, hidden_size]
        x = x.reshape(1, -1, self.encoding_size).repeat(self.window, 1, 1).float()
        out, _ = self.rnn(x)  # out shape = [seq_len, batch_size, num_directions*hidden_size]
        out = self.nn(out)
        out = out.permute(1, 0, 2).float()
        return out


class Transformation(torch.nn.Module):
    def __init__(self, input_size, window_size):
        super(Transformation, self).__init__()
        self.input_size = input_size
        self.window_size = window_size
        self.linear1 = torch.nn.Linear(self.input_size, self.input_size)
        # self.linear2 = torch.nn.Linear(self.input_size, self.input_size)
        # self.linear3 = torch.nn.Linear(self.input_size, self.input_size)
        self.linear4 = torch.nn.Linear(self.input_size, self.input_size)
        self.leaky_relu = nn.LeakyReLU(0.3)
        self.sign_data_grad = torch.zeros(self.input_size)
        self.bias = nn.Parameter(torch.empty(self.window_size, self.input_size))

    def forward(self, x):
        """
        Transforms input x with a mask M(x) followed by multiplication with x.
        """
        x_org = x.detach()
        n_feature = x.shape[2]  # 55
        # ratio = float(np.random.randint(low=50, high=100, size=1))/100.0  # 0.75
        ratio = 0.7
        list = np.arange(0, n_feature, 1).tolist()  # 0 ~ 54
        remain_feature = random.sample(list, int(n_feature * ratio))
        # x = x + 0.1 * self.sign_data_grad
        # x = torch.clamp(x, 0, 1)
        h = self.linear1(x.float())
        h = self.leaky_relu(h)

        # h = self.linear2(h)
        # h = self.leaky_relu(h)
        #
        # h = self.linear3(h)
        # h = self.leaky_relu(h)

        h = self.linear4(h)
        h += self.bias
        # m = torch.sigmoid(h)
        m = torch.tanh(h) + 1.0
        # m = self.leaky_relu(h)
        t_x = m * x.float()
        t_x[:, :, remain_feature] = x_org[:, :, remain_feature]
        return t_x  # T(x) = M(x) * x (element-wise multiplication)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
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
    def __init__(self, num_inputs, num_channels, kernel_size=5, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y = self.tcn(inputs.transpose(2, 1))  # input should have dimension (N, C, L)
        out = self.linear(y[:, :, -1])
        # last_hidden = y[:, :, -1].view(inputs.shape[0], -1)
        return out


def get_pos_encoder():
    return FixedPositionalEncoding


class FixedPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024, scale_factor=1.0):
        super(FixedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # positional encoding
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = scale_factor * pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: [batch, window,dim]
        :return: [batch, window,dim]
        """
        # x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        # x = x.permute(0, 1, 2)
        return x


class MyTransformer(nn.Module):
    def __init__(self, feature_size, dropout, d_model):
        super(MyTransformer, self).__init__()
        self.linear = nn.Linear(feature_size, d_model)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=2*d_model, dropout=dropout, batch_first=True)
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, 2, encoder_norm)
        # self.pos_enc = get_pos_encoder()(d_model, dropout=dropout)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = 8

        self.batch_first = True
        self.linear2 = nn.Linear(d_model, feature_size)
        # self.transformer = Transformer(nhead=8, num_encoder_layers=6, num_decoder_layers=6, d_model=128, dim_feedforward=512, batch_first=True, device=device)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, src):
        """
        :param src: [batch, window, feature]
        :return: [batch, window, feature]
        """
        # tgt = self.linear(tgt)
        # out = self.transformer(src, tgt)
        src = self.linear(src)
        # src = self.pos_enc(src)
        out = self.encoder(src)
        out = self.linear2(out)
        return out


class GraphAttentionNetwork(nn.Module):
    def __init__(self, features, window_size, dropout, alpha, d_embed=None):
        super(GraphAttentionNetwork, self).__init__()
        self.features = features
        self.window_size = window_size
        self.dropout = dropout
        self.d_embed = d_embed if d_embed is not None else window_size
        self.nodes = features

        # GATv2
        self.d_embed *= 2
        d_input = 2 * window_size
        d_w_input = self.d_embed

        # GAT
        # d_input = window_size
        # d_w_input = 2 * self.d_embed

        self.linear = nn.Linear(d_input, self.d_embed)
        self.w = nn.Parameter(torch.empty((d_w_input, 1)))
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

        # GAT
        # Wx = self.linear(input)                                         # (b, k, k, embed_dim)
        # a_input = self._make_attention_matrix(Wx)                       # (b, k, k, 2*embed_dim)
        # e = self.leakyrelu(torch.matmul(a_input, self.w)).squeeze(3)    # (b, k, k, 1)
        # e += self.bias

        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)
        # DataFrame(np.asarray(attention[0].squeeze(0).detach().cpu().numpy())).to_csv('attention.csv')

        h = self.sigmoid(torch.matmul(attention, input))

        return h.permute(0, 2, 1)

    def _make_attention_matrix(self, x):
        K = self.nodes
        left = x.repeat_interleave(K, dim=1)
        right = x.repeat(1, K, 1)
        combined = torch.cat((left, right), dim=2)  # batch, K*K, 2*window_size
        # GATv2
        return combined.view(x.size(0), K, K, 2 * self.window_size)
        # GAT
        # return combined.view(x.size(0), K, K, 2 * self.d_embed)


class FeatureExtractor(nn.Module):
    def __init__(self, features, kernel_size=7):
        super(FeatureExtractor, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=features, out_channels=features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.conv(x)
        x = self.relu(x)
        x = x.permute(0, 2, 1)
        return x


class Forecaster(nn.Module):
    def __init__(self, input_size, output_size, num_layers, dropout=0.2):
        super(Forecaster, self).__init__()
        layer = [nn.Linear(input_size, input_size)]
        for _ in range(num_layers - 1):
            layer.append(nn.Linear(input_size, input_size))
        layer.append(nn.Linear(input_size, output_size))

        self.layers = nn.ModuleList(layer)
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        """Inputs have to have dimension (N, C_in, L_in)"""
        for i in range(len(self.layers) - 1):
            x = self.leakyrelu(self.layers[i](x))
            x = self.dropout(x)
        return self.layers[-1](x)


class MyModel(nn.Module):
    def __init__(self, feature, window, kernel_size=5, dropout=0.2, d_model=256):
        super(MyModel, self).__init__()
        self.extractor = FeatureExtractor(features=feature, kernel_size=kernel_size)
        self.correlation = GraphAttentionNetwork(features=feature, window_size=window, dropout=dropout, alpha=0.5, d_embed=window)
        self.temporality = MyTransformer(feature_size=feature, dropout=dropout, d_model=d_model)
        # self.temporality = TemporalAttentionLayer(feature, window, dropout, 0.2, None, use_gatv2=True)
        self.encoder = TCN(input_size=3 * feature, output_size=d_model, num_channels=[d_model] * 4, kernel_size=kernel_size, dropout=dropout)
        # self.encoder = TCN(input_size=feature, output_size=d_model, num_channels=[d_model] * 4, kernel_size=kernel_size, dropout=dropout)
        # self.forecaster = Forecaster(d_model, feature, 4, dropout=dropout)

    def forward(self, x):
        v = self.extractor(x)
        h_temp = self.temporality(v)
        h_feat = self.correlation(v)
        h_cat = torch.cat([v, h_temp, h_feat], dim=2)
        # h_mul = torch.mul(h_temp, h_feat)
        h_last = self.encoder(h_cat)  # anchor
        # h_last = self.encoder(h_temp)
        pred = torch.zeros(1).cuda()
        # pred = self.forecaster(h_last)
        return h_last, pred

#############################################################################
# MTAD-GAT
#############################################################################
class ConvLayer(nn.Module):
    """1-D Convolution layer to extract high-level features of each time-series input
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param kernel_size: size of kernel to use in the convolution operation
    """

    def __init__(self, n_features, kernel_size=7):
        super(ConvLayer, self).__init__()
        self.padding = nn.ConstantPad1d((kernel_size - 1) // 2, 0.0)
        self.conv = nn.Conv1d(in_channels=n_features, out_channels=n_features, kernel_size=kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = self.relu(self.conv(x))
        return x.permute(0, 2, 1)  # Permute back


class FeatureAttentionLayer(nn.Module):
    """Single Graph Feature/Spatial Attention Layer
    :param n_features: Number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer
    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(FeatureAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.embed_dim = embed_dim if embed_dim is not None else window_size
        self.use_gatv2 = use_gatv2
        self.num_nodes = n_features
        self.use_bias = use_bias

        # Because linear transformation is done after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * window_size
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = window_size
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(n_features, n_features))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For feature attention we represent a node as the values of a particular feature across all timestamps

        x = x.permute(0, 2, 1)

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)                 # (b, k, k, 2*window_size)
            a_input = self.leakyrelu(self.lin(a_input))             # (b, k, k, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)            # (b, k, k, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, k, k, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, k, k, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, k, k, 1)

        if self.use_bias:
            e += self.bias

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        # Computing new node features using the attention
        h = self.sigmoid(torch.matmul(attention, x))

        return h.permute(0, 2, 1)

    def _make_attention_input(self, v):
        """Preparing the feature attention mechanism.
        Creating matrix with all possible combinations of concatenations of node.
        Each node consists of all values of that node within the window
            v1 || v1,
            ...
            v1 || vK,
            v2 || v1,
            ...
            v2 || vK,
            ...
            ...
            vK || v1,
            ...
            vK || vK,
        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)  # (b, K*K, 2*window_size)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.window_size)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class TemporalAttentionLayer(nn.Module):
    """Single Graph Temporal Attention Layer
    :param n_features: number of input features/nodes
    :param window_size: length of the input sequence
    :param dropout: percentage of nodes to dropout
    :param alpha: negative slope used in the leaky rely activation function
    :param embed_dim: embedding dimension (output dimension of linear transformation)
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param use_bias: whether to include a bias term in the attention layer

    """

    def __init__(self, n_features, window_size, dropout, alpha, embed_dim=None, use_gatv2=True, use_bias=True):
        super(TemporalAttentionLayer, self).__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        self.embed_dim = embed_dim if embed_dim is not None else n_features
        self.num_nodes = window_size
        self.use_bias = use_bias

        # Because linear transformation is performed after concatenation in GATv2
        if self.use_gatv2:
            self.embed_dim *= 2
            lin_input_dim = 2 * n_features
            a_input_dim = self.embed_dim
        else:
            lin_input_dim = n_features
            a_input_dim = 2 * self.embed_dim

        self.lin = nn.Linear(lin_input_dim, self.embed_dim)
        self.a = nn.Parameter(torch.empty((a_input_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        if self.use_bias:
            self.bias = nn.Parameter(torch.empty(window_size, window_size))

        self.leakyrelu = nn.LeakyReLU(alpha)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features
        # For temporal attention a node is represented as all feature values at a specific timestamp

        # 'Dynamic' GAT attention
        # Proposed by Brody et. al., 2021 (https://arxiv.org/pdf/2105.14491.pdf)
        # Linear transformation applied after concatenation and attention layer applied after leakyrelu
        if self.use_gatv2:
            a_input = self._make_attention_input(x)              # (b, n, n, 2*n_features)
            a_input = self.leakyrelu(self.lin(a_input))          # (b, n, n, embed_dim)
            e = torch.matmul(a_input, self.a).squeeze(3)         # (b, n, n, 1)

        # Original GAT attention
        else:
            Wx = self.lin(x)                                                  # (b, n, n, embed_dim)
            a_input = self._make_attention_input(Wx)                          # (b, n, n, 2*embed_dim)
            e = self.leakyrelu(torch.matmul(a_input, self.a)).squeeze(3)      # (b, n, n, 1)

        if self.use_bias:
            e += self.bias  # (b, n, n, 1)

        # Attention weights
        attention = torch.softmax(e, dim=2)
        attention = torch.dropout(attention, self.dropout, train=self.training)

        h = self.sigmoid(torch.matmul(attention, x))    # (b, n, k)

        return h

    def _make_attention_input(self, v):
        """Preparing the temporal attention mechanism.
        Creating matrix with all possible combinations of concatenations of node values:
            (v1, v2..)_t1 || (v1, v2..)_t1
            (v1, v2..)_t1 || (v1, v2..)_t2

            ...
            ...

            (v1, v2..)_tn || (v1, v2..)_t1
            (v1, v2..)_tn || (v1, v2..)_t2

        """

        K = self.num_nodes
        blocks_repeating = v.repeat_interleave(K, dim=1)  # Left-side of the matrix
        blocks_alternating = v.repeat(1, K, 1)  # Right-side of the matrix
        combined = torch.cat((blocks_repeating, blocks_alternating), dim=2)

        if self.use_gatv2:
            return combined.view(v.size(0), K, K, 2 * self.n_features)
        else:
            return combined.view(v.size(0), K, K, 2 * self.embed_dim)


class MTAD_GAT(nn.Module):
    """ MTAD-GAT model class.

    :param n_features: Number of input features
    :param window_size: Length of the input sequence
    :param out_dim: Number of features to output
    :param kernel_size: size of kernel to use in the 1-D convolution
    :param feat_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in feat-oriented GAT layer
    :param time_gat_embed_dim: embedding dimension (output dimension of linear transformation)
           in time-oriented GAT layer
    :param use_gatv2: whether to use the modified attention mechanism of GATv2 instead of standard GAT
    :param gru_n_layers: number of layers in the GRU layer
    :param gru_hid_dim: hidden dimension in the GRU layer
    :param forecast_n_layers: number of layers in the FC-based Forecasting Model
    :param forecast_hid_dim: hidden dimension in the FC-based Forecasting Model
    :param recon_n_layers: number of layers in the GRU-based Reconstruction Model
    :param recon_hid_dim: hidden dimension in the GRU-based Reconstruction Model
    :param dropout: dropout rate
    :param alpha: negative slope used in the leaky rely activation function

    """

    def __init__(
        self,
        n_features,
        window_size,
        out_dim,
        kernel_size=7,
        feat_gat_embed_dim=None,
        time_gat_embed_dim=None,
        use_gatv2=True,
        gru_n_layers=1,
        gru_hid_dim=150,
        forecast_n_layers=1,
        forecast_hid_dim=150,
        recon_n_layers=1,
        recon_hid_dim=150,
        dropout=0.2,
        alpha=0.2
    ):
        super(MTAD_GAT, self).__init__()

        self.conv = ConvLayer(n_features, kernel_size)
        self.feature_gat = FeatureAttentionLayer(n_features, window_size, dropout, alpha, feat_gat_embed_dim, use_gatv2)
        self.temporal_gat = TemporalAttentionLayer(n_features, window_size, dropout, alpha, time_gat_embed_dim, use_gatv2)
        # self.gru = GRULayer(3 * n_features, gru_hid_dim, gru_n_layers, dropout)
        self.encoder = TCN(input_size=3 * n_features, output_size=gru_hid_dim, num_channels=[gru_hid_dim] * 4, kernel_size=kernel_size, dropout=dropout)
        # self.forecasting_model = Forecasting_Model(gru_hid_dim, forecast_hid_dim, out_dim, forecast_n_layers, dropout)
        # self.recon_model = ReconstructionModel(window_size, gru_hid_dim, recon_hid_dim, out_dim, recon_n_layers, dropout)

    def forward(self, x):
        # x shape (b, n, k): b - batch size, n - window size, k - number of features

        x = self.conv(x)
        h_feat = self.feature_gat(x)
        h_temp = self.temporal_gat(x)

        h_cat = torch.cat([x, h_feat, h_temp], dim=2)  # (b, n, 3k)

        _, h_end = self.gru(h_cat)
        h_end = h_end.view(x.shape[0], -1)   # Hidden state for last timestamp

        # predictions = self.forecasting_model(h_end)
        # recons = self.recon_model(h_end)

        # return predictions, recons

        return h_end