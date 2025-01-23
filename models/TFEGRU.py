import torch
import torch.nn as nn


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class STFT(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, window_type='hann'):
        super(STFT, self).__init__()
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_size = hop_size
        self.window_type = window_type

    def forward(self, x, mode: str):
        if mode == 'stft':
            x = self._stft(x)
        elif mode == 'istft':
            x = self._istft(x)
        else:
            raise NotImplementedError
        return x

    def _stft(self, x):
        nperseg = self.window_size #n_fft = win_length = seq_len, hop_length = 1 or 6, M = 1 + win_length // 2, N = 1 + win_length // hop_length
        noverlap = 1
        stft_result = torch.stft(x, n_fft=nperseg, hop_length=noverlap, window=None, center=True, return_complex=True)
        return stft_result

    def _istft(self, x):
        nperseg = self.window_size
        noverlap = 1
        istft_result = torch.istft(x, n_fft=nperseg, hop_length=noverlap, window=None, center=True)
        return istft_result

def complex_relu(input):
    return nn.functional.relu(input.real).type(torch.complex64)+1j*nn.functional.relu(input.imag).type(torch.complex64)

class ComplexDropout(nn.Module):
    def __init__(self, dropout_prob):
        super(ComplexDropout, self).__init__()
        self.dropout_prob = dropout_prob

    def forward(self, input):
        if self.training:
            real_part = torch.nn.functional.dropout(input.real, self.dropout_prob, self.training)
            imag_part = torch.nn.functional.dropout(input.imag, self.dropout_prob, self.training)
            return torch.complex(real_part, imag_part)
        else:
            return input

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.pred_len = config.pred_len
        self.input_size = config.input_size
        self.hidden_size = config.hidden_size
        self.seq_len = config.seq_len
        self.rnn_type = config.model
        self.dropout = config.dropout

        self.decompsition = series_decomp(3) # 3 or 7
        self.sample_rate = 44100
        self.freq = 8 // 2 + 1
        if self.seq_len == 5:
            self.time = 1 + self.seq_len
        else:
            self.time = 1 + self.seq_len // 6
        self.embedding_size = self.hidden_size

        self.embedding_layers = nn.Embedding(config.input_size, self.embedding_size)
        self.valueEmbedding = nn.Sequential(
            nn.Linear(1, self.hidden_size),
            nn.ReLU()
        )

        self.stft = STFT(self.sample_rate, 8, self.seq_len - self.pred_len)
        self.fc_layer = nn.Sequential(nn.Linear(self.freq * self.time, self.freq * self.time, dtype=torch.complex64))
        self.dropout_layer = ComplexDropout(self.dropout)
        self.conv = nn.Conv1d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=1, padding=0, stride=1)
        self.rnn = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=1, bias=True,
                           batch_first=True, bidirectional=False)

        self.output = nn.Sequential(nn.Dropout(self.dropout), nn.Linear(self.hidden_size + self.embedding_size, self.pred_len))
        self.act = torch.nn.Tanh()
        self.drop = torch.nn.Dropout(0.05)
        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, (self.hidden_size) // 64, batch_first=True)
        self.attn_linear = nn.Sequential(nn.Linear(self.seq_len + 1, 1), nn.ReLU(), nn.Dropout(self.dropout))

    def forward(self, x):
        embeddings = []
        x = x.squeeze()
        for i in range(self.input_size):
            channel_indices = torch.full((x.size(0), 1,), i, dtype=torch.long, device=x.device)
            embedded_channel = self.embedding_layers(channel_indices)  # using index 1 as the input for the embedding
            embeddings.append(embedded_channel)
        embeddings = torch.cat(embeddings, dim=1)  # bc, 1, 0.5d
        last = x[:, -1:, ]
        x = x - last
        x = x.permute(0, 2, 1)
        seasonal_init = x
        seasonal_init = self.stft(seasonal_init.reshape(seasonal_init.size(0) * self.input_size, self.seq_len),
                                  mode='stft')  # batch*channel seq
        seasonal = seasonal_init.clone()
        freq_season = self.fc_layer(seasonal.reshape(x.size(0) * self.input_size, -1)).view(-1, self.freq, self.time)
        freq_season = self.dropout_layer(freq_season)
        seasonal_init = self.stft(freq_season, mode='istft')
        seasonal_init = seasonal_init.reshape(x.size(0), self.input_size, -1) + x
        seasonal_init = seasonal_init.reshape(x.size(0) * self.input_size, self.seq_len, -1)  # bc, s, hiddensize
        seasonal_init = self.valueEmbedding(seasonal_init)
        x1 = self.drop(self.act(self.conv(seasonal_init.permute(0, 2, 1))))
        seasonal_init = x1.permute(0, 2, 1)
        _, hn = self.rnn(seasonal_init)  # bc, s, d  1, bc, d
        hn = hn.permute(1, 0, 2)
        attn_output, attn_output_weights = self.multihead_attn(hn, _, _)
        attn_output = attn_output * hn
        attn_output = attn_output.reshape(x.size(0), -1, self.hidden_size)
        hn = torch.cat([embeddings, attn_output], dim=2)
        seasonal_output = self.output(hn)
        out = seasonal_output.permute(0, 2, 1)
        out = out + last

        return out