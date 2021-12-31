import torch
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torch.nn as nn
import torch.utils
import torch.distributions

n_encode = 512

#timedistributed cnn network
class ConvBlock(nn.Module):
    def __init__(self, in_f, out_f, *args, **kwargs):        
        super(ConvBlock, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_f, out_f, *args, **kwargs),
            nn.BatchNorm2d(out_f),
            nn.ReLU()
            )

    def forward(self, x):
        return self.seq(x)

class CNNEncoder(nn.Module):
    def __init__(self, in_c, in_shape = 64 ,channels = [16, 32, 64], n_out = 512):        
        super(CNNEncoder, self).__init__()
        self.channels = [in_c] + channels
        conv_blocks = [ConvBlock(in_f, out_f, kernel_size=3,stride = 2 , padding=1) 
                       for in_f, out_f in zip(self.channels, self.channels[1:])]
        self.cnnmodule = nn.Sequential(*conv_blocks)
        in_n = int(self.channels[-1]*(in_shape//(2**len(channels)))*(in_shape//(2**len(channels))))
        self.linear = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_n, 2048), #change to in_n
            nn.ReLU(),
            nn.Linear(2048, n_encode)
        )
        

    def forward(self, x):
        x = self.cnnmodule(x)
        x = x.view(x.size(0),-1)
        #print('xshape',x.shape)
        x = self.linear(x)
        return x

    
class TimeDistributed(nn.Module):
    def __init__(self, layer, time_steps, *args, device = 'cpu'):        
        super(TimeDistributed, self).__init__()
        self.device = device
        self.layers = nn.ModuleList([layer(*args) for i in range(time_steps)])

    def forward(self, x):

        batch_size, time_steps, C, H, W = x.size()
        output = torch.tensor([]).to(self.device)
        for i in range(time_steps):
          output_t = self.layers[i](x[:, i, :, :, :])
          output_t  = output_t.unsqueeze(1)
          output = torch.cat((output, output_t ), 1)
        return output

class LSTMModule(torch.nn.Module):
    def __init__(self,n_features, n_hidden, n_layers, seq_len, n_out, device = 'cpu', bidirectional = True):
        super(LSTMModule, self).__init__()
        self.n_features = n_features
        self.n_hidden = n_hidden # number of hidden states
        self.n_layers = n_layers # number of LSTM layers (stacked)
        self.device = device
    
        self.l_lstm = nn.LSTM(input_size = self.n_features, 
                                 hidden_size = self.n_hidden,
                                 num_layers = self.n_layers, 
                                 batch_first = True,
                                 bidirectional = bidirectional,
                                 dropout = 0.3)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        l = 2 if bidirectional else 1
        self.decoder_network = nn.Sequential(
            nn.Linear(self.n_hidden*seq_len*l, 128),
            nn.ReLU(),
            nn.Linear(128, n_out)
        )
        
    
    def forward(self, data, device = 'cpu'):        
        batch_size, seq_len, _ = data.size()
        hidden_state0 = torch.zeros(self.n_layers*2,batch_size,self.n_hidden).to(self.device)
        cell_state0 = torch.zeros(self.n_layers*2,batch_size,self.n_hidden).to(self.device)

        lstm_out, (hidden,cell) = self.l_lstm(data.float(),(hidden_state0,cell_state0))
             
        out = lstm_out.contiguous().view(batch_size,-1)
        out = self.decoder_network(out)
        return out


class CNNLSTMNet(nn.Module):
    def __init__(self, channels = 3, ts = 5, n_out = 1, device = 'cpu'):
        super(CNNLSTMNet, self).__init__()
        self.ts_cnn = TimeDistributed(CNNEncoder, ts, channels,device = device)
        self.lstm_decoder = LSTMModule(n_encode, 128, 2, ts, n_out, device=device)

    def forward(self, x):
        x = self.ts_cnn(x)
        x = self.lstm_decoder(x)
        return x

