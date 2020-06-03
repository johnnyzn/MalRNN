import torch
import torch.nn as nn
import torch.nn.functional as F

class MalConv(nn.Module):
    # trained to minimize cross-entropy loss
    def __init__(self, out_size=2, channels=128, window_size=512, embd_size=8):
        super(MalConv, self).__init__()
        self.embd = nn.Embedding(257, embd_size, padding_idx=0)        
        self.window_size = window_size
        self.conv_1 = nn.Conv1d(embd_size, channels, window_size, stride=window_size, bias=True)
        self.conv_2 = nn.Conv1d(embd_size, channels, window_size, stride=window_size, bias=True)        
        self.pooling = nn.AdaptiveMaxPool1d(1)        
        self.fc_1 = nn.Linear(channels, channels)
        self.fc_2 = nn.Linear(channels, out_size)
    
    def forward(self, x):
        x = self.embd(x.long())
        x = torch.transpose(x,-1,-2)
        cnn_value = self.conv_1(x)
        gating_weight = torch.sigmoid(self.conv_2(x))        
        x = cnn_value * gating_weight        
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)        
        return x


class ConvNet(nn.Module):
    def __init__(self,input_length=2000000,window_size=500):
        super(ConvNet, self).__init__()
        self.embed = nn.Embedding(257, 8, padding_idx=0)
        self.conv_1 = nn.Conv1d(4, 16, window_size, stride=4, bias=True)
        self.conv_2 = nn.Conv1d(16, 32, window_size, stride=4, bias=True)
        self.pooling1 = nn.MaxPool1d(4)
        self.conv_3 = nn.Conv1d(32, 64, window_size, stride=8, bias=True)
        self.conv_4 = nn.Conv1d(128, 192, window_size, stride=8, bias=True)
        self.pooling2 = nn.AvgPool1d(4)
        self.fc_1 = nn.Linear(64*959,192)
        self.fc_2 = nn.Linear(192,160)
        self.fc_3 = nn.Linear(160,128)
        self.fc_4 = nn.Linear(192,1)
        self.sigmoid = nn.Sigmoid()
        self.selu = nn.SELU()
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.embed(x)
        x = torch.transpose(x,-1,-2)
        x =  self.conv_1(x.narrow(-2, 4, 4))
        x =  self.relu(x)
        x =  self.conv_2(x)
        x =  self.relu(x)
        x = self.pooling1(x)
        x =  self.conv_3(x)
        x =  self.relu(x)
        x = self.pooling2(x)
        x = x.view(-1,64*959)
        x = self.fc_1(x)
        x =  self.selu(x)
        x = self.fc_4(x)
        return x