import torch
import torch.nn as nn
from einops import rearrange

# 与空间维度的BiLSTM区分，命名为LSTM
class LSTM1(nn.Module):
    def __init__(self,input_size):#, sequence_length
        super(LSTM1, self).__init__()

        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, 20,batch_first=True, bidirectional=True)

    def forward(self, x):
        x = rearrange(x, '(b sl) i -> b sl i', sl=10)
        x, _ = self.lstm(x)
        out1 = x[:,-1,:]
        return out1

class LSTM2(nn.Module):
    def __init__(self,input_size):#, sequence_length
        super(LSTM2, self).__init__()
        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, 20,batch_first=True, bidirectional=True)

    def forward(self, x):
        x = rearrange(x, '(b sl) i -> b sl i', sl=10)
        x, _ = self.lstm(x)
        out2 = x[:,-1,:]
        return out2

class LSTM3(nn.Module):
    def __init__(self,input_size):#, sequence_length
        super(LSTM3, self).__init__()
        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, 20,batch_first=True, bidirectional=True)

    def forward(self, x):
        x = rearrange(x, '(b sl) i -> b sl i', sl=10)
        x, _ = self.lstm(x)
        out3 = x[:,-1,:]
        return out3

class LSTM4(nn.Module):
    def __init__(self,input_size):#, sequence_length
        super(LSTM4, self).__init__()
        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size,20,batch_first=True, bidirectional=True)

    def forward(self, x):
        x = rearrange(x, '(b sl) i -> b sl i', sl=10)
        x, _ = self.lstm(x)
        out4 = x[:,-1,:]
        return out4

class LSTM5(nn.Module):
    def __init__(self,input_size):#, sequence_length
        super(LSTM5, self).__init__()
        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size,20,batch_first=True, bidirectional=True)

    def forward(self, x):
        x = rearrange(x, '(b sl) i -> b sl i', sl=10)
        x, _ = self.lstm(x)
        out5 = x[:,-1,:]
        return out5

class LSTM6(nn.Module):
    def __init__(self,input_size):#, sequence_length
        super(LSTM6, self).__init__()
        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, 20,batch_first=True, bidirectional=True)

    def forward(self, x):
        x = rearrange(x, '(b sl) i -> b sl i', sl=10)
        x, _ = self.lstm(x)
        out6 = x[:,-1,:]
        return out6

class LSTM7(nn.Module):
    def __init__(self,input_size):#, sequence_length
        super(LSTM7, self).__init__()

        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, 20,batch_first=True, bidirectional=True)


    def forward(self, x):
        x = rearrange(x, '(b sl) i -> b sl i', sl=10)
        x, _ = self.lstm(x)
        out7 = x[:,-1,:]
        return out7

class LSTM8(nn.Module):
    def __init__(self,input_size):#, sequence_length
        super(LSTM8, self).__init__()

        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, 20,batch_first=True, bidirectional=True)


    def forward(self, x):
        x = rearrange(x, '(b sl) i -> b sl i', sl=10)
        x, _ = self.lstm(x)
        out8 = x[:,-1,:]
        return out8

class LSTM9(nn.Module):
    def __init__(self,input_size):#, sequence_length
        super(LSTM9, self).__init__()

        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, 20,batch_first=True, bidirectional=True)


    def forward(self, x):
        x = rearrange(x, '(b sl) i -> b sl i', sl=10)
        x, _ = self.lstm(x)
        out9 = x[:,-1,:]
        return out9

class Temporal_LSTM(nn.Module):
    def __init__(self,input_size=270):#, sequence_length
        super(Temporal_LSTM, self).__init__()

        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, 25,batch_first=True, bidirectional=True)


    def forward(self, x):
        x, _ = self.lstm(x)
        out = x[:,-1,:]
        return out
