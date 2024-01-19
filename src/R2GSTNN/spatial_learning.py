import torch.nn as nn
from einops import rearrange
import torch
import torch.nn.functional as F

def Region_apart(x):
    region1 = torch.flatten(x[:,0:5,:],1)
    region2 = torch.flatten(x[:, 5:14, :],1)
    region3 = torch.flatten(x[:, 15:22, :],1)
    region4 = torch.flatten(x[:, 24:31, :],1)
    region5 = torch.cat((x[:, 14, :], x[:, 22, :], x[:, 23, :],x[:, 31, :],x[:, 32, :], x[:,40, :]), dim=1)
    region6 = torch.flatten(x[:, 33:40, :],1)
    region7 = torch.flatten(x[:, 41:50, :],1)
    region8 = torch.flatten(x[:, 50:57, :],1)
    region9 = torch.flatten(x[:, 57:62, :],1)
    # region1 = region1.reshape(batch, 4, 128)
    # region2 = region2.reshape(batch, 5, 128)
    # region3 = region3.reshape(batch, 2, 128)
    # region4 = region4.reshape(batch, 3, 128)
    # region5 = region5.reshape(batch, 4, 128)
    # region6 = region6.reshape(batch, 4, 128)
    # region7 = region7.reshape(batch, 5, 128)
    # region8 = region8.reshape(batch, 2, 128)
    # region9 = region9.reshape(batch, 3, 128)
    return region1,region2,region3,region4,region5,region6,region7,region8,region9

class BiLSTM1(nn.Module):
    def __init__(self,input_size):#, sequence_length
        super(BiLSTM1, self).__init__()

        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, 10,batch_first=True, bidirectional=True)

    def forward(self, x):
        x = rearrange(x, 'b (i sl) -> (b sl) i', sl=10)
        x = x.unsqueeze(2)
        x, _ = self.lstm(x)
        out1 = x[:,-1,:]
        return out1

class BiLSTM2(nn.Module):
    def __init__(self,input_size):#, sequence_length
        super(BiLSTM2, self).__init__()
        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, 10,batch_first=True, bidirectional=True)

    def forward(self, x):
        x = rearrange(x, 'b (i sl) -> (b sl) i', sl=10)
        x = x.unsqueeze(2)
        x, _ = self.lstm(x)
        out2 = x[:,-1,:]
        return out2

class BiLSTM3(nn.Module):
    def __init__(self,input_size):#, sequence_length
        super(BiLSTM3, self).__init__()
        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, 10,batch_first=True, bidirectional=True)

    def forward(self, x):
        x = rearrange(x, 'b (i sl) -> (b sl) i', sl=10)
        x = x.unsqueeze(2)
        x, _ = self.lstm(x)
        out3 = x[:,-1,:]
        return out3

class BiLSTM4(nn.Module):
    def __init__(self,input_size):#, sequence_length
        super(BiLSTM4, self).__init__()
        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size,10,batch_first=True, bidirectional=True)

    def forward(self, x):
        x = rearrange(x, 'b (i sl) -> (b sl) i', sl=10)
        x = x.unsqueeze(2)
        x, _ = self.lstm(x)
        out4 = x[:,-1,:]
        return out4

class BiLSTM5(nn.Module):
    def __init__(self,input_size):#, sequence_length
        super(BiLSTM5, self).__init__()
        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size,10,batch_first=True, bidirectional=True)

    def forward(self, x):
        x = rearrange(x, 'b (i sl) -> (b sl) i', sl=10)
        x = x.unsqueeze(2)
        x, _ = self.lstm(x)
        out5 = x[:,-1,:]
        return out5

class BiLSTM6(nn.Module):
    def __init__(self,input_size):#, sequence_length
        super(BiLSTM6, self).__init__()
        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, 10,batch_first=True, bidirectional=True)

    def forward(self, x):
        x = rearrange(x, 'b (i sl) -> (b sl) i', sl=10)
        x = x.unsqueeze(2)
        x, _ = self.lstm(x)
        out6 = x[:,-1,:]
        return out6

class BiLSTM7(nn.Module):
    def __init__(self,input_size):#, sequence_length
        super(BiLSTM7, self).__init__()

        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, 10,batch_first=True, bidirectional=True)


    def forward(self, x):
        x = rearrange(x, 'b (i sl) -> (b sl) i', sl=10)
        x = x.unsqueeze(2)
        x, _ = self.lstm(x)
        out7 = x[:,-1,:]
        return out7

class BiLSTM8(nn.Module):
    def __init__(self,input_size):#, sequence_length
        super(BiLSTM8, self).__init__()

        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, 10,batch_first=True, bidirectional=True)


    def forward(self, x):
        x = rearrange(x, 'b (i sl) -> (b sl) i', sl=10)
        x = x.unsqueeze(2)
        x, _ = self.lstm(x)
        out8 = x[:,-1,:]
        return out8

class BiLSTM9(nn.Module):
    def __init__(self,input_size):#, sequence_length
        super(BiLSTM9, self).__init__()

        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, 10,batch_first=True, bidirectional=True)


    def forward(self, x):
        x = rearrange(x, 'b (i sl) -> (b sl) i', sl=10)
        x = x.unsqueeze(2)
        x, _ = self.lstm(x)
        out9 = x[:,-1,:]
        return out9


class Region_Att_Layer(nn.Module):


    def __init__(self,input=20,output=20):
        super(Region_Att_Layer, self).__init__()

        self.P_linear = nn.Linear(input, output, bias=True)
        self.V_linear = nn.Linear(input, 9, bias=False)


    def forward(self, att_input):

        P = self.P_linear(att_input)

        feature = torch.tanh(P)
        alpha = self.V_linear(feature)
        # 下面开始softmax
        alpha = F.softmax(alpha, dim=2)
        out = torch.matmul(alpha, att_input)
        return out

class Spatial_BiLSTM(nn.Module):
    def __init__(self,input_size=20):#, sequence_length
        super(Spatial_BiLSTM, self).__init__()

        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, 15,batch_first=True, bidirectional=True)


    def forward(self, x):
        x, _ = self.lstm(x)
        return x