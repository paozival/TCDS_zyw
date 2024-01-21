import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
from typing import Any, Optional, Tuple
from einops import rearrange

# class ReverseLayerF(Function):
#
#     @staticmethod
#     def forward(ctx, x, alpha):
#         ctx.alpha = alpha
#
#         return x.view_as(x)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         output = grad_output.neg() * ctx.alpha
#
#         return output, None

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

def Region_apart(x):
    left = torch.cat((x[:,0,:],x[:,3,:],x[:,5,:],x[:,6,:],x[:,7,:],x[:,8,:],x[:,14,:],x[:,15,:],x[:,16,:],
                     x[:,17,:],x[:,23,:],x[:,24,:],x[:,25,:],x[:,26,:],x[:,32,:],x[:,33,:],x[:,34,:],x[:,35,:],
                     x[:,41,:],x[:,42,:],x[:,43,:],x[:,44,:],x[:,50,:],x[:,51,:],x[:,52,:],
                     x[:,57,:],x[:,58,:]),dim=1)
    # left = torch.reshape(left,(-1,56,10))
    right = torch.cat((x[:,2,:],x[:,4,:],x[:,10,:],x[:,11,:],x[:,12,:],x[:,13,:],x[:,19,:],x[:,20,:],x[:,21,:],
                     x[:,22,:],x[:,28,:],x[:,29,:],x[:,30,:],x[:,31,:],x[:,37,:],x[:,38,:],x[:,39,:],x[:,40,:],
                     x[:,46,:],x[:,47,:],x[:,48,:],x[:,49,:],x[:,54,:],x[:,55,:],x[:,56,:],
                     x[:,60,:],x[:,61,:]),dim=1)
    left = rearrange(left, 'b (i sl) -> b i sl', i=27)
    right = rearrange(right, 'b (i sl) -> b i sl', i=27)
    return left,right

class LSTM(nn.Module):
    def __init__(self,input_size,hidden_size):#, sequence_length
        super(LSTM, self).__init__()
        self.hidden_size= hidden_size

        #对应特征维度
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, hidden_size,batch_first=True, bidirectional=False)


    def forward(self, l,r):
        xl = rearrange(l,'b i sl -> b sl i',i=27)
        xr = rearrange(r,'b i sl -> b sl i',i=27)
        outl, _ = self.lstm(xl)
        outr, _ = self.lstm(xr)
        out = torch.cat((outl,outr),dim=1)
        return out

class mynet(torch.nn.Module):
    def __init__(self):
        super(mynet, self).__init__()  ###复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        self.lstm = LSTM(27,20)
        self.emo_categories=3
        # 1x20x20 =>16x18x18
        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                      out_channels=16,
                                      kernel_size=(3,3),
                                      stride=1,
                                      padding=0)
        # 16x18x18 => 16x9x9
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2,2),
                                         stride=2,
                                         padding=0)
        # 16x7x7
        self.conv2 = torch.nn.Conv2d(in_channels=16,
                                     out_channels=16,
                                     kernel_size=(3, 3),
                                     stride=1,
                                     padding=0)
        # 16x3x3
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 2),
                                        stride=2,
                                        padding=0)

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(16*3*3, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, self.emo_categories))


        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(16*3*3, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))


        # self.linear1 = torch.nn.Linear(16*6, 2)
        # self.dropout = torch.nn.Dropout(0.5)

    # 定义权值初始化
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight.detach())
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.MaxPool2d):
                torch.nn.init.kaiming_normal_(m.weight.data.detach())
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data.detach())
                # m.weight.data.normal_(0,0.01)
                m.bias.data.zero_()

    def forward(self, x, alpha):
        left,right = Region_apart(x)
        out = self.lstm(left,right)
        x = out.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # out.view平铺到一维，类似于keras中的flatten
        feature = x.view(-1,16*3*3)
        reverse_feature = ReverseLayerF.apply(feature,alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output