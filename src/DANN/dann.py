import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function
from typing import Any, Optional, Tuple




class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class mynet(torch.nn.Module):
    def __init__(self):
        super(mynet, self).__init__()  ###复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        self.emo_categories=4
    
        # 1x32x10 =>16x30x8
        self.conv1 = torch.nn.Conv2d(in_channels=1,
                                      out_channels=16,
                                      kernel_size=(3,3),
                                      stride=1,
                                      padding=0)
        # 16x30x8 => 16x15x4
        self.pool1 = torch.nn.MaxPool2d(kernel_size=(2,2),
                                         stride=2,
                                         padding=0)
        # 16x13x2
        self.conv2 = torch.nn.Conv2d(in_channels=16,
                                     out_channels=16,
                                     kernel_size=(3, 1),
                                     stride=1,
                                     padding=0)
        # 16x6x1
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(2, 1),
                                        stride=2,
                                        padding=0)

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(16*14*2, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, self.emo_categories))
        # self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(16*14*2, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        # self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

        # self.linear1 = torch.nn.Linear(16*6, 2)
        # self.dropout = torch.nn.Dropout(0.5)
    """
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
    """
    def forward(self, x,alpha):

        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        # out.view平铺到一维，类似于keras中的flatten
        feature = x.view(-1,16*14*2)
        reverse_feature = ReverseLayerF.apply(feature,alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output