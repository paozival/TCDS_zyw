from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Function
from typing import Any, Optional, Tuple

class GradientReverseFunction(Function):
    """
    重写自定义的梯度计算方式
    """
    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


class EEGLSTM(nn.Module):

    def __init__(self, 
                 num_layer=1,
                 emo_categories=3,
                 input_channel=62,
                 hidden_size=16):
        super(EEGLSTM, self).__init__()
        # self.num_class = num_class
        self.emo_categories=emo_categories
        self.num_layer = num_layer
        self.input_channel = input_channel
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_channel, hidden_size, num_layer,
                            batch_first=True, bidirectional=True)
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(hidden_size*2*10, 200))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(200))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(200, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, self.emo_categories))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(hidden_size*2*10, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))

    def forward(self, x):
        x = rearrange(x, 'b (sl i) -> b i sl', i=10)
        x, (h, c) = self.lstm(x)
        feature = x.reshape(-1,320)
        reverse_feature = GradientReverseFunction.apply(feature)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output,domain_output