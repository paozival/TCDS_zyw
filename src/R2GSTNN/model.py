import torch.nn as nn
from grl import ReverseLayerF
from einops import rearrange
import torch
import torch.nn.functional as F
from spatial_learning import *
from regional_temporal import *
from grl import ReverseLayerF

# 此代码用于复现论文 “From Regional to Global Brain: A Novel Hierarchical Spatial-Temporal Neural Network Model for EEG Emotion Recognition”
# DOI 10.1109/TAFFC.2019.2922912



class R2GSTNN(nn.Module):
    def __init__(self, input_size):
        super(R2GSTNN, self).__init__()
        self.emo_categories=3
        self.slstm1 = BiLSTM1(input_size)
        self.slstm2 = BiLSTM2(input_size)
        self.slstm3 = BiLSTM3(input_size)
        self.slstm4 = BiLSTM4(input_size)
        self.slstm5 = BiLSTM5(input_size)
        self.slstm6 = BiLSTM6(input_size)
        self.slstm7 = BiLSTM7(input_size)
        self.slstm8 = BiLSTM8(input_size)
        self.slstm9 = BiLSTM9(input_size)

        self.attention = Region_Att_Layer()

        self.GSL = Spatial_BiLSTM()
        self.project = nn.Linear(30,10)

        self.tlstm1 = LSTM1(20)
        self.tlstm2 = LSTM2(20)
        self.tlstm3 = LSTM3(20)
        self.tlstm4 = LSTM4(20)
        self.tlstm5 = LSTM5(20)
        self.tlstm6 = LSTM6(20)
        self.tlstm7 = LSTM7(20)
        self.tlstm8 = LSTM8(20)
        self.tlstm9 = LSTM9(20)

        self.GTL = Temporal_LSTM(90)

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(410, 200))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_fc2', nn.Linear(200, 50))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(50, 20))
        self.class_classifier.add_module('c_relu3', nn.ReLU(True))
        self.class_classifier.add_module('c_fc4', nn.Linear(20, self.emo_categories))


        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('c_fc1', nn.Linear(410, 200))
        self.domain_classifier.add_module('c_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('c_fc2', nn.Linear(200, 50))
        self.domain_classifier.add_module('c_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('c_fc3', nn.Linear(50, 20))
        self.domain_classifier.add_module('c_relu3', nn.ReLU(True))
        self.domain_classifier.add_module('c_fc4', nn.Linear(20, 2))

        self.batchnorm1 = nn.BatchNorm1d(10)
        self.batchnorm2 = nn.BatchNorm1d(410)

    def forward(self,x,alpha):
        # Spatial feature learning
        # Regional handcraft feature
        region1,region2,region3,region4,region5,region6,region7,region8,region9 = Region_apart(x)

        # Regional feature extraction
        x1 = self.slstm1(region1) # (10b,20)
        x2 = self.slstm2(region2)
        x3 = self.slstm3(region3)
        x4 = self.slstm4(region4)
        x5 = self.slstm5(region5)
        x6 = self.slstm6(region6)
        x7 = self.slstm7(region7)
        x8 = self.slstm8(region8)
        x9 = self.slstm9(region9)
        x = torch.stack([x1,x2,x3,x4,x5,x6,x7,x8,x9], dim=1) # (10b,9,20)

        # Dynamic weighting
        region_feature = self.attention(x) # (10b,9,20)

        # Global spatial feature
        s_pattern = self.GSL(region_feature) # (10b,9,30)
        global_spatial_feature = torch.sigmoid(self.project(s_pattern)) # (10b,9,10)
        global_spatial_feature = rearrange(global_spatial_feature, '(b h) w c -> b h (w c)', h=10)
        global_spatial_feature = self.batchnorm1(global_spatial_feature)

        # Temporal feature learning
        h1 = x[:, 0, :] # (10b,20) -> (b,10,20)
        h2 = x[:, 1, :]
        h3 = x[:, 2, :]
        h4 = x[:, 3, :]
        h5 = x[:, 4, :]
        h6 = x[:, 5, :]
        h7 = x[:, 6, :]
        h8 = x[:, 7, :]
        h9 = x[:, 8, :]

        y1 = self.tlstm1(h1) # (b,10,40)
        y2 = self.tlstm1(h2)
        y3 = self.tlstm1(h3)
        y4 = self.tlstm1(h4)
        y5 = self.tlstm1(h5)
        y6 = self.tlstm1(h6)
        y7 = self.tlstm1(h7)
        y8 = self.tlstm1(h8)
        y9 = self.tlstm1(h9)

        t_pattern = self.GTL(global_spatial_feature)
        rg_feature = torch.cat((y1,y2,y3,y4,y5,y6,y7,y8,y9,t_pattern),dim=1)

        rg_feature = self.batchnorm2(rg_feature)

        logits1 = self.class_classifier(rg_feature)
        probas1 = F.softmax(logits1,dim=1)
        reverse_feature = ReverseLayerF.apply(rg_feature,alpha)
        logits2 = self.domain_classifier(reverse_feature)
        probas2 = F.softmax(logits2,dim=1)
        return logits1,probas1,logits2,probas2













