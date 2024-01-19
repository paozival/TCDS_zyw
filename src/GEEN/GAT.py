import torch
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from einops import rearrange

class GAT(torch.nn.Module):
    def __init__(self,num_features):
        super(GAT,self).__init__()
        self.gat1 = GATConv(num_features,16,heads=4,dropout=0.6)
        self.gat2 = GATConv(64,16,heads=4,dropout=0.6)
        self.gat3 = GATConv(64,16,heads=4,dropout=0.6)
        self.gat4 = GATConv(64,64,heads=1,dropout=0.6)
        # self.dropout = torch.nn.Dropout(0.6)
        #self.lstm = torch.nn.LSTM(16, 8, bidirectional=True)
        self.fc = torch.nn.Linear(64,2)

    def forward(self,data):
        x,edge_index,batch = data.x, data.edge_index,data.batch
        x = F.leaky_relu(self.gat1(x,edge_index))
        x = F.leaky_relu(self.gat2(x,edge_index))
        x = F.leaky_relu(self.gat3(x,edge_index))
        x = F.leaky_relu(self.gat4(x,edge_index))
        x = gmp(x,batch)
        # x = torch.cat((gmp(x, batch), gap(x, batch)), dim=1)
        # x = self.dropout(x)
        # x = rearrange(x, 'b (sl i) -> i b sl', i=1)
        # # h0,c0默认为0,维度(layer_num,batch_size,output_size)
        # x, _ = self.lstm(x)
        # # 将输出转成二维(batch_size,output)

        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas
