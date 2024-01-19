import os
import torch
import scipy.io
import numpy as np
from torch_geometric.data import InMemoryDataset, Data, DataLoader
from adjmatrix import get_adjacency_matrix


class DEAPDataset(InMemoryDataset):
    def __init__(self, root, raw_dir, processed_dir, feature_name, channel_name, position_3d, global_connection,
                 transform=None, pre_transform=None):
        """
        :param root:根目录
        :param raw_dir:未经过图处理的原始数据所在文件夹（根目录下的子目录）
        :param processed_dir:原始数据经过图处理、归一化等处理后的保存子目录
        :param pre_transform:在读取数据之前做一个数据预处理的操作
        :param transform:在访问之前动态转换数据对象(因此最好用于数据扩充)
        """
        self._raw_dir = raw_dir
        self._processed_dir = processed_dir
        self.channel_names = channel_name
        self.positions_3d = position_3d
        self.global_connections = global_connection
        self.feature_name = feature_name
        super(DEAPDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):  # 原始数据存放位置，之后进行图处理
        return f'{self.root}/{self._raw_dir}'

    @property
    def processed_dir(self):  # 进行图处理之后的数据存放路径
        return f'{self.root}/{self._processed_dir}'

    @property
    def raw_file_names(self):
        raw_names = [f for f in os.listdir(self.raw_dir)]
        raw_names.sort()
        return raw_names

    @property
    def processed_file_names(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        return [f'deap_processed_graph_{self.feature_name}.dataset']

    def process(self):  # 定义process函数，即如何进行图处理
        # Number of nodes per graph
        # node对应各个channel n_node=32
        n_nodes = 32

        # source node+destination node组成边edge index
        source_nodes, target_nodes = np.repeat(np.arange(0, n_nodes), n_nodes), np.tile(np.arange(0, n_nodes),n_nodes)

        # 处理边权重
        # 此时edge_index,edge_attr包含self-loop
        adj_matrix = get_adjacency_matrix(self.channel_names,self.positions_3d,self.global_connections)
        edge_attr = adj_matrix[source_nodes, target_nodes]
        edge_attr = torch.FloatTensor(edge_attr)
        edge_index = torch.tensor([source_nodes, target_nodes],dtype=torch.long)

        # Remove zero weight links 利用ma模块，创建掩码数组 .mask即edge_attr中的元素＝0输出false，不等于0输出true
        # self-loop去除，在rgnn中再加上
        # mask = np.ma.masked_not_equal(edge_attr, 0).mask
        # # 去除对应权重值为0的node,报错的话改成np.array(e_a)[mask]
        # edge_attr, source_nodes, target_nodes = edge_attr[mask], source_nodes[mask], target_nodes[mask]
        # edge_attr, edge_index = torch.FloatTensor(edge_attr), torch.tensor([source_nodes, target_nodes],
        #                                                                    dtype=torch.long)

        # List of graphs that will be written to file
        participant_data = scipy.io.loadmat(f'{self.raw_dir}/data.mat')
        signal_data = torch.FloatTensor(participant_data[self.feature_name])#.unsqueeze(2)
        arousal = torch.LongTensor(participant_data['arousal'])
        valence = torch.LongTensor(participant_data['valence'])
        #label = valence.squeeze(1)
        label1 = arousal.squeeze(1)
        label2 = valence.squeeze(1)

        # Create graph 创建32x120张图组成dataset，每张图data包含32个节点
        data_list = []
        for trial_num, node_feature in enumerate(signal_data):
            y1 = label1[trial_num]
            y2 = label2[trial_num]
            data = Data(x=node_feature, edge_attr=edge_attr, edge_index=edge_index, y1=y1,y2=y2)# 一张图
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class HCIDataset(InMemoryDataset):
    def __init__(self, root, raw_dir, processed_dir, feature_name, channel_name, position_3d, global_connection,
                 transform=None, pre_transform=None):
        """
        :param root:根目录
        :param raw_dir:未经过图处理的原始数据所在文件夹（根目录下的子目录）
        :param processed_dir:原始数据经过图处理、归一化等处理后的保存子目录
        :param pre_transform:在读取数据之前做一个数据预处理的操作
        :param transform:在访问之前动态转换数据对象(因此最好用于数据扩充)
        """
        self._raw_dir = raw_dir
        self._processed_dir = processed_dir
        self.channel_names = channel_name
        self.feature_name = feature_name
        self.positions_3d = position_3d
        self.global_connections = global_connection
        super(HCIDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):  # 原始数据存放位置，之后进行图处理
        return f'{self.root}/{self._raw_dir}'

    @property
    def processed_dir(self):  # 进行图处理之后的数据存放路径
        return f'{self.root}/{self._processed_dir}'

    @property
    def raw_file_names(self):
        raw_names = [f for f in os.listdir(self.raw_dir)]
        raw_names.sort()
        return raw_names

    @property
    def processed_file_names(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        return [f'hci_processed_graph_{self.feature_name}.dataset']

    def process(self):  # 定义process函数，即如何进行图处理
        # Number of nodes per graph
        # node对应各个channel n_node=32
        n_nodes = 32

        # source node+destination node组成边edge index
        source_nodes, target_nodes = np.repeat(np.arange(0, n_nodes), n_nodes), np.tile(np.arange(0, n_nodes),n_nodes)

        # 处理边权重
        # 此时edge_index,edge_attr包含self-loop
        adj_matrix = get_adjacency_matrix(self.channel_names,self.positions_3d,self.global_connections)
        edge_attr = adj_matrix[source_nodes, target_nodes]
        edge_attr = torch.FloatTensor(edge_attr)
        edge_index = torch.tensor([source_nodes, target_nodes],dtype=torch.long)

        # Remove zero weight links 利用ma模块，创建掩码数组 .mask即edge_attr中的元素＝0输出false，不等于0输出true
        # self-loop去除，在rgnn中再加上
        # mask = np.ma.masked_not_equal(edge_attr, 0).mask
        # # 去除对应权重值为0的node,报错的话改成np.array(e_a)[mask]
        # edge_attr, source_nodes, target_nodes = edge_attr[mask], source_nodes[mask], target_nodes[mask]
        # edge_attr, edge_index = torch.FloatTensor(edge_attr), torch.tensor([source_nodes, target_nodes],
        #                                                                    dtype=torch.long)

        # List of graphs that will be written to file
        participant_data = scipy.io.loadmat(f'{self.raw_dir}/data.mat')
        signal_data = torch.FloatTensor(participant_data[self.feature_name])#.unsqueeze(2)
        arousal = torch.LongTensor(participant_data['arousal'])
        valence = torch.LongTensor(participant_data['valence'])
        #label = valence.squeeze(1)
        label1 = arousal.squeeze(1)
        label2 = valence.squeeze(1)

        # Create graph 创建32x120张图组成dataset，每张图data包含32个节点
        data_list = []
        for trial_num, node_feature in enumerate(signal_data):
            y1 = label1[trial_num]
            y2 = label2[trial_num]
            data = Data(x=node_feature, edge_attr=edge_attr, edge_index=edge_index, y1=y1,y2=y2)# 一张图
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])