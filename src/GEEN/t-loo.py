import torch
import torch.nn as nn
from torch_geometric.data import DataLoader,Data
from graph_data import HCIDataset
from sklearn.model_selection import LeaveOneOut
from LTSGAT import mynet
import scipy.io as io
import numpy as np
from sklearn.metrics import f1_score
import itertools
import adabound
import random
import argparse

# 参数调整
def parse_args():
    parser = argparse.ArgumentParser(description='GEEN Training')
    parser.add_mutually_exclusive_group()
    parser.add_argument('--root_dir',
                        type=str,
                        default='D:\EEGRecognition\Project_2103\labeled_features\de_1s',
                        help='data root path')
    parser.add_argument('--raw_dir',
                        type=str,
                        default='try',
                        help='data original folder')
    parser.add_argument('--processed_dir',
                        type=str,
                        default='geen',
                        help='graph data save folder')
    parser.add_argument('--feature_name',
                        type=str,
                        default='gamma',
                        choices=['x', 'theta','alpha', 'beta', 'gamma'],
                        help='feature name')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='Batch size for training')
    # arousal 0.0001 valence 0.001
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Learning rate')
    # arousal 30 valence 15
    parser.add_argument('--epochs',
                        type=int,
                        default=15,
                        help='Number of epochs')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./predict_label_save',
                        help='Directory for saving predict labels')
    parser.add_argument('--save_name',
                        type=str,
                        default='hci_geen_y2_gamma.mat',
                        help='File name for saving predict labels')

    return parser.parse_args()

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机数种子
args = parse_args()
setup_seed(3407)

#准备训练和测试数据集
ROOT_DIR = args.root_dir
RAW_DIR = args.raw_dir
PROCESSED_DIR = args.processed_dir

position_3d = np.array([[-27,83,-3],[-36,76,24],[-71,51,-3],[-48,59,44],
        [-33,33,74],[-78,30,27],[-87,0,-3],[-63,0,61],
        [-33,-33,74],[-78,-30,27],[-71,-51,-3],[-48,-59,44],
        [0,-63,61],[-36,-76,24],[-27,-83,-3],[0,-87,-3],
        [27,-83,-3],[36,-76,24],[48,-59,44],[71,-51,-3],
        [78,-30,27],[33,-33,74],[63,0,61],[87,0,-3],
        [78,30,27],[33,33,74],[48,59,44],[71,51,-3],
        [36,76,24],[27,83,-3],[0,63,61],[0,0,88]])
channel_name = np.array(['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7',
        'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2',
        'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz'])
global_connection = np.array([['Fp1','Fp2'],['AF3','AF4'],['F3','F4'],['FC5','FC6'],
                               ['T7','T8'],['CP5','CP6'],['P3','P4'],['PO3','PO4'],['O1','O2']])

dataset = HCIDataset(ROOT_DIR, RAW_DIR, PROCESSED_DIR, args.feature_name, channel_name, position_3d, global_connection)
a =dataset[0]
# 把dataset分成24组，每组60张图，对应24个被试
list=[]
for i in range(24):
    sub_set = dataset[i*60:(i+1)*60]
    sub_set = [i for i in sub_set]
    list.append(sub_set)

# 利用GPU训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute(model, data_loader):
    correct_pred, num_examples = 0, 0
    predict_label,target_label=[],[]

    for data in data_loader:
        data = data.to(device)
        targets = data.y2.long().to(device)
        logits,class_output,_,_ = model(data,alpha)
        _, predicted_labels = torch.max(class_output, 1)
        # 计算精度,f1分数
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
        predict_label.append(predicted_labels.cpu())
        target_label.append(targets.cpu())

    # 保存预测的标签
    predict_label = np.array([j for i in predict_label for j in i])
    target_label = np.array([j for i in target_label for j in i])

    acc = correct_pred.float() / num_examples * 100
    f1 = f1_score(target_label, predict_label, zero_division=True)

    return acc,f1,predict_label,target_label



score_list = []
f1_list = []
predict_list,target_list=[],[]
# 留一法划分训练集和测试集（跨被:）
subject_id = [i for i in range(24)]

loo = LeaveOneOut()
for n,(train_index,test_index) in enumerate(loo.split(subject_id)):

    # 搭建神经网络模型
    model = mynet().to(device)
    # model = GCN(num_features=10).to(device)
    # 创建loss function 交叉熵

    loss_class = nn.CrossEntropyLoss()

    loss_domain = nn.CrossEntropyLoss()

    # 定义优化器 adam
    learning_rate = args.lr
    BATCH_SIZE = args.batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # ,weight_decay=1e-3
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1, last_epoch=-1)
    epochs = args.epochs

    print(f'===========================第{n + 1 }次交叉验证===============================')
    test_set = list[np.squeeze(test_index)]
    # target_data = torch.zeros(60,32,10)
    # target_label = torch.zeros(60)
    # for i in range(60):
    #     test_data = test_set[i].x
    #     test_label = test_set[i].y
    #     target_data[i,:,:] = test_data
    #     target_label[i] = test_label
    train_set = list[:np.squeeze(test_index)] + list[np.squeeze(test_index) + 1:]
    train_set = [j for i in train_set for j in i]
    # val_set = train_set[1320:]
    # train_set = train_set[:1320]
    # val_loader = DataLoader(val_set,batch_size=BATCH_SIZE)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

    for epoch in range(epochs):
        len_dataloader = len(train_loader)
        # start training
        model.train()
        for batch_idx, data in enumerate(train_loader):
            para = []
            p = float(batch_idx + epoch * len_dataloader) / epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            s_data = data.to(device)
            # y1 arousal y2 valence 在此处改变标签
            s_class_label = data.y2.long().to(device)
            s_domain_label = torch.zeros(len(s_class_label))
            s_domain_label = s_domain_label.long().to(device)
            #正向传播
            s_class_output, s_class_proba, s_domain_output, s_domain_proba = model(s_data, alpha)
            loss_s_label = loss_class(s_class_output, s_class_label)

            loss_s_domain = loss_domain(s_domain_output, s_domain_label)

            # for parameters in model.parameters():
            #     para.append(parameters.data.reshape(-1).unsqueeze(1))
            # para = torch.cat(para, dim=0)

            for data in test_loader:
                t_data = data.to(device)
                t_domain_label = torch.ones(len(t_data.y2))
                t_domain_label = t_domain_label.long().to(device)
                _, _, t_domain_output, t_domain_proba = model(t_data, alpha)

                loss_t_domain = loss_domain(t_domain_output, t_domain_label)

# 0.4 0.6
            loss = 0.4*(loss_t_domain + loss_s_domain) + 0.6*loss_s_label+0.0001*torch.norm(para,p=1)
            # 反向传播，利用optimizer优化，首先梯度清零
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not batch_idx % 50:
                train_acc, _, _, _ = compute(model, train_loader)
                print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f | ACC: %.2f%%'
                  % (epoch + 1, epochs, batch_idx+1,
                     len(train_loader),loss, train_acc))
    # start testing
    model.eval()
    test_acc, test_f1, test_predict, test_target = compute(model, test_loader)
    print('Epoch: %03d/%03d test accuracy: %.2f%% ' % (epoch + 1, epochs, test_acc))
    f1_list.append(test_f1)
    predict_list.append(test_predict)
    score_list.append(test_acc.cpu().numpy())
    target_list.append(test_target)
    # torch.save(model.state_dict(), f'gat-model{n+1}.pt')
    # model = GAT(num_features=10).to(device)
    # state_dict = torch.load('gat.pt')
    # model.load_state_dict(state_dict)
    # model.eval()
    # print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)[0]))
predict_label = np.array(predict_list)
target_label = np.array(target_list)
io.savemat(f'{args.save_dir}/{args.save_name}',{'x_pred':predict_label,'x_label':target_label})
print('====================END=====================')
print(f'highest test accuracy:{max(score_list)}')
print(f'average test accuracy:{np.mean(score_list)}')
print(f'f1 score:{np.mean(f1_list)}')