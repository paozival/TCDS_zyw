import torch
import torch.nn as nn
import torch.utils.data as Data
from tlstm import EEGLSTM
import numpy as np
import scipy.io as io
from sklearn.model_selection import train_test_split,KFold,LeaveOneOut
from sklearn.metrics import f1_score
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='t-LSTM Training')
    parser.add_mutually_exclusive_group()
    parser.add_argument('--file_dir',
                        type=str,
                        default='D:\EEGRecognition\Project_2103\labeled_features\de_1s/try',
                        help='Feature set root path')
    parser.add_argument('--band_name',
                        type=str,
                        default='x',
                        choices=['x', 'theta', 'beta', 'alpha', 'gamma'],
                        help='different frequency bands')
    parser.add_argument('--label_name',
                        type=str,
                        default='arousal',
                        choices=['arousal', 'valence'],
                        help='label name')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Batch size for training')
    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help='Learning rate')
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help='Number of epochs')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./predict_label_save',
                        help='Directory for saving predict labels')
    parser.add_argument('--save_name',
                        type=str,
                        default='deap-tlstm_loo_label1.mat',
                        help='File name for saving predict labels')

    return parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute(model, data_loader):
    correct_pred, num_examples = 0, 0
    predict_label,target_label=[],[]

    for data,targets in data_loader:
        data = data.to(device)
        targets = targets.squeeze(1).long().to(device)
        logits, _ = model(data)
        _, predicted_labels = torch.max(logits, 1)
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

args = parse_args()
score_list,f1_list = [],[]
predict_list,target_list = [],[]
# 准备数据和标签，每次一个被试做测试，其余被试做训练
loo = LeaveOneOut()
subject_num = [i for i in range(1, 33)]


for n, (train_index, test_index) in enumerate(loo.split(subject_num)):
    print(f'======================第{n+1}次交叉验证======================')
    # 测试集
    # 测试集
    test_file = io.loadmat(f'{args.file_dir}/p{int(test_index+1)}')
    test_data, test_labels = test_file[args.band_name], test_file[args.label_name]
    test_data = torch.FloatTensor(np.reshape(test_data,(120,320)))
    test_label = torch.LongTensor(test_labels)

    # 训练集
    train_data,train_labels=[],[]
    for i in train_index:
        train_file = io.loadmat(f'{args.file_dir}/p{i+1}')
        train_feature = train_file[args.band_name]
        train_label = train_file[args.label_name]
        train_data.append(train_feature)
        train_labels.append(train_label)

    tr_data = torch.FloatTensor(np.reshape(np.array(train_data),(3720,320)))
    tr_label = torch.LongTensor(np.reshape(np.array(train_labels),(3720,1)))
    # val_data = torch.FloatTensor(np.reshape(np.array(train_data),(3720,320)))[3500:,:]
    # val_label = torch.LongTensor(np.reshape(np.array(train_labels1), (3720, 1)))[3500:, :]

    # 搭建神经网络模型
    model = EEGLSTM(num_class=2).to(device)
    # 创建loss function 交叉熵
    loss_class = nn.CrossEntropyLoss()
    loss_domain = nn.CrossEntropyLoss()

    # 定义优化器 adam
    learning_rate = args.lr
    BATCH_SIZE = args.batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # weight_decay=1e-3

    epochs = args.epochs

    train_set = Data.TensorDataset(tr_data,tr_label)
    # val_set = Data.TensorDataset(val_data,val_label)
    test_set = Data.TensorDataset(test_data,test_label)
    train_loader = Data.DataLoader(train_set,batch_size=BATCH_SIZE)
    test_loader = Data.DataLoader(test_set,batch_size=BATCH_SIZE)
    # val_loader = Data.DataLoader(val_set,batch_size=BATCH_SIZE)

    for epoch in range(epochs):
        # start training
        model.train()
        for batch_idx, source_data in enumerate(train_loader):
            s_data, s_label = source_data
            s_domain_label = torch.zeros(len(s_label))
            s_domain_label = s_domain_label.long().to(device)
            s_class_label = s_label.squeeze(1).to(device)
            s_data = s_data.to(device)
            s_class_output, s_domain_output = model(s_data)
            # 创建loss function 交叉熵
            loss_s_label = loss_class(s_class_output, s_class_label)
            loss_s_domain = loss_domain(s_domain_output, s_domain_label)

            t_domain_label = torch.ones(len(test_data))
            t_domain_label = t_domain_label.long().to(device)
            t_data = test_data.to(device)
            _, t_domain_output = model(t_data)
            loss_t_domain = loss_domain(t_domain_output, t_domain_label)

            loss = loss_t_domain + loss_s_domain + loss_s_label
            # 利用optimizer优化，首先梯度清零
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not batch_idx % 50:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                      % (epoch + 1, epochs, batch_idx+1,
                         len(train_loader), loss))

        # # start testing
        # model = model.eval()
        # print('Epoch: %03d/%03d validation accuracy: %.2f%%' % (
        #     epoch + 1, epochs,
        #     compute_accuracy(model, val_loader)))
    # start testing
    model.eval()
    test_acc, test_f1, test_predict, test_target = compute(model, test_loader)
    print('Epoch: %03d/%03d test accuracy: %.2f%% ' % (
        epoch + 1, epochs, test_acc))
    f1_list.append(test_f1)
    predict_list.append(test_predict)
    score_list.append(test_acc.cpu().numpy())
    target_list.append(test_target)

predict_label = np.array(predict_list)
target_label = np.array(target_list)
io.savemat(f'{args.save_dir}/{args.save_name}', {'x_pred': predict_label, 'x_label': target_label})
print('====================END=====================')
print(f'highest test accuracy:{max(score_list)}')
print(f'average test accuracy:{np.mean(score_list)}')
print(f'f1 score:{np.mean(f1_list)}')
print(f'std:{np.std(score_list)}')

# def save():
#     torch.save(EEGLSTM,'D:\PycharmProjects\pythonProject\deap cnn/first_train.pkl')
#
# def restore():
#     torch.load('D:\PycharmProjects\pythonProject\deap cnn/first_train.pkl')



