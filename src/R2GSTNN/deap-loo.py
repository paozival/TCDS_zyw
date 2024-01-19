import torch
import torch.nn as nn
import torch.utils.data as Data
from model import R2GSTNN
import numpy as np
import scipy.io as io
from sklearn.model_selection import train_test_split,KFold,LeaveOneOut
from sklearn.metrics import f1_score
import random
import argparse


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机数种子
setup_seed(3407)


# 参数调整
def parse_args():
    parser = argparse.ArgumentParser(description='R2GSTNN Training')
    parser.add_mutually_exclusive_group()
    parser.add_argument('--file_dir',
                        type=str,
                        default='D:\EEGRecognition\Project_2103\labeled_features\de_2s/try',
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
                        default=128,
                        help='Batch size for training')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Learning rate')
    parser.add_argument('--epochs',
                        type=int,
                        default=30,
                        help='Number of epochs')
    parser.add_argument('--save_dir',
                        type=str,
                        default='./predict_label_save',
                        help='Directory for saving predict labels')
    parser.add_argument('--save_name',
                        type=str,
                        default='deap_r2gstnn_loo_try.mat',
                        help='File name for saving predict labels')

    return parser.parse_args()




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute(model, data_loader):
    correct_pred, num_examples = 0, 0
    predict_label,target_label=[],[]

    for data,targets in data_loader:
        data = data.to(device)  #
        targets = targets.squeeze(1).to(device)
        class_output,_ ,_,_= model(data,alpha)
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

args = parse_args()
score_list,f1_list = [],[]
predict_list,target_list=[],[]
# 准备数据和标签，每次一个被试做测试，其余被试做训练
loo = LeaveOneOut()
subject_num = [i for i in range(1, 33)]
file_dir = args.file_dir

for n, (train_index, test_index) in enumerate(loo.split(subject_num)):
    print(f'======================第{n+1}次交叉验证======================')
    # 搭建神经网络模型
    model = R2GSTNN(1).to(device)
    # 创建loss function 交叉熵
    loss_domain = nn.CrossEntropyLoss()
    loss_class = nn.CrossEntropyLoss()
    # 定义优化器 adam
    learning_rate = args.lr
    BATCH_SIZE = args.batch_size
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # weight_decay=1e-3
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1, last_epoch=-1)
    epochs = args.epochs

    feature_name = args.band_name
    label_name = args.label_name

    # 测试集
    test_file = io.loadmat(f'{file_dir}/p{int(test_index+1)}')
    test_data, test_labels = test_file[feature_name], test_file[label_name]
    test_data = torch.FloatTensor(test_data)
    test_label = torch.LongTensor(test_labels)

    # 训练集
    tr_data,tr_labels = [],[]
    for i in train_index:
        train_file = io.loadmat(f'{file_dir}/p{i+1}')
        train_feature = train_file[feature_name]
        tr_label = train_file[label_name]
        tr_data.append(train_feature)
        tr_labels.append(tr_label)

    train_data = torch.FloatTensor(np.reshape(np.array(tr_data),(3720,32,10)))
    train_label = torch.LongTensor(np.reshape(np.array(tr_labels),(3720,1)))
    # val_data = torch.FloatTensor(np.reshape(np.array(tr_data),(3720,32,10)))[3500:,:,:]
    # val_label = torch.LongTensor(np.reshape(np.array(tr_labels2), (3720, 1)))[3500:,:]

    train_set = Data.TensorDataset(train_data,train_label)
    test_set = Data.TensorDataset(test_data,test_label)
    # val_set = Data.TensorDataset(val_data,val_label)
    train_loader = Data.DataLoader(train_set,batch_size=BATCH_SIZE)
    # val_loader = Data.DataLoader(val_set,batch_size=BATCH_SIZE)
    test_loader = Data.DataLoader(test_set,batch_size=BATCH_SIZE)
    len_dataloader = len(train_loader)

    for epoch in range(epochs):
        # start training
        model.train()
        for batch_idx, source_data in enumerate(train_loader):
            p = float(batch_idx + epoch * len_dataloader) / epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            s_data,s_label = source_data
            s_domain_label = torch.zeros(len(s_label))
            s_domain_label = s_domain_label.long().to(device)
            s_class_label = s_label.squeeze(1).to(device)
            s_data = s_data.to(device)#
            s_class_output,_,s_domain_output,_ = model(s_data,alpha)
            # 创建loss function 交叉熵
            loss_s_domain = loss_domain(s_domain_output,s_domain_label)
            loss_s_label = loss_class(s_class_output,s_class_label)

            for data,target in test_loader:
                t_data = data.to(device)#
                t_domain_label = torch.ones(len(target))
                t_domain_label = t_domain_label.long().to(device)
                _,_,t_domain_output,_ = model(t_data, alpha)
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

    model.eval()
    test_acc, test_f1, test_predict, test_target = compute(model, test_loader)
    print('Epoch: %03d/%03d test accuracy: %.2f%% ' % (
        epoch + 1, epochs, test_acc))
    f1_list.append(test_f1)
    score_list.append(test_acc.cpu().numpy())
    predict_list.append(test_predict)
    target_list.append(test_target)
predict_label = np.array(predict_list)
target_label = np.array(target_list)

# io.savemat(f'{args.save_dir}/{args.save_name}',{'x_pred':predict_label,'x_label':target_label})
print('====================END=====================')
print(f'highest test accuracy:{max(score_list)}')
print(f'average test accuracy:{np.mean(score_list)}')
print(f'f1 score:{np.mean(f1_list)}')
"""
        len_dataloader = min(len(train_loader), len(test_loader))
        data_source_iter = iter(train_loader)
        data_target_iter = iter(test_loader)

        i = 0
        while i < len_dataloader:
            p = float(i + epoch * len_dataloader) / epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            data_source = data_source_iter.next()
            s_data, s_label = data_source
            # mynet.zero_grad()
        # for batch_idx, (data, target) in enumerate(train_loader):
            data = s_data.unsqueeze(1).to(device)
            domain_label = torch.zeros(len(s_label))
            domain_label = domain_label.long().to(device)
            class_label = s_label.squeeze(1).to(device)
            class_output, domain_output = model(data,alpha)
            # 创建loss function 交叉熵
            loss_s_label = loss_class(class_output,class_label)
            loss_s_domain = loss_domain(domain_output,domain_label)

            data_target = data_target_iter.next()
            t_data, _ = data_target
            domain_label = torch.ones(len(t_data))
            domain_label = domain_label.long().to(device)
            data = t_data.unsqueeze(1).to(device)
            _, domain_output = model(data, alpha)
            loss_t_domain = loss_domain(domain_output, domain_label)
            loss = loss_t_domain + loss_s_domain + loss_s_label
            # 利用optimizer优化，首先梯度清零
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            i += 1
            print('epoch: %d, [iter: %d / all %d], loss_s_label: %f, loss_s_domain: %f, loss_t_domain: %f' \
            % (epoch, i, len_dataloader, loss_s_label.cpu().data.numpy(),
               loss_s_domain.cpu().data.numpy(), loss_t_domain.cpu().data.numpy()))
            # if not i % 5:
            # print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
            #   % (epoch + 1, epochs, i,
            #      len(train_loader), loss))

#
#         # start validating
#         model = model.eval()
#         correct_pred, num_examples = 0, 0
#         for data, targets in val_loader:
#             data = data.unsqueeze(1).to(device)
#             targets = targets.squeeze(1).to(device)
#             logits, probas = model(data)
#             loss = loss_fn(logits, targets)
#             _, predicted_labels = torch.max(probas, 1)
#             num_examples += targets.size(0)
#             correct_pred += (predicted_labels == targets).sum()
#         acc = correct_pred.float() / num_examples * 100
#         print('Epoch: %03d/%03d validation accuracy: %.2f%% loss:%.2f' %(epoch + 1, epochs,acc,loss))
#         # print('Epoch: %03d/%03d validation accuracy: %.2f%%' % (
#         #     epoch + 1, epochs,
#         #     compute_accuracy(model, val_loader)))
#
    model.eval()
    print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader)))
    score_list.append(compute_accuracy(model, test_loader).cpu().numpy())
print('====================END=====================')
print(f'highest test accuracy:{max(score_list)}')
print(f'average test accuracy:{np.mean(score_list)}')

def save():
    torch.save(mynet,'D:\PycharmProjects\pythonProject\deap cnn/first_train.pkl')

def restore():
    torch.load('D:\PycharmProjects\pythonProject\deap cnn/first_train.pkl')
"""


