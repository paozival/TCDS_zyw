from sklearn.ensemble import RandomForestClassifier
import scipy.io as io
import numpy as np
from sklearn.model_selection import train_test_split,KFold,LeaveOneOut
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
import torch
import random

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

subject_num = [i for i in range(1,24)]
file_dir = r'D:\EEGRecognition\Project_2103\labeled_features\de_1s\try\data.mat'
# feature = []
# label = []
# for subject_id in subject_num:
#     file_dir = r'D:\EEGRecognition\Project_2103\labeled_features\de_2s\DEAP'
#     file_name = f'p{subject_id}'
#     data = io.loadmat(f'{file_dir}/{file_name}')
#     x = data['x']
#     label1 = data['arousal']
#     label2 = data['valence']
#     feature.append(x)
#     label.append(label2)
# x = np.array(feature)
# y = np.array(label)
data = io.loadmat(file_dir)
gamma = data['gamma'].reshape(24,60,320)
theta = data['theta'].reshape(24,60,320)
beta = data['beta'].reshape(24,60,320)
alpha = data['alpha'].reshape(24,60,320)
x = data['x'].reshape(24,60,320)
x = x
# x = np.concatenate((theta,alpha,beta,gamma),axis=2)
label1 = data['arousal'].reshape(24,60,1)
label2 = data['valence'].reshape(24,60,1)
y = label2
# 留一法交叉验证
loo = LeaveOneOut()
acc = 0
F1_score = 0
acc_list,f1_list=[],[]
pred_list,target_list=[],[]
# kf = KFold(n_splits=10,shuffle=False,random_state=None)
# 调用split方法切分数据
for i,(train_index, test_index) in enumerate(loo.split(x)):
    train_data = np.reshape(x[train_index],(1380,320))
    test_data = np.reshape(x[test_index],(60,320))
    train_label = np.reshape(y[train_index],(1380,1))
    test_label = np.reshape(y[test_index],(60,1))
    classifier = RandomForestClassifier()
    classifier.fit(train_data, train_label.ravel())
    tra_label = classifier.predict(train_data)  # 训练集的预测标签
    tes_label = classifier.predict(test_data)
    test_label = np.reshape(test_label,(60,))# 测试集的预测标签
    # c_matrix = confusion_matrix(test_label, tes_label)
    # # print(c_matrix)
    # TP = c_matrix[0][0]
    # FP = c_matrix[0][1]
    # FN = c_matrix[1][0]
    # TN = c_matrix[1][1]
    # P = TP / (TP + FP)  # 精确度
    # R = TP / (TP + FN)  # 召回率
    # F1 = (2 * P * R) / (P + R)
    f1 = f1_score(test_label, tes_label, zero_division=True)
    pred_list.append(tes_label)
    target_list.append(test_label)
    print(f"=======================第{i + 1}次===================================")
    print("训练集：", accuracy_score(train_label, tra_label))
    print("测试集：", accuracy_score(test_label, tes_label))
    acc += accuracy_score(test_label, tes_label)
    acc_list.append(accuracy_score(test_label, tes_label))
    f1_list.append(f1)
predict_label = np.array(pred_list)
target_label = np.array(target_list),
io.savemat(f'./predict_label_save/hci_rf_x_y2.mat', {'x_pred': predict_label, 'x_label': target_label})
print("测试集平均精度为：", acc / 24)
print("测试集平均f1分数为：", np.mean(f1_list))
print(f'std:{np.std(acc_list) * 100}')
print(f'f1 std:{np.std(f1_list) * 100}')
