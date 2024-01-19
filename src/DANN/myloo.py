
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader,TensorDataset
import os
from scipy.io import loadmat
import argparse
from sklearn.model_selection import LeaveOneOut
from dann import mynet
from sklearn.metrics import f1_score


def compute(model, data_loader):
    correct_pred, num_examples = 0, 0
    predict_label,target_label=[],[]

    for data,targets in data_loader:
        data=data.float()
        data = data.unsqueeze(1).to(device)  #
        targets = targets.squeeze(1).to(device)
        class_output,_ = model(data,alpha)
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
    f1 = f1_score(target_label, predict_label, zero_division=True,average="macro")

    return acc,f1,predict_label,target_label

def parse_args():
    parser = argparse.ArgumentParser(description='DANN Training')
    parser.add_mutually_exclusive_group()
    parser.add_argument('--file_dir',
                        type=str,
                        default='D:/Projects/zhuyiwen_data/seed/',
                        help='Feature set root path')
    parser.add_argument('--band_name',
                        type=str,
                        default='x',
                        choices=['x', 'theta', 'beta', 'alpha', 'gamma'],
                        help='different frequency bands')
    parser.add_argument('--dataset',
                        type=str,
                        default='seed',)
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size for training')
    parser.add_argument('--lr',
                        type=float,
                        default=0.0001,
                        help='Learning rate')
    parser.add_argument('--epochs',
                        type=int,
                        default=30,
                        help='Number of epochs')
    parser.add_argument('--method',
                        type=str,
                        default="DANN")
 

    return parser.parse_args()



score_list,f1_list = [],[]
predict_list, target_list=[],[]
args=parse_args()
num_of_subject=15
# subject_num = [i for i in range(1, num_of_subject+1)]
# loo=LeaveOneOut()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("")

# 2025 -- [72,3,3,15] (trial_num,session_num,segment_num,subject_num)



# subject-independent loop
for p_idx in range(0,num_of_subject):

    tr_p = list(range(0,num_of_subject))
    del tr_p[p_idx]

  
    # load test set
    te_filename=args.file_dir+"p"+str(p_idx+1)+".mat"
    p_te_set = loadmat(te_filename)

    te_data,te_label= p_te_set[args.band_name],p_te_set["label"]
    # (135,62,10) , (135,1)
    te_data = torch.from_numpy(te_data)
    te_label = torch.from_numpy(te_label)

    te_set = TensorDataset(te_data,te_label)
    te_loader = DataLoader(te_set,args.batch_size,shuffle=False)


    tr_datas = []
    tr_labels=[]

    # load train set 
    for i in range(len(tr_p)):
        filename=args.file_dir+"p"+str(tr_p[i]+1)+".mat"
        p_tr_set=loadmat(filename)
        tr_data,tr_label = p_tr_set[args.band_name],p_tr_set["label"]
        # (135,62,10) , (135,1)

        tr_datas.append(tr_data)
        tr_labels.append(tr_label)

    tr_datas=np.concatenate(tr_datas,axis=0)
    tr_labels=np.concatenate(tr_labels,axis=0)
    
    
    tr_datas = torch.from_numpy(tr_datas)
    tr_labels = torch.from_numpy(tr_labels)
    
   
    # tr_data = torch.float(tr_data)
    # tr_label = torch.long(tr_label)
    # tr_set = TensorDataset(tr_data,tr_label)
    # tr_loader = DataLoader(tr_set,args.batch_size,shuffle=True,drop_last=False)
    tr_set = TensorDataset(tr_datas,tr_labels)

    tr_loader= DataLoader(tr_set,args.batch_size,shuffle=True,drop_last=False)
    len_loader = len(tr_loader)

    # define model , loss, optimizer
    model = mynet().to(device)
    loss_class=torch.nn.CrossEntropyLoss()
    loss_domain=torch.nn.CrossEntropyLoss()

    optimizer= torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,10,gamma=0.1,last_epoch=-1)

    for epoch in range(0,args.epochs):
        for batch_idx,(s_data,s_label) in enumerate(tr_loader):
            p = float(batch_idx + epoch * len_loader) / args.epochs / len_loader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            s_domain_label = torch.zeros(len(s_label))
            s_domain_label = s_domain_label.long().to(device)
            s_class_label = s_label.squeeze(1).to(device)
            s_data=s_data.float()
            s_data = s_data.unsqueeze(1).to(device)


            s_class_output, s_domain_output = model(s_data,alpha)
            loss_s_label = loss_class(s_class_output, s_class_label)
            loss_s_domain = loss_domain(s_domain_output,s_domain_label)

            for data,target in te_loader:
                data = data.float()
                t_data = data.unsqueeze(1).to(device)#
                t_domain_label = torch.ones(len(target))
                t_domain_label = t_domain_label.long().to(device)
                _,t_domain_output = model(t_data, alpha)
                loss_t_domain = loss_domain(t_domain_output, t_domain_label)

            loss = loss_t_domain + loss_s_domain + loss_s_label
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not batch_idx % 50:
                print('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f'
                  % (epoch + 1, args.epochs, batch_idx+1,
                     len_loader, loss))

    # evaluating
    model.eval()
    test_acc, test_f1, test_predict, test_target = compute(model, te_loader)
    f1_list.append(test_f1)
    predict_list.append(test_predict)
    score_list.append(test_acc.cpu().numpy())
    target_list.append(test_target)

    



####### save f1_list predict_list score_list target_list
score_list=np.array(score_list) # (15,)
f1_list=np.array(f1_list) # (15,)
predict_result = np.empty(shape=(num_of_subject,predict_list[0].shape[0]),dtype=np.int32)
target_result = np.empty(shape=(num_of_subject,target_list[0].shape[0]),dtype=np.int32)

for i in range(num_of_subject):
    predict_result[i,:] = predict_list[i]
    target_result[i,:] = target_list[i]


# save result
save_name = f"{os.getcwd()}/results/{args.dataset}_{args.method}_{args.band_name}.npz"

# print("check save name")
np.savez(save_name,score_list=score_list,f1_list=f1_list,\
         predict_result=predict_result,target_result=target_result)



print("1")





