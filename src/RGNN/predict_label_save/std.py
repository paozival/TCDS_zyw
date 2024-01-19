import numpy as np
import scipy.io as io
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
file = io.loadmat('deap_rgnn_x_y1.mat')
predict = file['x_pred']
target = file['x_label']
std,accuracy,matrix,f1 = [],[],[],[]
for i in range(24):
    sub_pred = predict[i,:]
    sub_target = target[i,:]
    sub_confusion = confusion_matrix(sub_target, sub_pred)
    sub_acc = accuracy_score(sub_target,sub_pred)
    sub_f1 = f1_score(sub_target,sub_pred)
    accuracy.append(sub_acc*100)
    f1.append(sub_f1*100)
    matrix.append(sub_confusion)
std = np.std(accuracy)
f1_std=np.std(f1)
ACC = np.mean(accuracy)
print(f1_std)
print(std)
print(ACC)