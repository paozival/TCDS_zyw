# encoding=utf-8
"""
    Created on 21:29 2018/11/12 
    @author: Jindong Wang
"""
import numpy as np
import scipy.io as io
import scipy.linalg
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,KFold,LeaveOneOut
from sklearn.metrics import accuracy_score,f1_score


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(
                np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, None, gamma)
    return K


class TCA:
    def __init__(self, kernel_type='primal', dim=30, lamb=1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma

    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = K @ M @ K.T + self.lamb * np.eye(n_eye), K @ H @ K.T
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]
        Z = A.T @ K
        Z /= np.linalg.norm(Z, axis=0)

        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        return Xs_new, Xt_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform Xs and Xt, then make predictions on target using 1NN
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, Xt_new = self.fit(Xs, Xt)
        clf = LogisticRegression()
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt_new)
        accuracy = sklearn.metrics.accuracy_score(Yt, y_pred)
        f1 = f1_score(Yt, y_pred,zero_division=True,average='macro')
        return accuracy, y_pred, f1

    # TCA code is done here. You can ignore fit_new and fit_predict_new.
"""
    def fit_new(self, Xs, Xt, Xt2):
        '''
        Map Xt2 to the latent space created from Xt and Xs
        :param Xs : ns * n_feature, source feature
        :param Xt : nt * n_feature, target feature
        :param Xt2: n_s, n_feature, target feature to be mapped
        :return: Xt2_new, mapped Xt2 with projection created by Xs and Xt
        '''
        # Computing projection matrix A from Xs an Xt
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        ns, nt = len(Xs), len(Xt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M = e * e.T
        M = M / np.linalg.norm(M, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot(
            [K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]

        # Compute kernel with Xt2 as target and X as source
        Xt2 = Xt2.T
        K = kernel(self.kernel_type, X1=Xt2, X2=X, gamma=self.gamma)

        # New target features
        Xt2_new = K @ A

        return Xt2_new

    def fit_predict_new(self, Xt, Xs, Ys, Xt2, Yt2):
        '''
        Transfrom Xt and Xs, get Xs_new
        Transform Xt2 with projection matrix created by Xs and Xt, get Xt2_new
        Make predictions on Xt2_new using classifier trained on Xs_new
        :param Xt: ns * n_feature, target feature
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt2: nt * n_feature, new target feature
        :param Yt2: nt * 1, new target label
        :return: Accuracy and predicted_labels on the target domain
        '''
        Xs_new, _ = self.fit(Xs, Xt)
        Xt2_new = self.fit_new(Xs, Xt, Xt2)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt2_new)
        acc = sklearn.metrics.accuracy_score(Yt2, y_pred)

        return acc, y_pred


def train_valid():
    # If you want to perform train-valid-test, you can use this function
    domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']
    for i in [1]:
        for j in [2]:
            if i != j:
                src, tar = 'data/' + domains[i], 'data/' + domains[j]
                src_domain, tar_domain = scipy.io.loadmat(
                    src), scipy.io.loadmat(tar)
                Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']

                # Split target data
                Xt1, Xt2, Yt1, Yt2 = train_test_split(
                    Xt, Yt, train_size=50, stratify=Yt, random_state=42)

                # Create latent space and evaluate using Xs and Xt1
                tca = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
                acc1, ypre1 = tca.fit_predict(Xs, Ys, Xt1, Yt1)

                # Project and evaluate Xt2 existing projection matrix and classifier
                acc2, ypre2 = tca.fit_predict_new(Xt1, Xs, Ys, Xt2, Yt2)

    print(f'Accuracy of mapped source and target1 data : {acc1:.3f}')  # 0.800
    print(f'Accuracy of mapped target2 data            : {acc2:.3f}')  # 0.706
"""

# Note: if the .mat file names are not the same, you can change them.
# Note: to reproduce the results of my transfer learning book, use the dataset here: https://www.jianguoyun.com/p/DWJ_7qgQmN7PCBj29KsD

file_dir = r'D:\EEGRecognition\Project_2103\labeled_features\de_2s\try\data.mat'
data = io.loadmat(file_dir)
gamma = data['gamma'].reshape(32,120,320)
theta = data['theta'].reshape(32,120,320)
beta = data['beta'].reshape(32,120,320)
alpha = data['alpha'].reshape(32,120,320)
x = data['x'].reshape(32,120,320)
label1 = data['arousal'].reshape(32,120,1)
label2 = data['valence'].reshape(32,120,1)
y = label2
x = x

loo = LeaveOneOut()
acc = 0
F1 = 0
acc_list,f1_list=[],[]
pred_list=[]
target_list=[]
# kf = KFold(n_splits=10,shuffle=False,random_state=None)

# 调用split方法切分数据
for i, (train_index, test_index) in enumerate(loo.split(x)):
    train_data = np.reshape(x[train_index], (3720, 320))
    test_data = np.reshape(x[test_index], (120, 320))
    train_label = np.reshape(y[train_index], (3720, 1))
    test_label = np.reshape(y[test_index], (120, 1))
    Xs, Ys, Xt, Yt = train_data,train_label,test_data,test_label
    tca = TCA(kernel_type='linear', dim=30, lamb=1, gamma=1)
    accuracy, ypred, f1 = tca.fit_predict(Xs, Ys, Xt, Yt)
    pred_list.append(ypred)
    target_list.append(Yt)
    print(f"=======================第{i + 1}次===================================")
    print("测试集：", accuracy)
    acc += accuracy
    F1 += f1
    f1_list.append(f1)
    acc_list.append(accuracy)
predict_label=np.array(pred_list)
target_label=np.array(target_list),
io.savemat(f'./predict_label_save/deap_tca_x_y2.mat',{'x_pred': predict_label, 'x_label': target_label})
print("测试集平均精度为：", acc / 24)
print("测试集平均f1分数为：", F1 / 24)
print(f'std:{np.std(acc_list)*100}')
print(f'f1 std:{np.std(f1_list)*100}')



