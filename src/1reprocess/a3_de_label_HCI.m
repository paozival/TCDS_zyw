%��������������������Ŀ������׼���ͻ���
clc;
clear;
close all;

%HCI�Ե�������ÿ�����԰���60������
load D:\EEGRecognition\Project_2103\features\de_1s\try\p1
x=zscore(feature);%��p1��feature��׼�� ������������ݵľ�ֵΪ0����׼��Ϊ1
theta=zscore(theta_feature);
alpha=zscore(alpha_feature);
beta=zscore(beta_feature);
gamma=zscore(gamma_feature);
for zq=2:24
eval(['load D:\EEGRecognition\Project_2103\features\de_1s\try\p' num2str(zq) ''])
feature=zscore(feature);
theta_f=zscore(theta_feature);
alpha_f=zscore(alpha_feature);
beta_f=zscore(beta_feature);
gamma_f=zscore(gamma_feature);
x=[x;feature];
theta=[theta;theta_f];
alpha=[alpha;alpha_f];
beta=[beta;beta_f];
gamma=[gamma;gamma_f];
end

load D:\EEGRecognition\Project_2103\labels\20s\HCI\p1
y_arousal=arousal;
y_valence=valence;
for zq=2:24
eval(['load D:\EEGRecognition\Project_2103\labels\20s\HCI\p' num2str(zq) ''])
y_arousal=[y_arousal;arousal];
y_valence=[y_valence;valence];
end
arousal=y_arousal;
valence=y_valence;
save D:\EEGRecognition\Project_2103\labeled_features\de_1s\try\data x theta alpha beta gamma arousal valence