clc;
clear;
close all;

%��ÿ�������EEG��Ϊǰ���У�������Ƭ�Ρ�

for zq=1:20 %20������
eval(['[data,numChan,labels,txt,fs,gain,prefiltering,ChanDim] = eeg_read_bdf(''D:\EEGRecognition\dataset\dataset_HCI\Part_30_S_Trial' num2str(zq) '_emotion.bdf'',''all'',''n'');'])
%%
%�²���256-128Hz
eeg=zeros(32,size(data,2)/2);
for i=1:32
eeg(i,:)=downsample(data(i,:),2);
end
%%
%�ο��缫ȡ32������ֵ
eeg_mean=mean(eeg);
for i=1:32
eeg(i,:)=eeg(i,:)-eeg_mean;
end
%%
%�����˲���ȥ�۵�
filterorder=7;
%��ͨ����,ֵ��ע���������Ĳ���ʱ��һ�����,1Ϊ�ο�˹��Ƶ��,������Ƶ�ʵ�һ��
filtercutoff=2*3/fs;
%�����˲�������,���ø�ͨ�˲���
[f_b, f_a]=butter(filterorder,filtercutoff,'high');
%�������˲�
for i=1:32
segment=eeg(i,:)';
segment=filtfilt(f_b,f_a,segment);
eeg(i,:)=segment';
end
%��ȡEEG,ɾ��ǰ15s���ߺͺ�46s�ʾ�ʱ��
eeg=eeg(:,128*15+1:end-128*46);
%ȡ�ۿ���Ƶʱ������30s
eeg=eeg(:,(end-30*128+1):end)
%%
%����EEG�źų���,��ÿ�������EEG��Ϊǰ���У�������Ƭ�Ρ�
length=size(eeg,2)/128;%ע���²���
length_one_third=floor(length/3);%�źų��ȵ�1/3
%%
for z1=1:3 %����3��Ƭ��
eeg_segment=eeg(1:32,1+(z1-1)*128*length_one_third:z1*128*length_one_third);
%%

%%
% %����eegʱ������
for i=1:32
eeg_segment_single=eeg_segment(i,:);
% %thetaƵ�Σ�4-8Hz)��f�е�������theta1-theta2
% %slow_alphaƵ��(8-10Hz)��f�е�����
% %alphaƵ��(8-12Hz)��f�е�������alpha1-alpha2
% %betaƵ��(12-30Hz)��f�е�������beta1-beta2
% %gammaƵ��(30-45Hz)��f�е�������gamma1-gamma2
Fs=128;%����Ƶ��
T=1/Fs;%��������
L=128;%�źų���
NFFT=2^nextpow2(L);
%psd=fft(eeg,NFFT)/L;
%psd=2*abs(psd(1:NFFT/2+1));
f=(Fs/2*linspace(0,1,NFFT/2+1))';
theta1=find(f==4);
theta2=find(f==8);
alpha1=theta2+1;
alpha2=find(f==12);
beta1=alpha2+1;
beta2=find(f==30);
gamma1=beta2+1;
gamma2=find(f==45);
%ÿ1����һ��΢��������
for t=1:10
eeg_1s=eeg_segment_single(:,(t-1)*128+1:t*128);
% ��Ƶ����ȡ΢����
theta=eeg_1s(1,theta1:theta2);
beta=eeg_1s(1,beta1:beta2,1);
alpha=eeg_1s(1,alpha1:alpha2);
gamma=eeg_1s(1,gamma1:gamma2);
eeg_de(i,t)=de_entropy(eeg_1s);
theta_de(i,t)=de_entropy(theta);
alpha_de(i,t)=de_entropy(alpha);
beta_de(i,t)=de_entropy(beta);
gamma_de(i,t)=de_entropy(gamma);
end
end
%%
%������������364
feature_everytrial(z1,:,:)=eeg_de;
theta_everytrial(z1,:,:)=theta_de;
alpha_everytrial(z1,:,:)=alpha_de;
beta_everytrial(z1,:,:)=beta_de;
gamma_everytrial(z1,:,:)=gamma_de;
end
feature(zq,:,:,:)=feature_everytrial;
theta_feature(zq,:,:,:)=theta_everytrial;
alpha_feature(zq,:,:,:)=alpha_everytrial;
beta_feature(zq,:,:,:)=beta_everytrial;
gamma_feature(zq,:,:,:)=gamma_everytrial;
end
feature=reshape(feature,[60,32,10]);
theta_feature=reshape(theta_feature,[60,32,10]);
alpha_feature=reshape(alpha_feature,[60,32,10]);
beta_feature=reshape(beta_feature,[60,32,10]);
gamma_feature=reshape(gamma_feature,[60,32,10]);
eval(['save D:\EEGRecognition\Project_2103\features\de_1s\try\p24 feature theta_feature alpha_feature beta_feature gamma_feature'])

