clc;
clear;
close all;

%将每次试验的EEG分为前，中，后，三个片段。

for zq=1:20 %20次试验
eval(['[data,numChan,labels,txt,fs,gain,prefiltering,ChanDim] = eeg_read_bdf(''D:\EEGRecognition\dataset\dataset_HCI\Part_30_S_Trial' num2str(zq) '_emotion.bdf'',''all'',''n'');'])
%%
%下采样256-128Hz
eeg=zeros(32,size(data,2)/2);
for i=1:32
eeg(i,:)=downsample(data(i,:),2);
end
%%
%参考电极取32导联均值
eeg_mean=mean(eeg);
for i=1:32
eeg(i,:)=eeg(i,:)-eeg_mean;
end
%%
%利用滤波器去眼电
filterorder=7;
%带通设置,值得注意的是这里的参数时归一化后的,1为奈奎斯特频率,即采样频率的一半
filtercutoff=2*3/fs;
%计算滤波器参数,采用高通滤波器
[f_b, f_a]=butter(filterorder,filtercutoff,'high');
%对数据滤波
for i=1:32
segment=eeg(i,:)';
segment=filtfilt(f_b,f_a,segment);
eeg(i,:)=segment';
end
%截取EEG,删除前15s基线和后46s问卷时间
eeg=eeg(:,128*15+1:end-128*46);
%取观看视频时间的最后30s
eeg=eeg(:,(end-30*128+1):end)
%%
%计算EEG信号长度,将每次试验的EEG分为前，中，后，三个片段。
length=size(eeg,2)/128;%注意下采样
length_one_third=floor(length/3);%信号长度的1/3
%%
for z1=1:3 %计算3个片段
eeg_segment=eeg(1:32,1+(z1-1)*128*length_one_third:z1*128*length_one_third);
%%

%%
% %计算eeg时域特征
for i=1:32
eeg_segment_single=eeg_segment(i,:);
% %theta频段（4-8Hz)在f中的索引是theta1-theta2
% %slow_alpha频段(8-10Hz)在f中的索引
% %alpha频段(8-12Hz)在f中的索引是alpha1-alpha2
% %beta频段(12-30Hz)在f中的索引是beta1-beta2
% %gamma频段(30-45Hz)在f中的索引是gamma1-gamma2
Fs=128;%采样频率
T=1/Fs;%采样周期
L=128;%信号长度
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
%每1秒提一个微分熵特征
for t=1:10
eeg_1s=eeg_segment_single(:,(t-1)*128+1:t*128);
% 分频段提取微分熵
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
%汇总所有特征364
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

