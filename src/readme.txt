## 1reprocess：网络输入数据处理代码
seed-iv与seed脑电样本划分方式同HCI
1.选取每个trial最后的30s数据进行处理
2.将30s数据均分为3段，每段10s作为一个EEG sample
3.在一个长度为10s的EEG sample中，每1s每个频段提取一个DE特征(theta beta alpha gamma 以及全频段）
最终每名被试提取的DE特征维度为 3Nxchannelsx50


theta：3N*channels*10
alpha:3N*channels*10
beta:3N*channels*10
gamma:3N*channels*10
allband: 3N*channels*10

N-- number of trials

## RGNN GEEN 需要首先下载torch.geometric包!

## 注意保存预测标签结果



