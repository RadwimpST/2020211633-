import os
import numpy as np
from torch.utils.data import Dataset
import pickle
import torch

#Dataset 从 torch.utils.data 导入，用于定义自定义数据集
#pickle 用于加载和保存Python对象，这里用来加载标签数据

class DataLoadAdni(Dataset):
    def __init__(self, choose_data,partroi,partition,fold):
        #1 参数
        #choose_data：选择使用的数据集（可能是 ADNI2 或 ADNI3）
        #指定感兴趣的脑区（ROI）数目
        #partition：指定数据的分割（例如，训练集或测试集）
        #fold：交叉验证的折数
        data_path = os.path.join(os.getcwd(),'data/{}'.format(choose_data),
                                 '{}_{}_{}_{}_data.npy'.format(choose_data,partroi,partition,fold))

        label_path = os.path.join(os.getcwd(),'data/{}'.format(choose_data),
                                 '{}_{}_{}_{}_label.pkl'.format(choose_data,partroi,partition,fold))
        #2 构建路径
        # 这部分就是通过组合参数得到需要的数据路径
        #os.getcwd()：获取当前工作目录的路径。
        #data:.npy的路径，label:.pkl的路径

        with open(label_path, 'rb') as f:
            self.labels= pickle.load(f)
        self.datas = torch.from_numpy(np.load(data_path).astype(np.float32))

        #3
        self.labels=torch.tensor(np.array(self.labels))
        # 将标签转为张量

    def __getitem__(self, item):
        label = self.labels[item]
        data = self.datas[item, :, :].transpose(1,0)
        return data, label
    #4 数据格式有待考究

    def __len__(self):
        return len(self.labels)