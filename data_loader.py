import json
import os
from itertools import groupby
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset


class eICUDataset(Dataset):
    def __init__(self, data_path):
        # 初始化文件路径
        self._diagnoses_path = data_path + '/preprocessed_diagnoses.csv'
        self._labels_path = data_path + '/preprocessed_labels.csv'
        self._flat_path = data_path + '/preprocessed_flat.csv'
        self._timeseries_path = data_path + '/preprocessed_timeseries.csv'

        # 读取诊断、标签、平坦特征数据
        self.labels = pd.read_csv(self._labels_path, index_col='patient')
        self.flat = pd.read_csv(self._flat_path, index_col='patient').iloc[:, 1:]
        self.diagnoses = pd.read_csv(self._diagnoses_path, index_col='patient')
        self.time_before_pred = 47

        # 计算时间序列特征数、诊断特征数和平坦特征数
        self.F = (pd.read_csv(self._timeseries_path, index_col='patient', nrows=1).shape[1] - 2)
        self.D = self.diagnoses.shape[1]
        self.no_flat_features = self.flat.shape[1]

        # 获取患者ID列表
        self.patients = list(self.labels.index)
        self.no_patients = len(self.patients)

        # 数据预处理
        self.process_data()

    def line_split(self, line):
        # 将CSV中的每行转换为浮点数列表
        return [float(x) for x in line.split(',')]

    def pad_sequences(self, all_ts_batch, all_patients, time_window):
        # 为时间序列数据添加填充，以便与时间窗口对齐
        padded_timeseries = []
        valid_patients = []
        for ts_batch, patients in zip(all_ts_batch, all_patients):
            if len(ts_batch) >= time_window:
                padded_timeseries.append(ts_batch[:time_window])
                valid_patients.append(patients)
        return padded_timeseries, valid_patients

    def get_los_labels(self, labels, times):
        # 计算住院时间标签，将时间差限制在最小值1/48以上
        times = labels.unsqueeze(1).repeat(1, times.shape[1]) - times
        return times.clamp(min=1 / 48)

    def process_data(self):
        # 读取时间序列数据
        with open(self._timeseries_path, 'r') as timeseries_file:
            self.timeseries_header = next(timeseries_file).strip().split(',')
            ts_data = [self.line_split(line) for line in timeseries_file]

        # 根据患者ID分组时间序列数据
        ts_data_grouped = groupby(ts_data, key=lambda line: line[0])
        timeseries, patients = [], []

        for key, group in ts_data_grouped:
            patient_id = key
            ts_data = [line[1:-1] for line in group]  # 去除无关列
            timeseries.append(ts_data)
            patients.append(patient_id)

        # 按照时间窗口填充
        timeseries, patients = self.pad_sequences(timeseries, patients, time_window=48)
        timeseries = torch.tensor(timeseries)
        timeseries[:, :, 0] /= 24  # 标准化时间序列

        # 读取平坦和诊断特征并进行转换
        flat = self.flat.loc[patients].values.astype(float)
        diagnoses = self.diagnoses.loc[patients].values.astype(int)

        # 读取标签并转换为张量
        self.labels = self.labels.loc[patients]
        los_labels = self.labels.iloc[:, 0].values
        mort_labels = self.labels.iloc[:, 1].values

        # 存储预处理后的数据
        self.los_labels = self.get_los_labels(torch.tensor(los_labels).type(torch.float), timeseries[:, :, 0])[:,
                          self.time_before_pred:]
        self.mort_labels = torch.tensor(mort_labels).type(torch.int)

        self.timeseries = timeseries[:,:,1:]
        self.diag = torch.tensor(diagnoses).type(torch.float)
        self.flat = torch.tensor(flat).type(torch.float)

    def __len__(self):
        # 返回数据集大小
        return len(self.los_labels)

    def __getitem__(self, idx):
        # 根据索引返回样本数据和标签
        return ({'times': self.timeseries[idx],
                 'diags': self.diag[idx],
                 'flat': self.flat[idx]},
                {'los': self.los_labels[idx],
                 'mort': self.mort_labels[idx]})


def get_dataset(args):

    train_loader = []
    test_loader = []
    val_loader = []
    global_test_loader = []

    # 读取医院患者字典，获取用户数量
    with open(args.eicu_dir + '/hospital_patient_dict.json', 'r') as f:
        id_list = list(json.load(f).keys())[0:args.num_users]

    for idx in id_list:
        # 加载每个医院的数据集
        fl_dir = os.path.join(args.eicu_dir, idx)
        hosDataset = eICUDataset(fl_dir)
        total_size = len(hosDataset)
        targets = hosDataset.mort_labels.numpy()

        # 使用StratifiedShuffleSplit进行分层抽样
        strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=args.seed)
        for train_idx, test_val_idx in strat_split.split(range(total_size), targets):
            train_idx = list(train_idx)
            test_val_idx = list(test_val_idx)

            # 分割测试和验证集
            test_val_targets = targets[test_val_idx]
            strat_val_split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=args.seed)
            for val_idx, test_idx in strat_val_split.split(test_val_idx, test_val_targets):
                val_idx = [test_val_idx[i] for i in val_idx]
                test_idx = [test_val_idx[i] for i in test_idx]

        # 创建子集和数据加载器
        train_eICUDataset = Subset(hosDataset, train_idx)
        val_eICUDataset = Subset(hosDataset, val_idx)
        test_eICUDataset = Subset(hosDataset, test_idx)

        hos_train_loader = DataLoader(train_eICUDataset, batch_size=args.local_bs, shuffle=True, drop_last=False)
        hos_val_loader = DataLoader(val_eICUDataset, batch_size=args.local_bs, shuffle=False)
        hos_test_loader = DataLoader(test_eICUDataset, batch_size=args.local_bs, shuffle=False)

        # 添加加载器到列表中
        train_loader.append(hos_train_loader)
        val_loader.append(hos_val_loader)
        test_loader.append(hos_test_loader)

    # 获取输入特征大小
    input_size = [hosDataset.F, hosDataset.D, hosDataset.no_flat_features]

    return train_loader, val_loader, test_loader, global_test_loader, id_list, input_size


from options import args_parser

if __name__ == '__main__':
    args = args_parser()
    args.dataset = 'eICU_hos'
    trains, vals, tests, gl, ids, input_sz = get_dataset(args)
    for val in vals:
        print('*****' * 8)
        for x, y in val:
            print(f"timeseries:{x['times']}\ndiags:{x['diags']}\nflat:{x['flat']}\nlos_label:{y['los']}\nmort_label:{y['mort']}")



