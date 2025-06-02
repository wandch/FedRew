import copy

import torch
import torch.nn as nn

class BaseHeadSplit(nn.Module):
    def __init__(self, base, head):
        super(BaseHeadSplit, self).__init__()
        self.base = base
        self.head = head
    def forward(self, x):
        out = self.base(x)
        out = self.head(out)
        return out

class GRU(nn.Module):
    def __init__(self, args,input_size, hidden_size, num_layers, bidirectional=True, batch_first=True):
        super(GRU, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                          bidirectional=bidirectional, batch_first=batch_first)

        self.args = args
        self.relu = nn.ReLU()
        self.combine_nn = nn.Linear(in_features=128*48, out_features=128)

    def forward(self, X):
        self.gru.flatten_parameters()
        output, hidden = self.gru(X)
        return output


class DiagEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(DiagEncoder, self).__init__()
        self.diagnosis_encoder = nn.Sequential(
            nn.Linear(input_size, 80),   # 将输入压缩到80维
            nn.ReLU(),                   # 激活函数
            nn.Linear(80, output_size)   # 输出指定维度
        )

    def forward(self, diagnoses):
        diagnoses_enc = self.diagnosis_encoder(diagnoses)
        return diagnoses_enc

class ToMortality(nn.Module):
    def __init__(self, input_size, last_linear_size):
        super(ToMortality, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.point_mort = nn.Linear(in_features=input_size, out_features=last_linear_size)
        self.point_final_mort = nn.Linear(in_features=last_linear_size, out_features=1)  # 输出单值

    def forward(self, combined_features):
        x = self.relu(self.point_mort(combined_features))
        mort_predictions = self.sigmoid(self.point_final_mort(x))  # 将输出映射到[0,1]区间
        return mort_predictions


class ToLOS(nn.Module):
    def __init__(self, input_size, last_linear_size):
        super(ToLOS, self).__init__()
        self.relu = nn.ReLU()
        self.hardtanh = nn.Hardtanh(min_val=1 / 48, max_val=8)  # 将输出限制在合理范围内\
        self.point_los = nn.Linear(in_features=input_size, out_features=last_linear_size)
        self.point_final_los = nn.Linear(in_features=last_linear_size, out_features=1)

    def forward(self, los_features):

        x = self.relu(self.point_los(los_features))
        los_predictions = self.hardtanh(self.point_final_los(x))
        return los_predictions


class BaseGRU(nn.Module):
    def __init__(self,args, F=None, D=None, no_flat_features=None):
        super(BaseGRU, self).__init__()

        self.diagnosis_size = 10  # 输出诊断编码维度
        self.F = F  # 时间序列特征数
        self.D = D  # 诊断特征数
        self.no_flat_features = no_flat_features
        self.args = args
        self.task = args.task
        self.relu = nn.ReLU()
        self.bidirectional = False
        self.n_layers = 2
        self.hidden_size = 128
        self.last_linear_size = 10
        self.gru_outsize = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.diag_encoder = DiagEncoder(input_size=self.D, output_size=self.diagnosis_size)
        self.gru_module = GRU(self.args,input_size=(self.F + self.diagnosis_size + self.no_flat_features), hidden_size=self.hidden_size,
                              num_layers=self.n_layers, bidirectional=self.bidirectional)
        self.input_size = self.gru_outsize
        self.time_before_pred = 47
        self.avg_pool = nn.AvgPool1d(kernel_size=5)
        self.classifier = ToLOS(self.input_size, self.last_linear_size) if self.task == 'LoS' else ToMortality(self.input_size, self.last_linear_size)

    def pool_last_five_hours(self, X_final):
        if X_final.size(1) < 5:
            raise ValueError("输入数据的时间步长小于 5，无法进行最后 5 个时间步的池化操作。")
        last_five_hours = X_final[:, -5:, :]
        last_five_hours_reshaped = last_five_hours.permute(0, 2, 1)
        return self.avg_pool(last_five_hours_reshaped).squeeze(-1)


    def forward(self, X):
        times, diagnoses, flat = X['times'], X['diags'], X['flat']
        B, T, _ = times.shape
        diagnoses_enc = self.relu(self.diag_encoder(diagnoses))
        flat_repeated = flat.unsqueeze(1).repeat(1, T, 1)
        diagnoses_enc_repeated = diagnoses_enc.unsqueeze(1).repeat(1, T, 1)
        combined_features = torch.cat((times,flat_repeated, diagnoses_enc_repeated), dim=-1)
        gru_output = self.gru_module(combined_features)
        X_final = self.relu(gru_output)
        if self.task == 'LoS':
            los_features = self.pool_last_five_hours(X_final)
            los_predictions = self.classifier(los_features)
            return los_predictions,los_features
        else:
            mort_features = X_final[:, -1, :]  # 最后时间步的特征用于死亡率预测
            mort_predictions = self.classifier(mort_features)
            return  mort_predictions, mort_features

if __name__=='__main__':
    pass
