# coding: utf-8
import copy
import json
import os
import numpy as np
import torch
from torch import nn
from collections import defaultdict
import tools
from metrics import print_metrics_mortality, print_metrics_regression


# 基础联邦学习类
class BaseFedLearning(object):
    def __init__(self, idx, args, train_set, val_set, test_set, model):
        """
        初始化方法，设置训练、验证和测试数据集，以及模型和其他参数。
        """
        self.idx = idx  # 当前客户端的索引
        self.args = args  # 参数对象，包含学习率、批次大小等信息
        self.train_data = train_set  # 训练数据集
        self.test_data = test_set  # 测试数据集
        self.val_data = val_set  # 验证数据集
        self.device = args.device  # 设备（GPU/CPU）
        self.local_model = model  # 本地模型
        self.agg_weight = self.aggregate_weight()  # 聚合权重

        # 损失函数
        self.mse_loss = nn.MSELoss()  # 均方误差
        self.bce_loss = nn.BCELoss()  # 二分类交叉熵损失
        self.lr_classifier = args.lr_classifier  # 分类器学习率
        self.task = args.task  # 任务类型 ('LoS' 或 'mortality')

        # 根据任务类型选择对应的分类器
        self.classifier = self.local_model.classifier
        self.w_local_keys = ['classifier.'+item for item in self.classifier.state_dict().keys()]
        # 本地批次大小等参数
        self.local_bs =args.local_bs
        self.save_path = args.save_path  # 保存路径
        self.save_results_csv =args.save_results_csv  # 保存结果的CSV文件
        self.weight_decay = args.weight_decay  # 权重衰减
        self.momentum = args.momentum  # 动量
        self.p_model = None
        self.best_auc  = 0
        self.best_mes  = 10000
        self.best_weight = None


    def aggregate_weight(self):  # 聚合数据集的大小作为权重
        data_size = len(self.train_data.dataset)  # 获取训练数据集的大小
        w = torch.tensor(data_size).to(self.device)  # 转为张量
        return w

    def update_local_model(self, global_weight):
        """
        更新本地模型的权重为全局模型的权重。
        """
        self.local_model.load_state_dict(global_weight)

    def update_base_model(self, global_weight):
        """
        更新基本模型，除了分类器部分，其他部分都使用全局权重。
        """
        local_weight = self.local_model.state_dict()  # 获取本地模型的权重
        w_local_keys = self.w_local_keys  # 分类器部分的权重
        for k in local_weight.keys():
            if k not in w_local_keys:
                local_weight[k] = global_weight[k]  # 更新模型权重
        self.local_model.load_state_dict(local_weight)

    def update_local_classifier(self, global_weight):
        """
        更新本地分类器的权重。
        """
        local_weight = self.local_model.state_dict()
        w_local_keys = self.w_local_keys
        for k in local_weight.keys():
            if k in w_local_keys:
                local_weight[k] = global_weight[k]
        self.local_model.load_state_dict(local_weight)

    def local_test(self, round):
        """
        在本地进行测试。
        """
        model = self.local_model
        model.load_state_dict(self.best_weight)  # 加载最佳模型权重
        test_loader = self.test_data  # 测试数据集
        center = self.idx  # 当前客户端索引
        test_loss = []  # 存储测试损失
        model.eval()  # 设置模型为评估模式
        test_y_hat_los = np.array([])  # 存储预测的住院时长
        test_y_los = np.array([])  # 存储真实的住院时长
        test_y_hat_mort = np.array([])  # 存储预测的死亡率
        test_y_mort = np.array([])  # 存储真实的死亡率

        with torch.no_grad():  # 不计算梯度
            for x, y in test_loader:
                x = {k: v.to(self.device) for k, v in x.items()}
                y = {k: v.to(self.device) for k, v in y.items()}
                y_pred, features = model(x)  # 获取模型预测
                loss = self.loss(y_pred, y)  # 计算损失

                test_loss.append(loss.item())
                if self.task == 'LoS':
                    test_y_hat_los = np.append(test_y_hat_los,  y_pred.flatten().detach().cpu().numpy())
                    test_y_los = np.append(test_y_los, y['los'].flatten().detach().cpu().numpy())
                if self.task == 'mortality':
                    test_y_hat_mort = np.append(test_y_hat_mort,  y_pred.flatten().detach().cpu().numpy())
                    test_y_mort = np.append(test_y_mort, y['mort'].flatten().detach().cpu().numpy())

            final_metrics = []

            # 计算并记录指标
            if self.task == 'LoS':
                # 将 test_y_los 和 test_y_hat_los 转换为每 43 个元素一组的二级列表
                test_y_los_2d = [test_y_los[i:i + 43].tolist() for i in range(0, len(test_y_los), 43)]
                test_y_hat_los_2d = [test_y_hat_los[i:i + 43].tolist() for i in range(0, len(test_y_hat_los), 43)]

                final_metrics_list = print_metrics_regression(test_y_los, test_y_hat_los, verbose=0)
                results_dict = {'True label': test_y_los_2d, 'predict label': test_y_hat_los_2d}
                filename = os.path.join(self.save_path, f'client_{self.idx}_round_{round}_LoS_all_label.json')
                with open(filename, 'w') as file:
                    json.dump(results_dict, file, indent=4)

            if self.task == 'mortality':
                final_metrics_list = print_metrics_mortality(test_y_mort, test_y_hat_mort, verbose=0)
                results_dict = {'True label': test_y_mort.tolist(), 'predict label': test_y_hat_mort.tolist()}
                filename = os.path.join(self.save_path, f'client_{self.idx}_round_{round}_mortality_all_label.json')
                with open(filename, 'w') as file:
                    json.dump(results_dict, file, indent=4)

            for metric in final_metrics_list:
                final_metrics.append(metric)

            if round >= 0:
                # 将结果保存到 CSV 文件
                self.save_result(round, center, final_metrics)

        return final_metrics

    def local_valid(self, round):
        """
        在本地进行验证。
        """
        model = self.local_model if self.p_model is None else self.p_model
        val_loader = self.val_data
        center = self.idx
        val_loss = []  # 存储验证损失
        model.eval()  # 设置模型为评估模式
        val_y_hat_los = np.array([])  # 存储预测的住院时长
        val_y_los = np.array([])  # 存储真实的住院时长
        val_y_hat_mort = np.array([])  # 存储预测的死亡率
        val_y_mort = np.array([])  # 存储真实的死亡率

        with torch.no_grad():  # 不计算梯度
            for x, y in val_loader:
                x = {k: v.to(self.device) for k, v in x.items()}
                y = {k: v.to(self.device) for k, v in y.items()}
                y_pred, features = model(x)
                loss = self.loss(y_pred, y)
                val_loss.append(loss.item())

                if self.task == 'LoS':
                    val_y_hat_los = np.append(val_y_hat_los, y_pred.flatten().detach().cpu().numpy())
                    val_y_los = np.append(val_y_los, y['los'].flatten().detach().cpu().numpy())
                if self.task == 'mortality':
                    val_y_hat_mort = np.append(val_y_hat_mort, y_pred.flatten().detach().cpu().numpy())
                    val_y_mort = np.append(val_y_mort, y['mort'].flatten().detach().cpu().numpy())

            loss_value = sum(val_loss) / len(val_loss)  # 平均验证损失

            final_metrics = []
            if self.task == 'LoS':
                final_metrics_list = print_metrics_regression(val_y_los, val_y_hat_los, verbose=0)
            if self.task == 'mortality':
                final_metrics_list = print_metrics_mortality(val_y_mort, val_y_hat_mort, verbose=0)
            for metric in final_metrics_list:
                final_metrics.append(metric)

            if self.task == 'LoS':
                curr_mse = final_metrics[-1]
                if curr_mse < self.best_mes:
                    self.best_mes = curr_mse
                    self.best_weight = copy.deepcopy(model.state_dict())

            elif self.task == 'mortality':
                curr_auc = final_metrics[3]
                if curr_auc > self.best_auc:
                    self.best_auc = curr_auc
                    self.best_weight =copy.deepcopy(model.state_dict())

            if round >= 0:
                self.save_result(round, center, final_metrics)

        return final_metrics, loss_value


    def local_training(self, local_epoch, round=0):
        """
        在本地进行训练。
        """
        model = self.local_model
        iter_loss = []
        model.zero_grad()
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, weight_decay=self.weight_decay,
                                    momentum=self.momentum)

        if local_epoch > 0:
            for ep in range(local_epoch):
                data_loader = iter(self.train_data)
                iter_num = len(data_loader)
                for it in range(iter_num):
                    x, y = next(data_loader)
                    x = {k: v.to(self.device) for k, v in x.items()}
                    y = {k: v.to(self.device) for k, v in y.items()}
                    y_pred, features = model(x)
                    optimizer.zero_grad()
                    loss = self.loss(y_pred, y)
                    loss.backward()
                    optimizer.step()
                    iter_loss.append(loss.item())
        loss_value = sum(iter_loss) / len(iter_loss)
        return model.state_dict(), loss_value

    def loss(self, y_pred, y, reduction="mean"):
        mse_loss = nn.MSELoss(reduction=reduction)
        bce_loss = nn.BCELoss(reduction=reduction)
        y_mort = torch.unsqueeze(y['mort'], 1).float()
        if self.task == 'mortality':
            loss = bce_loss (y_pred, y_mort)  # 对死亡率任务使用二分类交叉熵
        else:
            loss = mse_loss(y_pred, y['los'])  # 对住院时长任务使用均方误差
        return loss

    def save_result(self, round, center, final_metrics):
        """
        保存训练和测试的结果到 CSV 文件。
        """
        if self.task == 'mortality':
            file_path = self.save_path + '/Mort_results.csv'
            file_exists = os.path.exists(file_path)
            with open(file_path, 'a') as f:
                if not file_exists:
                    header = 'round,center,acc,prec0,rec0,auroc,auprc,f1macro\n'
                    f.write(header)
                data_to_write = [round, center] + final_metrics
                f.write(','.join(map(str, data_to_write)) + '\n')
        if self.task == 'LoS':
            file_path = self.save_path + '/Los_results.csv'
            file_exists = os.path.exists(file_path)
            with open(file_path, 'a') as f:
                if not file_exists:
                    header = 'round,center,mad,mse,mape,msle,r2,rmse\n'
                    f.write(header)
                data_to_write = [round, center] + final_metrics
                f.write(','.join(map(str, data_to_write)) + '\n')


