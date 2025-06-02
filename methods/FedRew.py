import copy

import numpy as np
import torch
from torch import nn

from .base import BaseFedLearning
from models.meta_model import MetaLOS,MetaMortality
from torch.utils.data import DataLoader
class eICU_LocalUpdate_FedRew(BaseFedLearning):
    def __init__(self, idx, args, train_set,val_set, test_set, model):
        super(eICU_LocalUpdate_FedRew, self).__init__(idx, args, train_set,val_set, test_set, model)

        self.rew_epoch = self.args.rew_epoch

        self.meta_model = MetaLOS(self.local_model.input_size,self.local_model.last_linear_size,self.args) \
            if self.args.task == 'LoS' else MetaMortality(self.local_model.input_size, self.local_model.last_linear_size,self.args)

        self.p_model = copy.deepcopy(model)
        self.alpha_history = []
        self.global_weight = None

    def update_local_model_with_premodel(self, global_weight,u):
        """
        更新本地模型的权重为全局模型的权重，u为0时等同于FedAvg
        """
        self.global_weight = copy.deepcopy(global_weight)
        lw = copy.deepcopy(self.local_model.state_dict())
        temp_weight = copy.deepcopy(lw)
        for key in lw.keys():
            temp_weight[key] = torch.zeros_like(lw[key])
            temp_weight[key] = global_weight[key] * (1 -u) + lw[key] * u

        self.local_model.load_state_dict(temp_weight)
    def local_training(self, local_epoch, round=0):
        model = self.local_model
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr, weight_decay=self.weight_decay,
                                    momentum=self.momentum)
        iter_loss = []
        # Step 1: 本地训练阶段
        for ep in range(local_epoch):
            for x, y in self.train_data:
                x = {k: v.to(self.device) for k, v in x.items()}
                y = {k: v.to(self.device) for k, v in y.items()}

                # 前向传播和损失计算
                y_pred, features = model(x)
                optimizer.zero_grad()
                loss = self.loss(y_pred, y)
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                iter_loss.append(loss.item())
        loss_value = sum(iter_loss) / len(iter_loss)
        if round <0:
            return model.state_dict(), loss_value


        # 调用重新加权训练
        # 创建元模型
        for meta_param, param in zip(self.meta_model.params(), model.classifier.parameters()):
            meta_param.data = param.data.clone()
        # 更新特征提取器
        self.p_model.load_state_dict(model.state_dict())
        rew_classifier = self.reweighted_training(self.meta_model, self.train_data, self.val_data,rew_epochs=self.rew_epoch)
        # 更新分类器
        for p_param, rew_param in zip(self.p_model.classifier.parameters(), rew_classifier.params()):
            p_param.data = rew_param.data.clone()

        self.calculate_alpha()
        results, val_loss_value = self.local_valid(round=round)
        return model.state_dict(), loss_value, val_loss_value, results


    def calculate_alpha(self):
        alpha = 0
        for x, y in self.train_data:
            # 确保数据和模型都在同一设备上
            x = {k: v.to(self.device) for k, v in x.items()}
            y = {k: v.to(self.device) for k, v in y.items()}

            # 获取meta模型的预测结果
            meta_y, meta_features = self.local_model(x)
            meta_loss = self.loss(meta_y, y, reduction='mean')  # 计算所有数据的均值损失


            # 获取其他模型的预测结果
            p_y, p_features = self.p_model(x)
            p_loss = self.loss(p_y, y, reduction='mean')  # 计算所有数据的均值损失

            # 计算聚合系数
            alpha += 2 * meta_loss.cpu().detach().numpy() / (
                    meta_loss.cpu().detach().numpy() + p_loss.cpu().detach().numpy())


        # 调试输出
        print(self.idx, ' alpha:',  alpha)
        self.alpha_history.append(alpha)

    def reweighted_training(self, model, train_loader, val_loader, rew_epochs=5):
        model = model.to(self.device)
        model_optimizer = torch.optim.SGD(model.params(), self.args.lr_classifier)

        for epoch in range(rew_epochs):
            # ----------------
            # 1. 主模型训练阶段（Inner Loop）
            # ----------------
            model.train()
            for batch_idx, (x, y) in enumerate(train_loader):
                x = {k: v.to(self.device) for k, v in x.items()}
                y = {k: v.to(self.device) for k, v in y.items()}

                # ----------------
                # 2. 元学习更新（Meta-Model）
                # ----------------
                # 创建元模型，复制主模型的参数
                meta_model = copy.deepcopy(model).to(self.device)

                # 前向传播，计算损失,训练完的本地模型
                tp, features = self.local_model(x)
                y_pred = meta_model(features)
                loss = self.loss(y_pred, y, reduction='none')

                batch_size = len(loss)
                epsilon_batch = torch.ones(batch_size, device=self.device, requires_grad=True) / batch_size

                meta_loss = torch.sum(loss * epsilon_batch.view(-1, 1))

                meta_model.zero_grad()

                # Line 6 perform a parameter update
                grads = torch.autograd.grad(meta_loss, (meta_model.params()), create_graph=True, retain_graph=True)

                # 更新元模型参数
                meta_model.update_params(lr_inner=self.args.lr, source_params=grads)

                # ----------------
                # 3. 计算与 epsilon_batch 相关的梯度
                # ----------------

                val_x, val_y = next(iter(val_loader))  # 使用迭代器取一批数据
                val_x = {k: v.to(self.device) for k, v in val_x.items()}
                val_y = {k: v.to(self.device) for k, v in val_y.items()}
                vp, val_features = self.local_model(val_x)
                val_y_pred = meta_model(val_features)
                val_loss = self.loss(val_y_pred, val_y)

                # 获取 epsilon_batch 的梯度
                grad_eps_batch = torch.autograd.grad(val_loss, epsilon_batch, only_inputs=True, retain_graph=True)[0]

                # ----------------
                # 更新权重（w_tilde）
                # ----------------
                w_tilde = torch.clamp(-grad_eps_batch, min=0)
                norm_c = torch.sum(w_tilde)

                if norm_c != 0:
                    w = w_tilde / norm_c
                else:
                    w = w_tilde
                # if epoch == rew_epochs - 1:
                #     print(w)

                # ----------------
                # 4. 在主模型中计算加权损失并进行优化
                # ----------------
                yp, features = self.local_model(x)
                y_pred = model(features)
                loss = self.loss(y_pred, y, reduction='none')

                weighted_loss = torch.sum(loss * w)

                model_optimizer.zero_grad()
                weighted_loss.backward()
                model_optimizer.step()
        return model

