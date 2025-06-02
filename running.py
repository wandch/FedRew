import torch
import tools
import numpy as np
import copy
from tools import average_weights_weighted,average_pFedme_weights_weighted,Dwa_weighted,average_weights_weighted_margin
def one_round_training(rule):
    Train_Round = {
        'FedAvg': train_round_FedAvg,
        'FedRew': train_round_FedRew,
    }
    return Train_Round[rule]


def train_round_FedAvg(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd + 1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2, results, agg_weight = [], [], [], [], []
    local_epoch = args.local_epoch
    # 本地模型训练
    for idx in idx_users:
        local_client = local_clients[idx]
        agg_weight.append(local_client.agg_weight)
        w, loss1 = local_client.local_training(local_epoch=local_epoch, round=rnd)
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(loss1)

    # 模型聚和
    agg_weight = torch.stack(agg_weight).to(args.device)
    global_weight = average_weights_weighted(local_weights, agg_weight)
    global_model.load_state_dict(global_weight)

    # 加载并且测试全局模型
    for idx in idx_users:
        local_client = local_clients[idx]
        local_client.update_local_model(global_weight=global_weight)
        result, loss2 = local_client.local_valid(round=rnd)
        local_losses2.append(loss2)
        results.append(result)

    agg_weight_np = agg_weight.cpu().numpy()
    avg_train_loss = np.average(local_losses1, weights=agg_weight_np)
    avg_val_loss = np.average(local_losses2, weights=agg_weight_np)
    avg_results = np.average(results, axis=0, weights=agg_weight_np).round(4).tolist()

    return avg_train_loss, avg_val_loss, avg_results

def train_round_FedRew(args, global_model, local_clients, rnd, **kwargs):
    print(f'\n---- Global Communication Round : {rnd+1} ----')
    num_users = args.num_users
    m = max(int(args.frac * num_users), 1)
    if (rnd >= args.epochs):
        m = num_users
    idx_users = np.random.choice(range(num_users), m, replace=False)
    idx_users = sorted(idx_users)
    local_weights, local_losses1, local_losses2, results, agg_weight,alpha_weight = [], [], [], [],[],[]

    global_weight = global_model.state_dict()
    for idx in idx_users:
        local_client = local_clients[idx]
        local_epoch = args.local_epoch
        agg_weight.append(local_client.agg_weight)
        local_client.update_local_model_with_premodel(global_weight=global_weight,u=args.lambda_u)
        w, loss1, loss2, result = local_client.local_training(local_epoch=local_epoch, round=rnd)
        alpha_weight.append(torch.tensor(local_client.alpha_history[-1]))
        local_weights.append(copy.deepcopy(w))
        local_losses1.append(loss1)
        local_losses2.append(loss2)
        results.append(result)

    agg_weight = torch.stack(agg_weight).to(args.device)

    alpha_weight = torch.stack(alpha_weight).to(args.device)
    global_weight = average_weights_weighted(local_weights, alpha_weight)

    global_model.load_state_dict(global_weight)
    agg_weight_np = agg_weight.cpu().numpy()
    avg_train_loss = np.average(local_losses1, weights=agg_weight_np)
    avg_val_loss = np.average(local_losses2, weights=agg_weight_np)
    avg_results = np.average(results, axis=0, weights=agg_weight_np).round(4).tolist()
  
    return avg_train_loss, avg_val_loss, avg_results

