import copy
import os
import torch
import types
import math
import numpy as np




def get_protos(protos):
    """
    Returns the average of the feature embeddings of samples from per-class.
    """
    protos_mean = {}
    for [label, proto_list] in protos.items():
        proto = 0 * proto_list[0]
        for i in proto_list:
            proto += i
        protos_mean[label] = proto / len(proto_list)

    return protos_mean


def protos_aggregation(local_protos_list, local_sizes_list):
    agg_protos_label = {}
    agg_sizes_label = {}
    for idx in range(len(local_protos_list)):
        local_protos = local_protos_list[idx]
        local_sizes = local_sizes_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
                agg_sizes_label[label].append(local_sizes[label])
            else:
                agg_protos_label[label] = [local_protos[label]]
                agg_sizes_label[label] = [local_sizes[label]]

    for [label, proto_list] in agg_protos_label.items():
        sizes_list = agg_sizes_label[label]
        proto = 0 * proto_list[0]
        for i in range(len(proto_list)):
            proto += sizes_list[i] * proto_list[i]
        agg_protos_label[label] = proto / sum(sizes_list)

    return agg_protos_label


def average_weights_weighted(w, avg_weight):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    weight = avg_weight.clone().detach()
    # 权重比例
    agg_w = weight / (weight.sum(dim=0))
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key])
        for i in range(len(w)):
             w_avg[key] += agg_w[i] * w[i][key]
        # w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

def average_weights_weighted_margin(w, margin_weight):
    """
    根据 margin_weight 生成新的权重，然后计算加权平均权重。
    """
    # 对 margin_weight 进行处理，生成新的权重
    new_avg_weight = []
    for weight in margin_weight:
        new_weight = math.exp(-weight)
        new_avg_weight.append(new_weight)
    new_avg_weight = torch.tensor(new_avg_weight)

    # 深拷贝权重列表中的第一个元素作为初始的平均权重
    w_avg = copy.deepcopy(w[0])
    # 克隆并分离平均权重，避免后续操作影响原始数据
    weight = new_avg_weight.clone().detach()
    # 计算每个权重的比例
    agg_w = weight / (weight.sum(dim=0))

    # 遍历 w_avg 中的每个键
    for key in w_avg.keys():
        # 初始化一个全零张量
        w_avg[key] = torch.zeros_like(w_avg[key])
        # 根据权重比例对 w 中所有元素对应键的值进行加权求和
        for i in range(len(w)):
            w_avg[key] += agg_w[i] * w[i][key]
    return w_avg

def average_pFedme_weights_weighted(w, avg_weight,beta,previous_model):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    w_pfedme = copy.deepcopy(w[0])
    weight = avg_weight.clone().detach()
    #权重比例
    agg_w = weight/(weight.sum(dim=0))
    for key in w_avg.keys():
        w_avg[key] = torch.zeros_like(w_avg[key])
        w_pfedme[key] = torch.zeros_like(w_avg[key])
        for i in range(len(w)):
             w_avg[key] += agg_w[i]*w[i][key]
        # w_avg[key] = torch.div(w_avg[key], len(w))
        w_pfedme[key] = w_avg[key] * beta + previous_model[key] * (1 - beta)
    return w_pfedme


def agg_classifier_weighted_p(w, avg_weight, keys, idx):
    """
    Returns the average of the weights.
    """
    w_0 = copy.deepcopy(w[idx])
    for key in keys:
        w_0[key] = torch.zeros_like(w_0[key])
    wc = 0
    for i in range(len(w)):
        wi = avg_weight[i]
        wc += wi
        for key in keys:
            w_0[key] += wi*w[i][key]
    for key in keys:
        w_0[key] = torch.div(w_0[key], wc)
    return w_0

# --------------------------------------------------------------------- #
# Gradient access

def grad_of(tensor):
    """ Get the gradient of a given tensor, make it zero if missing.
    Args:
    tensor Given instance of/deriving from Tensor
    Returns:
    Gradient for the given tensor
    """
    # Get the current gradient
    grad = tensor.grad
    if grad is not None:
        return grad
    # Make and set a zero-gradient
    grad = torch.zeros_like(tensor)
    tensor.grad = grad
    return grad


def grads_of(tensors):
    """ Iterate of the gradients of the given tensors, make zero gradients if missing.
    Args:
    tensors Generator of/iterable on instances of/deriving from Tensor
    Returns:
    Generator of the gradients of the given tensors, in emitted order
    """
    return (grad_of(tensor) for tensor in tensors)

# ---------------------------------------------------------------------------- #
# "Flatten" and "relink" operations

def relink(tensors, common):
    """ "Relink" the tensors of class (deriving from) Tensor by making them point to another contiguous segment of memory.
    Args:
    tensors Generator of/iterable on instances of/deriving from Tensor, all with the same dtype
    common  Flat tensor of sufficient size to use as underlying storage, with the same dtype as the given tensors
    Returns:
    Given common tensor
    """
    # Convert to tuple if generator
    if isinstance(tensors, types.GeneratorType):
        tensors = tuple(tensors)
    # Relink each given tensor to its segment on the common one
    pos = 0
    for tensor in tensors:
        npos = pos + tensor.numel()
        tensor.data = common[pos:npos].view(*tensor.shape)
        pos = npos
    # Finalize and return
    common.linked_tensors = tensors
    return common


def flatten(tensors):
    # Convert to tuple if generator
    if isinstance(tensors, types.GeneratorType):
        tensors = tuple(tensors)
    # Common tensor instantiation and reuse
    common = torch.cat(tuple(tensor.view(-1) for tensor in tensors))
    # Return common tensor
    return relink(tensors, common)

# ---------------------------------------------------------------------------- #

def get_gradient(model):
    gradient = flatten(grads_of(model.parameters()))
    return gradient

def set_gradient(model, gradient):
    grad_old = get_gradient(model)
    grad_old.copy_(gradient)

def get_gradient_values(model):
    gradient = torch.cat([torch.reshape(param.grad, (-1,)) for param in model.parameters()]).clone().detach()
    return gradient

def set_gradient_values(model, gradient):
    cur_pos = 0
    for param in model.parameters():
        param.grad = torch.reshape(torch.narrow(gradient, 0, cur_pos, param.nelement()), param.size()).clone().detach()
        cur_pos = cur_pos + param.nelement()

def get_parameter_values(model):
    parameter = torch.cat([torch.reshape(param.data, (-1,)) for param in model.parameters()]).clone().detach()
    return parameter

def set_parameter_values(model, parameter):
    cur_pos = 0
    for param in model.parameters():
        param.data = torch.reshape(torch.narrow(parameter, 0, cur_pos, param.nelement()), param.size()).clone().detach()
        cur_pos = cur_pos + param.nelement()
# ---------------------------------------------------------------------------- #


def create_folder(parent_path, folder):
    if not parent_path.endswith('/'):
        parent_path += '/'
    folder_path = parent_path + folder
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


def save_to_csv(save_dir, data, path, header=None):
    """
        Saves a numpy array to csv in the experiment save dir

        Args:
            data: The array to be stored as a save file
            path: sub path in the save folder (or simply filename)
    """

    folder_path = create_folder(save_dir, os.path.dirname(path))
    file_path = folder_path + '/' + os.path.basename(path)
    if not file_path.endswith('.csv'):
        file_path += '.csv'
    np.savetxt(file_path, data, delimiter=',', header=header, comments='')
    return
#模型分类器初始化



def model_test(args, local_clients):
    final_met_txt_file = os.path.join(args.save_path, 'final_metric.txt')
    results, agg_weight, ids = [], [], []
    for idx in range(args.num_users):
        local_client = local_clients[idx]
        agg_weight.append(local_client.agg_weight.cpu().numpy())
        result= local_client.local_test(round=args.epochs)
        results.append(result)
        ids.append(local_client.idx)

    agg_weight_np = np.array(agg_weight)
    avg_results = np.average(results, axis=0, weights=agg_weight_np).round(4).tolist()

    results_dict = {'final_metrics': avg_results}
    with open(final_met_txt_file, 'w') as file:
        file.write(str(results_dict))



    print('Final metric: {}'.format(avg_results))

    for center, metric in zip(ids, results):
        if args.task == 'mortality':
            file_path = os.path.join(args.save_path, 'Final_Mort_results.csv')
            file_exists = os.path.exists(file_path)
            with open(file_path, 'a') as f:
                if not file_exists:
                    header = 'center,acc,prec0,rec0,auroc,auprc,f1macro\n'
                    f.write(header)
                data_to_write = [center] + metric
                f.write(','.join(map(str, data_to_write)) + '\n')

        if args.task == 'LoS':
            file_path = os.path.join(args.save_path, 'Final_Los_results.csv')
            file_exists = os.path.exists(file_path)
            with open(file_path, 'a') as f:
                if not file_exists:
                    header = 'center,mad,mse,mape,msle,r2,rmse\n'
                    f.write(header)
                data_to_write = [center] + metric
                f.write(','.join(map(str, data_to_write)) + '\n')


def calculate_weighted_combination_top_k(wi, Wj, k):
    """
    计算基于前 k 个最大权重的模型参数加权组合
    参数:
    wi (torch.Tensor): 目标模型参数
    Wj (list of torch.Tensor): 其他模型参数列表
    k (int): 选取的前 k 个最大权重的数量
    返回:
    torch.Tensor: 加权组合后的模型参数
    """
    # 检查输入参数的有效性
    if k > len(Wj):
        raise ValueError(f"k ({k}) 不能大于 Wj 的长度 ({len(Wj)})")

    eps = 1e-8  # 用于数值稳定性的小常数

    # 计算分子部分: ||wi - wj||^{-2}
    numerator = torch.tensor([torch.norm(wi - wj + eps).pow(-2) for wj in Wj])

    # 计算分母部分: sum ||wi - wk||^{-2}
    denominator = numerator.sum()

    # 计算初始的概率分布 p_{i,j}
    p_i_j = numerator / denominator

    # 获取前 k 个最大的概率的索引
    top_k_indices = torch.topk(p_i_j, k).indices

    # 选取前 k 个对应的概率和模型参数
    p_top_k = p_i_j[top_k_indices]
    W_top_k = [Wj[i] for i in top_k_indices]

    # 重新对前 k 个概率进行归一化
    p_top_k_normalized = p_top_k / p_top_k.sum()

    # 使用归一化后的概率对前 k 个模型参数进行加权求和
    weighted_combination = sum(p * wj for p, wj in zip(p_top_k_normalized, W_top_k))

    return weighted_combination

def Dwa_weighted(wi, Wj, k=10):
    """
    Returns the Dwa of the weights.
    """
    Dwa_w = copy.deepcopy(wi)
    for key in Dwa_w.keys():
        Wj_key = [temp[key] for temp in Wj]
        Dwa_w[key] = calculate_weighted_combination_top_k(wi[key],Wj_key ,k)
    return Dwa_w

if __name__ == '__main__':
    # 示例使用
    wi = torch.tensor([1.0, 2.0, 3.0])  # 假设是第 i 个模型的参数
    Wj = [torch.tensor([1, 2, 3]), torch.tensor([0.5, 1.5, 2.5])]  # 模型参数列表

    weighted_params = calculate_weighted_combination_top_k(wi,Wj,1 )
    print(weighted_params)
