import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class MetaModule(nn.Module):
    def params(self):
        for name, param in self.named_params(self):
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p

    def update_params(self, lr_inner, first_order=False, source_params=None, detach=False):
        if source_params is not None:
            for tgt, src in zip(self.named_params(self), source_params):
                name_t, param_t = tgt
                # name_s, param_s = src
                # grad = param_s.grad
                # name_s, param_s = src
                grad = src
                if first_order:
                    grad = to_var(grad.detach().data)
                tmp = param_t - lr_inner * grad
                self.set_param(self, name_t, tmp)
        else:

            for name, param in self.named_params(self):
                if not detach:
                    grad = param.grad
                    if first_order:
                        grad = to_var(grad.detach().data)
                    tmp = param - lr_inner * grad
                    self.set_param(self, name, tmp)
                else:
                    param = param.detach_()
                    self.set_param(self, name, param)

    def set_param(self, curr_mod, name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in curr_mod.named_children():
                if module_name == name:
                    self.set_param(mod, rest, param)
                    break
        else:
            setattr(curr_mod, name, param)

    def detach_params(self):
        for name, param in self.named_params(self):
            self.set_param(self, name, param.detach())

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            if not same_var:
                param = to_var(param.data.clone(), requires_grad=True)
            self.set_param(name, param)


class MetaLinear(MetaModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        ignore = nn.Linear(*args, **kwargs)

        self.register_buffer('weight', to_var(ignore.weight.data, requires_grad=True))
        self.register_buffer('bias', to_var(ignore.bias.data, requires_grad=True))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def named_leaves(self):
        return [('weight', self.weight), ('bias', self.bias)]

class VNet(MetaModule):
    def __init__(self, input, hidden, output):
        super(VNet, self).__init__()
        self.linear1 = MetaLinear(input, hidden)
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = MetaLinear(hidden, output)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        out = self.linear2(x)
        return F.sigmoid(out)

class MetaMortality(MetaModule):  # 继承MetaModule
    def __init__(self, input_size, last_linear_size,args):
        super(MetaMortality, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.point_mort = MetaLinear(in_features=input_size, out_features=last_linear_size)  # 使用MetaLinear
        self.point_final_mort = MetaLinear(in_features=last_linear_size, out_features=1)  # 使用MetaLinear
        self.args = args
    def forward(self, combined_features):
        x = self.relu(self.point_mort(combined_features))  # 使用MetaLinear层
        mort_predictions = self.sigmoid(self.point_final_mort(x))  # 输出死亡预测
        return mort_predictions


class MetaLOS(MetaModule):  # 继承MetaModule
    def __init__(self, input_size, last_linear_size,args):
        super(MetaLOS, self).__init__()
        self.relu = nn.ReLU()
        self.hardtanh = nn.Hardtanh(min_val=1 / 48, max_val=8)  # 限制输出范围
        self.point_los = MetaLinear(in_features=128, out_features=10)  # 使用MetaLinear
        self.point_final_los = MetaLinear(in_features=10, out_features=1)  # 使用MetaLinear
        self.args = args

    def forward(self, los_features):
        x = self.relu(self.point_los(los_features))
        los_predictions = self.hardtanh(self.point_final_los(x))
        return los_predictions
