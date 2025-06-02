import json
import os.path

import numpy as np
import torch
from data_loader import get_dataset
from running import one_round_training
from methods import local_update
from models import BaseGRU
from options import args_parser
import copy
import os
from tools import  model_test
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def initialize(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    save_path = args.save_path + f'/{args.task}/{args.train_rule}/{seed}'
    args.save_path = save_path
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    return args

if __name__ == '__main__':
    args = args_parser()
    args = initialize(args)
    result_file_path = os.path.join(args.save_path, 'results.json')
    print(args)

    # load dataset and user groups
    train_loader, val_loader,test_loader, gm, ids,input_size = get_dataset(args)

    global_model = BaseGRU(args,F=input_size[0], D=input_size[1],no_flat_features=input_size[2]).to(args.device)

    # Training Rule
    LocalUpdate = local_update(args.train_rule)

    # One Round Training Function
    train_round_parallel = one_round_training(args.train_rule)

    # Training
    train_loss, val_loss,train_acc,test_acc,final_list,local_clients = [], [] ,[], [], [] ,[]

#======================================================================================================#
    #inintial and upload GM
    for idx in range(args.num_users):
        local_clients.append(LocalUpdate(idx= ids[idx], args=args, train_set=train_loader[idx], val_set = val_loader[idx],test_set=test_loader[idx],model=copy.deepcopy(global_model)))

    if args.train_rule == 'FedRew':
        train_round_parallel = one_round_training('FedAvg')
        loss1 , loss2, result  =  train_round_parallel (args, global_model, local_clients,-1)
        print("Train Loss: {},Val Loss: {}".format(loss1, loss2))
        print("Init Model  Metrics: {}%".format(result))
        train_round_parallel = one_round_training(args.train_rule)

    for rnd in range(args.epochs):
        loss1,loss2, result = train_round_parallel(args, global_model, local_clients, rnd)
        train_loss.append(loss1)
        val_loss.append(loss2)
        print(args.train_rule)
        print("Train Loss: {},Val Loss: {}".format(loss1,loss2))
        print("Local Metrics on Local Data: {}%".format(result))
        final_list.append(result)

        # model_test(args, local_clients)


    results = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'local_metrics': final_list,
    }

    with open(result_file_path, 'w') as file:
        json.dump(results, file, indent=4)

    #test
    model_test(args,local_clients)

