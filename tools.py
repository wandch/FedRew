import copy
import os
import torch
import math
import numpy as np







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

