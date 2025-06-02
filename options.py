
import argparse
def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--train_rule', type=str, default='FedRew', help='the training rule for personalized FL')
    parser.add_argument('--seed', type=int, default=10,
                        help="number of training epochs")
    parser.add_argument('--num_users', type=int, default=10,
                        help="number of users: n")
    parser.add_argument('--frac', type=float, default=1,
                        help='the fraction of clients: C')
    parser.add_argument('--lambda_u', type=float, default=0.4,
                        help='组合系数')
    parser.add_argument('--epochs', type=int, default=50,
                        help="number of training epochs")
    parser.add_argument('--local_bs', type=int, default=50,
                        help="local batch size: b")
    parser.add_argument('--local_epoch', type=int, default=5,
                        help="the number of local epochs")
    parser.add_argument('--lr', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--time_before', type=int, default=47,
                        help="number of Finetuning  epochs")

    parser.add_argument('--lr_classifier', default=0.005, type=float, help='learning rate of classifier')
    parser.add_argument('--rew_epoch', default=5, type=int, help='epoch of rew')
    parser.add_argument('--dataset', type=str, default='eICU_hos',
                        help="name of dataset")
    parser.add_argument('--eicu_dir', type=str, default='final_dataset/hos',
                        help="dir of eICU dataset")
    parser.add_argument('--num_classes', type=int, default=2, help="number of classes")
    parser.add_argument('--device', type=str, default='cuda:0', help="To use cuda, set to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='sgd', help="type of optimizer")
    parser.add_argument('--weight_decay', type=float, default=0.001, help="")
    parser.add_argument('--momentum', type=float, default=0.9, help="")
    parser.add_argument('--save_results_csv', action='store_true')#默认不保存
    parser.add_argument('--save_path', default='Results/',help="path of result,test_path,and result_path")
    parser.add_argument('--task', default='mortality', type=str, help='can be either LoS, mortality')

    args = parser.parse_args()
    return args
