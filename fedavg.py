import os
import argparse
import copy
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from data import load_data
from utils import get_subgraph, to_numpy
from model import GCN
from logger import Logger


class Server(object):
    def __init__(self, client_list, client_ids, dataset, trainIdx, hidden, device, logger):
        self.client_list = client_list
        self.client_ids = client_ids
        self.dataset = dataset
        self.trainIdx = trainIdx
        self.classes = [c for c in range(dataset['num_classes'])]
        self.num_train_nodes = [len(self.trainIdx[client_id]) for client_id in self.client_ids]
        self.coefficients = [num_train_nodes / sum(self.num_train_nodes) for num_train_nodes in self.num_train_nodes]
        self.train_label_count = [client.label_dist for client in self.client_list]
        self.gnn = GCN(input_dim=client_list[0].num_features, hidden_dim=hidden, output_dim=dataset['num_classes']).to(device)
        self.logger = logger
        self.device = device

    def train(self, rounds):
        best_val_loss = 1e6
        best_model = copy.deepcopy(self.gnn)
        for round in range(1, rounds+1):
            gnn_averaged_weights = {}
            for i, client in enumerate(self.client_list):
                # collect updated parameters from client i
                client.set_parameters(self.gnn)
                gnn_weight = client.local_update()

                for key in self.gnn.state_dict().keys():
                    if key in gnn_averaged_weights.keys():
                        gnn_averaged_weights[key] += self.coefficients[i] * gnn_weight[key]
                    else:
                        gnn_averaged_weights[key] = self.coefficients[i] * gnn_weight[key]

            # update global model parameters
            self.gnn.load_state_dict(gnn_averaged_weights)

            loss_list = []
            val_loss_list = []
            num_val_list = []
            num_test_list = []
            correct_train_list = []
            correct_val_list = []
            correct_test_list = []
            label_test_list = []

            # get accuracy
            for i, client in enumerate(self.client_list):
                client.set_parameters(self.gnn)
                loss, val_loss, num_val, num_test, correct_train, correct_val, correct_test, label_test = client.stats()
                loss_list.append(loss)
                val_loss_list.append(val_loss)
                num_val_list.append(num_val)
                num_test_list.append(num_test)
                correct_train_list.append(correct_train)
                correct_val_list.append(correct_val)
                correct_test_list.append(correct_test)
                label_test_list.append(label_test)

            total_val = np.sum(num_val_list)
            total_test = np.sum(num_test_list)
            train_loss = np.sum(loss_list) / np.sum(self.num_train_nodes)
            val_loss = np.sum(val_loss_list) / total_val
            acc_train = np.sum(correct_train_list) / np.sum(self.num_train_nodes)
            acc_val = np.sum(correct_val_list) / total_val
            acc_test = np.sum(correct_test_list) / total_test
            log_info = ''.join(['| Round:{:4d} '.format(round),
                                '| train_loss: {:7.5f} '.format(train_loss),
                                '| val_loss: {:7.5f} '.format(val_loss),
                                '| acc_train: {:7.5f} '.format(acc_train),
                                '| acc_val: {:7.5f} '.format(acc_val),
                                '| acc_test: {:7.5f} |'.format(acc_test)])
            self.logger.info(log_info)

            if val_loss < best_val_loss - 0:
                best_val_loss = val_loss
                best_test_acc = acc_test
                best_model = copy.deepcopy(self.gnn)

        return best_test_acc


class Client(object):
    def __init__(self, client_id, dataset, trainIdx, valIdx, testIdx, plm, lr, hidden, epochs, device):
        self.client_id = client_id
        self.dataset_name = dataset['dataset_name']
        self.node_list = trainIdx + valIdx + testIdx
        self.trainIdx = list(range(0, len(trainIdx)))
        self.valIdx = list(range(len(trainIdx), len(trainIdx) + len(valIdx)))
        self.testIdx = list(range(len(trainIdx) + len(valIdx), len(trainIdx) + len(valIdx) + len(testIdx)))

        self.label_list = [dataset['label_list'][node] for node in self.node_list]
        self.labels = torch.tensor(self.label_list).to(device)

        # plm = 'bert'
        self.text_list = [dataset['text_list'][node] for node in self.node_list]
        self.features = torch.load(f'/mnt/nvme0/xingbo/FederatedTAG/text_embeddings/{self.dataset_name}_{plm}_features.pt')[self.node_list].to(device)
        self.num_features = self.features.shape[1]
        self.classes = [c for c in range(dataset['num_classes'])]
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.label_dist = [len(torch.where(self.labels[self.trainIdx] == c)[0]) / len(self.trainIdx) for c in self.classes]

        # get local graph
        self.subgraph = get_subgraph(subset=torch.tensor(self.node_list, dtype=torch.long),
                                     edge_index=dataset['edge_index'],
                                     relabel_nodes=True,
                                     num_nodes=dataset['num_nodes'])
        self.edge_index = self.subgraph[0].to(device)

        # initialize model
        self.gnn = GCN(input_dim=self.num_features, hidden_dim=hidden, output_dim=dataset['num_classes']).to(device)
        self.optimizer = torch.optim.Adam(self.gnn.parameters(), lr=self.lr)

    def set_parameters(self, gnn):
        for (_, new_param), (name, old_param) in zip(gnn.named_parameters(), self.gnn.named_parameters()):
            old_param.data = new_param.data.clone()

    def local_update(self):
        # self.gnn.load_state_dict(gnn.state_dict())
        self.gnn.train()
        for epoch in range(1, self.epochs + 1):
            self.optimizer.zero_grad()
            output_gnn = self.gnn(self.features, self.edge_index)
            ce_loss = F.cross_entropy(output_gnn[self.trainIdx], self.labels[self.trainIdx])
            loss = 1. * ce_loss
            loss.backward()
            self.optimizer.step()

        return self.gnn.state_dict()

    def stats(self):
        # self.gnn.load_state_dict(gnn.state_dict())
        self.gnn.eval()
        output = self.gnn(self.features, self.edge_index)
        loss = F.cross_entropy(output[self.trainIdx], self.labels[self.trainIdx])
        val_loss = F.cross_entropy(output[self.valIdx], self.labels[self.valIdx])
        pred = output.argmax(dim=1)
        correct_train = sum(np.array(self.labels[self.trainIdx].cpu()) == np.array(pred[self.trainIdx].cpu()))
        correct_val = sum(np.array(self.labels[self.valIdx].cpu()) == np.array(pred[self.valIdx].cpu()))
        correct_test = sum(np.array(self.labels[self.testIdx].cpu()) == np.array(pred[self.testIdx].cpu()))
        label_test = np.array(self.labels[self.testIdx].cpu())

        return loss.item() * len(self.trainIdx), val_loss.item() * len(self.valIdx), \
               len(self.valIdx), len(self.testIdx), correct_train, correct_val, correct_test, label_test


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(dataset_name, data_path, plm, hidden, lr, epochs, rounds, gpu_id, seed):
    arch_name = os.path.basename(__file__).split('.')[0]
    file_names = sorted(os.listdir('./partition/'))                        # directory to stored partition
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    filename = f'log/{dataset_name}_{arch_name}_{plm}_{hidden}_{lr}_{epochs}_{rounds}_{seed}.log'
    logger = Logger(filename, formatter)

    set_random_seed(seed)
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    dataset, num_clients, trainIdx, valIdx, testIdx = load_data(dataset_name, data_path, file_names)

    plm_list = [plm.split('+')[0] for _ in range(num_clients // 2)]
    plm_list.extend([plm.split('+')[1] for _ in range(num_clients // 2, num_clients)])
    print(plm_list)

    client_ids = [i for i in range(num_clients)]
    client_list = [Client(i, dataset, trainIdx[i], valIdx[i], testIdx[i], plm_list[i], lr, hidden, epochs, device) for i in client_ids]

    server = Server(client_list, client_ids, dataset, trainIdx, hidden, device, logger)
    best_test_acc = server.train(rounds=rounds)
    log_info = ''.join(['| Arch: {:s} '.format(arch_name),
                        '| dataset: {:s} '.format(dataset_name),
                        '| plm: {:s} '.format(plm),
                        '| hidden:{:3d} '.format(hidden),
                        '| lr: {:6.4f} '.format(lr),
                        '| epochs:{:2d} '.format(epochs),
                        '| rounds:{:2d} '.format(rounds),
                        '| seed: {:2d} '.format(seed),
                        '| best_test_acc: {:6.4f} |'.format(best_test_acc)])
    logger.info(log_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Official code')
    parser.add_argument('--dataset_name', type=str, default='History', help='dataset used for training')
    parser.add_argument('--data_path', type=str, default='/mnt/nvme0/xingbo/FederatedTAG/CSTAG', help='data directory')
    parser.add_argument('--hidden', type=int, default=128, help='hidden size in the model (default: 100)')
    parser.add_argument('--plm', type=str, default='bert+gpt2', help='pre-trained language models')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate (default: 0.001)')
    parser.add_argument('--epochs', type=int, default=5, help='number of local epochs (default: 5)')
    parser.add_argument('--rounds', type=int, default=500, help='number of rounds (default: 100)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU device ID (default: 0)')
    args = parser.parse_args()

    dataset_name = args.dataset_name
    # dataset_list = ['History', 'Photo', 'Children', 'Computers', 'Fitness']
    data_path = args.data_path
    hidden = args.hidden
    plm = args.plm
    # plm_list = ['bert+gpt2', 'bert+roberta', 'gpt2+roberta']
    lr = args.lr
    epochs = args.epochs
    rounds = args.rounds
    gpu_id = args.gpu_id

    for seed in range(5):
        print('Dataset_name: ', dataset_name,
              ' | hidden: ', hidden,
              ' | plm: ', plm,
              ' | lr: ', lr,
              ' | epochs: ', epochs,
              ' | rounds: ', rounds,
              ' | gpu_id: ', gpu_id,
              ' | seed: ', seed)
        run(dataset_name=dataset_name, data_path=data_path, plm=plm, hidden=hidden,
            lr=lr, epochs=epochs, rounds=rounds, gpu_id=gpu_id, seed=seed)