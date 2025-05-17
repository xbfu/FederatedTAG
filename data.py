import numpy as np
import torch
from torch_geometric.utils import to_undirected
from datasets import load_dataset


def load_data(dataset_name, data_path, file_names):
    data = load_dataset(path=f'{data_path}/{dataset_name}')
    dataset = {'category_list': [], 'label_list': [], 'head_nodes': [], 'tail_nodes': [], 'text_list': []}
    head_nodes = []
    tail_nodes = []
    for node in data['train']:
        head_node = node['node_id']
        dataset['category_list'].append(node['category'])               # label name
        dataset['label_list'].append(node['label'])                     # label index
        dataset['text_list'].append(node['text'])                       # text attribute

        if node['neighbour'] != '[]':                                   # if the node has neighbors
            neighbor_list = node['neighbour'][1:-1].split(',')          # split to a set of neighboring nodes
            for neighbor in neighbor_list:
                tail_node = int(neighbor)
                if head_node != tail_node:                              # discard self loops
                    head_nodes.append(head_node)                        # append head node
                    tail_nodes.append(tail_node)                        # append tail node

    edge_index = torch.tensor([head_nodes, tail_nodes])                 # construct edge_index
    if dataset_name != 'Arxiv':
        edge_index = to_undirected(edge_index)                          # convert to undirected graph
    dataset['edge_index'] = edge_index
    dataset['dataset_name'] = dataset_name
    dataset['num_nodes'] = len(data['train'])
    dataset['num_classes'] = max(dataset['label_list']) + 1

    trainIdx = []
    valIdx = []
    testIdx = []
    file_list = []
    num_clients = 0
    for file in file_names:
        if file.find(f'{dataset_name}') == 0:
            file_list.append(file)
            num_clients += 1
    np.random.shuffle(file_list)

    for file in file_list:
        node_list = np.loadtxt(f'./partition/{file}').astype(int)
        np.random.shuffle(node_list)
        # select 10% as training, 10% validation, the rest test
        trainIdx.append(list(node_list)[: int(0.1 * len(node_list))])
        valIdx.append(list(node_list)[int(0.1 * len(node_list)): int(0.2 * len(node_list))])
        testIdx.append(list(node_list)[int(0.2 * len(node_list)):])
    return dataset, num_clients, trainIdx, valIdx, testIdx

