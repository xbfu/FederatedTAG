import torch
from torch import Tensor
from typing import List, Optional, Union
import random
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
EOS = 1e-10


def to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def normalize(adj):
    inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
    return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]


def get_subgraph(subset: Union[Tensor, List[int]], edge_index: Tensor, edge_attr: Optional[Tensor] = None, relabel_nodes: bool = False, num_nodes: Optional[int] = None):

    device = edge_index.device
    # num_nodes = maybe_num_nodes(edge_index, num_nodes)
    node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    node_mask[subset] = 1

    if relabel_nodes:
        node_idx = torch.zeros(num_nodes, dtype=torch.long, device=device)
        node_idx[subset] = torch.arange(subset.size(0), device=device)

    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask] if edge_attr is not None else None

    edge_index = node_idx[edge_index]
    return edge_index, edge_attr


def generate_feature(pretrained_model_name, text_list, device, batch_size=32):
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    language_model = AutoModel.from_pretrained(pretrained_model_name).to(device)
    language_model.eval()

    all_features = []

    with torch.no_grad():
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i + batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = language_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_features.append(embeddings.cpu())

    return torch.cat(all_features, dim=0)
