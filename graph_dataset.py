import torch
import numpy as np
import networkx as nx
import pickle
import os

from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data

from network_graph_creation import convertModelToGraph, saveGraph
from users_definitions import networks_data

'''
This file:
    - defines the dataset to train our GNN based on the trained model's graph
'''

############################
### Building the dataset ###
############################

class GraphDataset(InMemoryDataset):
    def __init__(self, root, network_name, isWRN=False, graph_path=None, transform=None, pre_transform=None):
        self.network_name = network_name
        self.isWRN = isWRN
        if graph_path is None:
            self.graph_path = networks_data.get(network_name).get('graph_path') 
        else:
            self.graph_path = graph_path
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        raw_file_name = os.path.basename(self.graph_path)
        return [raw_file_name]

    @property
    def processed_file_names(self):
        raw_file_name = os.path.basename(self.graph_path)
        return [raw_file_name.replace('.pt', '_processed.dataset')]

    def download(self):
        net_name_graph = convertModelToGraph(self.network_name)
        saveGraph(net_name_graph, self.graph_path)

    def process(self):
        data_list = []
        raw_files = self.raw_paths

        for file in raw_files:
            with open(file, 'rb') as f:
                network_graph = pickle.load(f)

            # x - node features, y - node labels, edge_index - edges
            x = torch.zeros([len(set(network_graph)), 6])
            nodes = set(network_graph)
            nodes_dict = {}

            if self.isWRN:
                counter = 0
                for i in range(16):
                    node = 'conv1.weight_c{}'.format(i)
                    node_stat = network_graph.nodes[node]['stat']
                    x[counter, :] = node_stat
                    nodes_dict[node] = counter

                    counter += 1

                planes = [32, 64, 128]
                for layer in range(1,4):
                    for block in range(6):
                        for conv in range(1,3):
                            for j in range(planes[layer-1]):
                                node = 'layer{}.{}.conv{}.weight_c{}'.format(layer, block, conv, j)
                                node_stat = network_graph.nodes[node]['stat']
                                x[counter, :] = node_stat
                                nodes_dict[node] = counter

                                counter +=1

                        if block == 0:
                            for j in range(planes[layer-1]):
                                node = 'layer{}.0.shortcut.0.weight_c{}'.format(layer, j)
                                node_stat = network_graph.nodes[node]['stat']
                                x[counter, :] = node_stat
                                nodes_dict[node] = counter

                                counter +=1
            else:
                for i, node in enumerate(nodes):
                    node_stat = network_graph.nodes[node]['stat']
                    x[i, :] = node_stat
                    nodes_dict[node] = i

            edge_index = torch.zeros([2, len(nx.edges(network_graph))], dtype=torch.long)
            edge_c = 0

            for i, node in enumerate(nodes):
                edges = nx.edges(network_graph, node)
                node_i = nodes_dict[node]

                for j, edge in enumerate(edges):
                    edge_index[:, edge_c] = torch.tensor([node_i, nodes_dict[edge[1]]])
                    edge_c += 1

            # y = torch.full(x.shape, 0.5)

            data = Data(x=x, y=None, edge_index=edge_index)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# dataset = MNISTActivationsDataset('/Users/meitarshechter/Git/GNN-pruning/MNIST_model/')
# loader = DataLoader(dataset, batch_size=512, shuffle=True)