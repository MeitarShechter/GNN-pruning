import torch
import numpy as np
import networkx as nx
import pickle

from scipy.stats import moment

from users_definitions import networks_data


'''
This file:
    - Load the trained network
    - Creates a graph representing the netwrok where:
        - Each node is a filter/neuron.
        - There's an edge between each two channels in the same layer and from each layer to the next
        - Each node has attributes depending on the weights of the matching layer
    - Save the network's graph
'''


def getStats(curr_filter):
    mean = torch.mean(curr_filter)
    std = torch.std(curr_filter)
    mom3 = moment(curr_filter, 3, None)
    mom4 = moment(curr_filter, 4, None)
    num_e = curr_filter.numel()

    return mean, std, mom3, mom4, num_e


def buildGraph(net_sd, network_name_graph):
    network_graph = nx.DiGraph()

    top_sort = nx.topological_sort(network_name_graph) 

    for node in top_sort:
        layer_w = net_sd[node]
        num_comp = layer_w.size()[0]

        if node.replace('weight', 'bias') in net_sd:
            biases = net_sd[node.replace('weight', 'bias')]
        else:
            biases = [0] * num_comp

        layer_w = layer_w.view(num_comp, -1)
        for i in range(num_comp):
            curr_comp = layer_w[i, :]

            mean, std, mom3, mom4, num_e = getStats(curr_comp)
            bias = biases[i]

            node_name = node + '_c' + str(i)
            network_graph.add_node(node_name, stat=torch.Tensor([mean, std, mom3, mom4, num_e, bias]))

            # connect between nodes within layer
            for j in range(i):
                prev_node = node + '_c' + str(j)
                network_graph.add_edge(node_name, prev_node)
                network_graph.add_edge(prev_node, node_name)
            
            predecessors = network_name_graph.predecessors(node)

            for pre_node in predecessors:
                pre_num_comp = net_sd[pre_node].size()[0]
                for j in range(pre_num_comp):
                    prev_node = pre_node + '_c' + str(j)
                    network_graph.add_edge(prev_node, node_name)
                    # connection opposite to data flow
                    # network_graph.add_edge(node_name, prev_node)
    
    return network_graph


def createNameGraph(network_name):
    network_name_graph = nx.DiGraph()

    info = networks_data.get(network_name) 
    CG_func = info.get('create_graph_func')
    network_name_graph = CG_func(network_name_graph)

    return network_name_graph


def convertModelToGraph(network_name):
    info = networks_data.get(network_name)

    network     = info.get('network')()
    net_sd_path = info.get('sd_path')

    # create the network's graph where each node is a layer in the model's state dict
    net_name_graph = createNameGraph(network_name)
    # get the model state dict
    net_sd = torch.load(net_sd_path, map_location=torch.device('cpu')) 
    # create the model's graph that the GNN will process
    network_graph = buildGraph(net_sd, net_name_graph)

    return network_graph


def saveGraph(net_graph, graph_path):
    with open(graph_path, 'wb') as f:
        pickle.dump(net_graph, f)


def main():
    # set parameters
    network_name = 'WRN_40_2'

    # get the desired path to save
    graph_path = networks_data.get(network_name).get('graph_path')

    # get the graph that represents the model
    net_graph = convertModelToGraph(network_name)

    # save it
    saveGraph(net_graph, graph_path)


if __name__ == "__main__":
    main()