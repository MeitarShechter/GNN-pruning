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
            # for j in range(i):
            #     prev_node = node + '_c' + str(j)
            #     network_graph.add_edge(node_name, prev_node)
            #     network_graph.add_edge(prev_node, node_name)
            
            predecessors = network_name_graph.predecessors(node)

            for pre_node in predecessors:
                pre_num_comp = net_sd[pre_node].size()[0]
                for j in range(pre_num_comp):
                    prev_node = pre_node + '_c' + str(j)
                    network_graph.add_edge(prev_node, node_name)
                    # connection opposite to data flow
                    network_graph.add_edge(node_name, prev_node)
    
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
    # model_named_graph = createNamedGraph(model)
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
    network_name = 'resnet50'

    # get the desired path to save
    graph_path = networks_data.get(network_name).get('graph_path')

    # get the graph that represents the model
    net_graph = convertModelToGraph(network_name)

    # save it
    saveGraph(net_graph, graph_path)


if __name__ == "__main__":
    main()



# TODO: use onnx to create the graph (if possible)
# def createNamedGraph(model):
#     model_named_graph = nx.DiGraph()

#     # args = torch.zeros([1, 1, 32, 32])
#     args = torch.zeros([1, 3, 32, 32])
#     trace, _ = torch.jit._get_trace_graph(model, args)
#     torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)

#     # g, par_dict, t_out = torch.onnx.utils._model_to_graph(model, args)
#     # for node in g.nodes():
#     #     # print(node)
#     #     op = node.kind()
#     #     print(op)

#     #     inputs = [i.unique() for i in node.inputs()]
#     #     outputs = [o.unique() for o in node.outputs()]

#     #     model_G.add_node(node.__repr__())
#     #     # Add edges
#     #     for target_torch_node in g.nodes():
#     #         target_outputs = [i.unique() for i in target_torch_node.outputs()]
#     #         if set(inputs) & set(target_outputs):
#     #             # hl_graph.add_edge_by_id(pytorch_id(torch_node), pytorch_id(target_torch_node), shape)

#     #             # model_G.add_node(node_name, stat=torch.Tensor([mean, std, mom3, mom4, num_e, bias]))
#     #             model_G.add_edge(target_torch_node.__repr__(), node.__repr__())
#     #                     # model_G.add_edge(prev_node, node_name)

#     torch_graph = trace
#     for torch_node in torch_graph.nodes():
#         # Op
#         op = torch_node.kind()
#         # print(op)
#         # if 'conv' in op or 'addmm' in op:
#         #     print(op)

#         # Parameters
#         # params = {k: torch_node[k] for k in torch_node.attributeNames()}
#         # print(params) 
#         # Inputs/outputs
#         inputs = [i.unique() for i in torch_node.inputs()]
#         outputs = [o.unique() for o in torch_node.outputs()]
#         # Get output shape
#         shape = get_shape(torch_node)
#         # print(shape)
#         # Add HL node
#         # hl_node = Node(uid=pytorch_id(torch_node), name=None, op=op, 
#         #                 output_shape=shape, params=params)
#         # hl_graph.add_node(hl_node)
#         model_G.add_node(torch_node.__repr__())
#         # Add edges
#         for target_torch_node in torch_graph.nodes():
#             target_outputs = [i.unique() for i in target_torch_node.outputs()]
#             if set(inputs) & set(target_outputs):
#                 # hl_graph.add_edge_by_id(pytorch_id(torch_node), pytorch_id(target_torch_node), shape)

#                 # model_G.add_node(node_name, stat=torch.Tensor([mean, std, mom3, mom4, num_e, bias]))
#                 model_G.add_edge(target_torch_node.__repr__(), torch_node.__repr__())
#                         # model_G.add_edge(prev_node, node_name)

#     top_sort = nx.topological_sort(model_G)
#     for node in top_sort:
#         if 'convolution' in node or 'addmm' in node or 'batch_norm' in node:
#             continue

#         pre = model_G.predecessors(node)
#         suc = model_G.successors(node)
#         for p in pre:
#             for s in suc:    
#                 model_G.add_edge(p, s)
        
#         model_G.remove_node(node)

#     return model_named_graph



### A WORKING FUNCION FOR LeNet5 that given the network's state dict creates the network's graph ###
# def buildGraph(model_sd):
#     model_G = nx.DiGraph()

#     prev_layer_n = None
#     for idx, layer_n in enumerate(model_sd):
#         if 'bias' in layer_n:
#             continue

#         layer_w = model_sd[layer_n]
#         biases = model_sd[layer_n.replace('weight', 'bias')]

#         if 'convnet' in layer_n:
#             num_filters = layer_w.size()[0]
#             if 'weight' in layer_n:
#                 for i in range(num_filters):
#                     curr_filter = layer_w[i,:,:,:]

#                     mean, std, mom3, mom4, num_e = getStats(curr_filter)
#                     bias = biases[i]

#                     node_name = layer_n + '_f' + str(i)
#                     model_G.add_node(node_name, stat=torch.Tensor([mean, std, mom3, mom4, num_e, bias]))

#                     for j in range(i):
#                         prev_node = layer_n + '_f' + str(j)
#                         model_G.add_edge(node_name, prev_node)
#                         model_G.add_edge(prev_node, node_name)
                    
#                     if prev_layer_n is not None:                    
#                         prev_num_filters = model_sd[prev_layer_n].size()[0]
#                         for j in range(prev_num_filters):
#                             prev_node = prev_layer_n + '_f' + str(j)
#                             model_G.add_edge(prev_node, node_name)
#             else:
#                 print("ERROR: A layer without a bias and without weight")

#         elif 'fc' in layer_n:
#             num_neurons = layer_w.size()[0]
#             if 'weight' in layer_n:
#                 for i in range(num_neurons):
#                     curr_nueron = layer_w[i,:]

#                     mean, std, mom3, mom4, num_e = getStats(curr_nueron)
#                     bias = biases[i]

#                     node_name = layer_n + '_n' + str(i)
#                     model_G.add_node(node_name, stat=torch.Tensor([mean, std, mom3, mom4, num_e, bias]))

#                     for j in range(i):
#                         prev_node = layer_n + '_n' + str(j)
#                         model_G.add_edge(node_name, prev_node)
#                         model_G.add_edge(prev_node, node_name)
                    
#                     if prev_layer_n is not None:                    
#                         prev_num_params = model_sd[prev_layer_n].size()[0]
#                         for j in range(prev_num_params):
#                             if 'convnet' in prev_layer_n:
#                                 prev_node = prev_layer_n + '_f' + str(j)
#                             elif 'fc' in prev_layer_n:
#                                 prev_node = prev_layer_n + '_n' + str(j)
#                             model_G.add_edge(prev_node, node_name)
#             else:
#                 print("ERROR: A layer without a bias and without weight")

#         prev_layer_n = layer_n
    
#     return model_G




# maybe I need this for the onnx function??
# import re
# def get_shape(torch_node):
#     """Return the output shape of the given Pytorch node."""
#     # Extract node output shape from the node string representation
#     # This is a hack because there doesn't seem to be an official way to do it.
#     # See my quesiton in the PyTorch forum:
#     # https://discuss.pytorch.org/t/node-output-shape-from-trace-graph/24351/2
#     # TODO: find a better way to extract output shape
#     # TODO: Assuming the node has one output. Update if we encounter a multi-output node.
#     m = re.match(r".*Float\(([\d\s\,]+)\).*", str(next(torch_node.outputs())))
#     if m:
#         shape = m.group(1)
#         shape = shape.split(",")
#         shape = tuple(map(int, shape))
#     else:
#         shape = None
#     return shape