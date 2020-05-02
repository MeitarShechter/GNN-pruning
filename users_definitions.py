import torchvision.transforms as transforms

from torchvision.datasets.mnist import MNIST
from torchvision.datasets.cifar import CIFAR10

from modules import LeNet5, Masked_LeNet5, Pruned_LeNet5, resnet50, masked_resnet50, pruned_resnet50
from users_functions import *
# from modules import ResNeXt29_2x64d


'''
This file contains all the data the user should fill
'''


# A dictionary of the available train datasets
datasets_train = {
    'mnist': MNIST('./Data/',
                    download=True,
                    transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor()])),
    'cifar10': CIFAR10('./Data/',
                    download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]))
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))                   
}

# A dictionary of the available test datasets
datasets_test = {
    'mnist': MNIST('./Data/',
                    train=False,
                    download=True,
                    transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor()])),
    'cifar10': CIFAR10('./Data/',
                    train=False,
                    download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]))
                        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))                   
}

# A dictionary of the available networks and all their necessary data
networks_data = {
    'LeNet5': {
        'network' : LeNet5,
        'masked_network' : Masked_LeNet5,
        'pruned_network' : Pruned_LeNet5,
        'root' : './MNIST_models/',
        'graph_path' : './MNIST_models/raw/new_LeNet5_graph.pt',
        'sd_path' : './MNIST_models/new_LeNet5.pt',
        'dataset_name' : 'mnist',
        'create_graph_func' : LeNet5CreateNameGraph,
        'update_pruned_net_func' : LeNet5UpdatePrunedNet,
        'trained_GNN_path': './GNN_model/extra_confidence_GNNPrunningNet.pt'
    },
    'resnet50': {
        'network' : resnet50,
        'masked_network' : masked_resnet50,
        'pruned_network' : pruned_resnet50,
        'root' : './CIFAR10_models/',
        'graph_path' : './CIFAR10_models/raw/resnet50_graph.pt',
        'sd_path' : './CIFAR10_models/resnet50.pt',
        'dataset_name' : 'cifar10',
        'create_graph_func' : resnset50CreateNameGraph,
        'update_pruned_net_func' : resnet50UpdatePrunedNet,
        'trained_GNN_path': './GNN_model/extra_confidence_GNNPrunningNet_resnet50.pt'
    }
}
    # 'resnext' : ResNeXt29_2x64d,