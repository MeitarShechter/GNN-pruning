import numpy as np

import torch
import torch.nn as nn

from torch_geometric.data import DataLoader
from torchvision import transforms
from torchvision.datasets.mnist import MNIST

from graph_dataset import GraphDataset
from modules import GNNPrunningNet
from loss import GNN_prune_loss
from users_definitions import datasets_train, datasets_test, networks_data


#########################
### Train the network ###
#########################

def getPrunedNet(params_scores, orig_net, network_name, prune_factor=0.5):
    # prune_th = 0.5 
    prune_th, _ = torch.kthvalue(params_scores, int(np.ceil(prune_factor*params_scores.numel())))
    print("The thershold is {}".format(prune_th))
    params_mask = (params_scores > prune_th).detach().cpu().numpy()

    info = networks_data.get(network_name)

    ### build a new network based on params_mask ###
    # new_net = Pruned_LeNet5(params_mask)
    new_net = info.get('pruned_network')(params_mask)
    # new_net = Pruned_LeNet5(params_scores)

    new_dict = new_net.state_dict()
    pretrained_dict = orig_net.state_dict()  

    updatePrunedNet = info.get('update_pruned_net_func')

    new_dict = updatePrunedNet(new_dict, pretrained_dict, params_mask)

    # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    new_net.load_state_dict(new_dict)

    return new_net


def main():
    # set parameters
    network_name = 'resnet50'
    num_epochs = 10
    batch_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_retrain_epochs = 2

    # get GNN path
    info = networks_data.get(network_name)
    trained_model_path = info.get('trained_GNN_path')'

    # declare GNN model
    model = GNNPrunningNet(in_channels=6, out_channels=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    crit = GNN_prune_loss


    ########

    # # define the data loaders
    # # data_train_loader = DataLoader(data_train, batch_size=256, shuffle=True, num_workers=8)
    # data_train_loader = DataLoader(data_train, batch_size=128, shuffle=True, num_workers=8)
    # # data_test_loader = DataLoader(data_test, batch_size=1024, num_workers=8)
    # data_test_loader = DataLoader(data_test, batch_size=64, num_workers=8)

    #########

    root            = info.get('root')
    net_graph_path  = info.get('graph_path')
    sd_path         = info.get('sd_path')
    net             = info.get('network')

    train_dataset   = GraphDataset(root, network_name, net_graph_path)
    train_loader    = DataLoader(train_dataset, batch_size=batch_size)

    orig_net = net().to(device)
    orig_net.load_state_dict(torch.load(sd_path))



    # train_dataset = GraphDataset('/Users/meitarshechter/Git/GNN-pruning/MNIST_model/')
    # train_loader = DataLoader(train_dataset, batch_size=batch_size)

    # orig_net = LeNet5().to(device)
    # orig_net.load_state_dict(torch.load('/Users/meitarshechter/Git/GNN-pruning/MNIST_model/new_LeNet5.pt'))

    model.train()

    dataset_name = info.get('dataset_name')
    network_train_data = datasets_train.get(dataset_name)

    # mnist_train_data = MNIST('/Users/meitarshechter/Git/GNN-pruning/MNIST_Data/', 
    #     train=True, 
    #     download=True,
    #     transform=transforms.Compose([
    #                 transforms.Resize((32, 32)),
    #                 transforms.ToTensor()]))

    print("Start training")

    # for epoch in range(num_epochs):
    #     loss_all = 0
        
    #     for data in train_loader:
    #         data = data.to(device)
    #         optimizer.zero_grad()
    #         output = model(data)
    #         loss = crit(output, orig_net, network_name, network_train_data)
    #         # print("Calculated loss: {}!".format(loss.item()))
    #         loss.backward()
    #         loss_all += data.num_graphs * loss.item()
    #         optimizer.step()
            
    #         print("epoch {}. total loss is: {}".format(epoch+1, loss_all / len(train_dataset)))

    # torch.save(model.state_dict(), trained_model_path)            

    print("Start evaluating")

    model.load_state_dict(torch.load(trained_model_path))

    model.eval()

    network_val_data = datasets_test.get(dataset_name)

    # mnist_val_data = MNIST('/Users/meitarshechter/Git/GNN-pruning/MNIST_Data/', 
    #     train=False, 
    #     download=True,
    #     transform=transforms.Compose([
    #                 transforms.Resize((32, 32)),
    #                 transforms.ToTensor()]))
    val_data_loader = torch.utils.data.DataLoader(network_val_data, batch_size=1024, shuffle=False, num_workers=8) 

    with torch.no_grad():
        for data in train_loader:
            data = data.to(device)

            # pred = model(data).detach().cpu().numpy()
            pred = model(data)

            prunedNet = getPrunedNet(pred, orig_net, network_name, prune_factor=0.8)

    # Train the pruned network
    prunedNet.train()
    # data_train_loader = torch.utils.data.DataLoader(mnist_train_data, batch_size=256, shuffle=False, num_workers=8) 
    data_train_loader = torch.utils.data.DataLoader(network_train_data, batch_size=256, shuffle=False, num_workers=8) 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(prunedNet.parameters(), lr=2e-3)
    for epoch in range (n_retrain_epochs):
        for i, (images, labels) in enumerate(data_train_loader):
            optimizer.zero_grad()

            output = prunedNet(images)

            loss = criterion(output, labels)

            if i % 10 == 0:
                print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch+1, i, loss.detach().cpu().item()))

            loss.backward()
            optimizer.step()



    with torch.no_grad():

        total_correct = 0
        
        for i, (images, labels) in enumerate(val_data_loader):
            output = prunedNet(images)
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()

        acc = float(total_correct) / len(mnist_val_data)

        print("The pruned network accuracy is: {}".format(acc))







# for epoch in range(1):
#     loss = train()
#     train_acc = evaluate(train_loader)
#     val_acc = evaluate(val_loader)    
#     test_acc = evaluate(test_loader)
#     print('Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}, Val Auc: {:.5f}, Test Auc: {:.5f}'.



# for idx, m in enumerate(model.modules()):
#     print(idx, '->', m)

# for i, (images, labels) in enumerate(data_loader):
    # output = model(images)

if __name__ == "__main__":
    main()





