# import matplotlib.pyplot as plt
import numpy as np
import random

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.data import DataLoader
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from torch.optim.lr_scheduler import MultiStepLR

from graph_dataset import GraphDataset
from modules import GNNPrunningNet, get_n_params as gnp
from loss import GNN_prune_loss
from users_definitions import datasets_train, datasets_test, networks_data


#########################
### Train the network ###
#########################

def getPrunedNet(params_scores, orig_net, network_name, prune_factor=0.5):
    # prune_th = prune_factor
    prune_th, _ = torch.kthvalue(params_scores, int(np.ceil(prune_factor*params_scores.numel())))
    print("The thershold is {}".format(prune_th))
    params_mask = (params_scores > prune_th).detach().cpu().numpy()

    info = networks_data.get(network_name)

    ### build a new network based on params_mask ###
    new_net = info.get('pruned_network')(params_mask)

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
    # random.seed(0)
    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)
    # torch.cuda.manual_seed(0)

    # set all hyperparameters
    network_name = 'WRN_40_2'
    num_epochs = 35
    batch_size = 1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_retrain_epochs = 40 
    trials = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9]
    lr = 3e-4
    opt = "Adam"
    use_temp = False
    use_steps = False

    # set paths
    checkpointPath = './GNN_model/CIFAR10_checkpoints/CP__num_e_{}__retrain_e_{}__lr_{}__opt_{}__useTemp_{}__useSteps_{}__epoch_{}.pt'.format(num_epochs, n_retrain_epochs, lr, opt, use_temp, use_steps, '{}')    
    continue_train = False
    checkpointLoadPath = './GNN_model/CIFAR10_checkpoints/CP__num_e_{}__retrain_e_{}__lr_{}__opt_{}__useTemp_{}__useSteps_{}__epoch_{}.pt'.format(num_epochs, n_retrain_epochs, lr, opt, use_temp, use_steps, '20')

    # get GNN path
    info = networks_data.get(network_name)
    trained_model_path = info.get('trained_GNN_path').replace('.pt', '___num_e_{}__retrain_e_{}__lr_{}__opt_{}__useTemp_{}__useSteps_{}.pt'.format(num_epochs, n_retrain_epochs, lr, opt, use_temp, use_steps))

    # declare GNN model
    model = GNNPrunningNet(in_channels=6, out_channels=128).to(device)
    if opt == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        # lr = 0.1
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=[int(elem*num_epochs) for elem in [0.3, 0.6, 0.8]], gamma=0.2)
    crit = GNN_prune_loss

    # declate TensorBoard writer
    summary_path = '{}-num_e_{}__retrain_e_{}__lr_{}__opt_{}__useTemp_{}__useSteps_{}/training'.format(network_name, num_epochs, n_retrain_epochs, lr, opt, use_temp, use_steps)
    writer = SummaryWriter(summary_path)


    root            = info.get('root')
    net_graph_path  = info.get('graph_path')
    sd_path         = info.get('sd_path')
    net             = info.get('network')
    orig_net_loss   = info.get('orig_net_loss') 

    isWRN = (network_name == "WRN_40_2")
    train_dataset   = GraphDataset(root, network_name, isWRN, net_graph_path)
    train_loader    = DataLoader(train_dataset, batch_size=batch_size)

    orig_net = net().to(device)
    orig_net.load_state_dict(torch.load(sd_path, map_location=device))

    model.train()

    dataset_name = info.get('dataset_name')
    network_train_data = datasets_train.get(dataset_name)

    print("Start training")

    if continue_train == True:
        cp = torch.load(checkpointLoadPath, map_location=device)
        trained_epochs = cp['epoch'] + 1
        sd = cp['model_state_dict']
        model.load_state_dict(sd)
        op_sd = cp['optimizer_state_dict']
        optimizer.load_state_dict(op_sd)
    else:
        trained_epochs = 0

    loss_all = 0.0
    data_all = 0.0
    sparse_all = 0.0
    if use_temp == True:
        T = 1.0
        if trained_epochs > 0:
            T = np.power(2, np.floor(trained_epochs / int(num_epochs/3)))

    for epoch in range(trained_epochs, num_epochs):
        
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)

            if use_temp == True:
            # Use temperature
                nom = torch.pow((torch.exp(torch.tensor(T, device=device))), output)
                dom = torch.pow((torch.exp(torch.tensor(T, device=device))), output) + torch.pow((torch.exp(torch.tensor(T, device=device))), (1-output))
                output = nom/dom
                # continue as usual

            sparse_term, data_term, data_grad = crit(output, orig_net, orig_net_loss, network_name, network_train_data, device, gamma1=10, gamma2=0.1)

            if use_steps == True:
                if epoch % 3 == 0: # do 2 steps in data direction then 1 in sparsity
                    sparse_term.backward()
                else:
                    output.backward(data_grad)
            else:            
                sparse_term.backward(retain_graph=True)
                output.backward(data_grad)

            data_all += data.num_graphs * data_term.item()
            sparse_all += data.num_graphs * sparse_term.item()
            loss_all += data_all + sparse_all
            optimizer.step()
            
        print("epoch {}. total loss is: {}".format(epoch+1, (data_term.item() + sparse_term.item()) / len(train_dataset)))
        
        if opt != "Adam":
            scheduler.step()

        if use_temp == True:
        # increase temperature 3 times
            if (epoch+1) % int(num_epochs/3) == 0:
                T *= 2

        if epoch % 10 == 9:
            writer.add_scalars('Learning curve', {
            'loss data term': data_all/10,
            'loss sparsity term': sparse_all/10,
            'training loss': loss_all/10
            }, epoch+1)            

            # save checkpoint
            if opt == "Adam":
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_all,
                }, checkpointPath.format(epoch+1))
            else:
                torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_all,
                'scheduler_state_dict': scheduler.state_dict(),
                }, checkpointPath.format(epoch+1))

            loss_all = 0.0
            data_all = 0.0
            sparse_all = 0.0
            

    torch.save(model.state_dict(), trained_model_path)            

    print("Start evaluating")

    model.load_state_dict(torch.load(trained_model_path, map_location=device))

    model.eval()

    network_val_data = datasets_test.get(dataset_name)
    val_data_loader = torch.utils.data.DataLoader(network_val_data, batch_size=1024, shuffle=False, num_workers=8) 

    for trial, p_factor in enumerate(trials):
        with torch.no_grad():
            for data in train_loader:
                data = data.to(device)

                pred = model(data)

                prunedNet = getPrunedNet(pred, orig_net, network_name, prune_factor=p_factor).to(device)

        # Train the pruned network
        prunedNet.train()

        data_train_loader = torch.utils.data.DataLoader(network_train_data, batch_size=256, shuffle=False, num_workers=8) 
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(prunedNet.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, milestones=[int(elem*n_retrain_epochs) for elem in [0.3, 0.6, 0.8]], gamma=0.2)

        for epoch in range (n_retrain_epochs):
            for i, (images, labels) in enumerate(data_train_loader):
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                output = prunedNet(images)
                loss = criterion(output, labels)

                if i % 30 == 0:
                    print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch+1, i, loss.detach().cpu().item()))

                loss.backward()
                optimizer.step()

            scheduler.step()

        # Evaluate the pruned net
        with torch.no_grad():

            total_correct = 0
            cuda_time = 0.0            
            cpu_time = 0.0

            for i, (images, labels) in enumerate(val_data_loader):
                images, labels = images.to(device), labels.to(device)

                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    output = prunedNet(images)
                cuda_time += sum([item.cuda_time for item in prof.function_events])
                cpu_time += sum([item.cpu_time for item in prof.function_events])

                pred = output.detach().max(1)[1]
                total_correct += pred.eq(labels.view_as(pred)).sum()

            p_acc = float(total_correct) / len(network_val_data)
            p_num_params = gnp(prunedNet)
            p_cuda_time = cuda_time / len(network_val_data)
            p_cpu_time = cpu_time / len(network_val_data)

            print("The pruned network for prune factor {} accuracy is: {}".format(p_factor, p_acc))
            print("The pruned network number of parameters is: {}".format(p_num_params))
            print("The pruned network cuda time is: {}".format(p_cuda_time))
            print("The pruned network cpu time is: {}".format(p_cpu_time))

        # Evaluate the original net
        with torch.no_grad():

            total_correct = 0
            cuda_time = 0.0            
            cpu_time = 0.0
            
            for i, (images, labels) in enumerate(val_data_loader):
                images, labels = images.to(device), labels.to(device)

                with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    output = orig_net(images)
                cuda_time += sum([item.cuda_time for item in prof.function_events])
                cpu_time += sum([item.cpu_time for item in prof.function_events])

                pred = output.detach().max(1)[1]
                total_correct += pred.eq(labels.view_as(pred)).sum()

            o_acc = float(total_correct) / len(network_val_data)
            o_num_params = gnp(orig_net)
            o_cuda_time = cuda_time / len(network_val_data)
            o_cpu_time = cpu_time / len(network_val_data)

            print("The original network accuracy is: {}".format(o_acc))
            print("The original network number of parameters is: {}".format(o_num_params))
            print("The original network cuda time is: {}".format(o_cuda_time))
            print("The original network cpu time is: {}".format(o_cpu_time))

        writer.add_scalars('Network accuracy', {
            'original': o_acc,
            'pruned': p_acc
            }, 100*p_factor)
        writer.add_scalars('Network number of parameters', {
            'original': o_num_params,
            'pruned': p_num_params
            }, 100*p_factor)
        writer.add_scalars('Network GPU time', {
            'original': o_cuda_time,
            'pruned': p_cuda_time
            }, 100*p_factor)
        writer.add_scalars('Network CPU time', {
            'original': o_cpu_time,
            'pruned': p_cpu_time
            }, 100*p_factor)

    writer.close()


if __name__ == "__main__":
    main()





