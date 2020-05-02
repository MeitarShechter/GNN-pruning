import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from users_definitions import datasets_train, datasets_test, networks_data


'''
This file:
    - Train the chosen network
    - Evaluates the network
    - Save the network's state_dict and accuracy
'''


def train(epoch, net, criterion, optimizer, data_train_loader, len_train):
    net.train()
    for i, (images, labels) in enumerate(data_train_loader):
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)

        if i % 10 == 0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, i, loss.detach().cpu().item()))

        loss.backward()
        optimizer.step()


def test(net, criterion, data_test_loader, len_test):
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    for i, (images, labels) in enumerate(data_test_loader):
        output = net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len_test
    acc = float(total_correct) / len_test 
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.detach().cpu().item(), acc))

    return acc


def main():
    # set parameters
    num_epochs  = 16
    train_bs    = 128
    test_bs     = 64
    dataset_name = 'cifar10'
    network_name = 'resnet50'
    save_path   = '/Users/meitarshechter/Git/GNN-pruning/CIFAR10_models/resnet50_retrained.pt'

    # define the datasets
    data_train = datasets_train.get(dataset_name)
    data_test = datasets_test.get(dataset_name)

    # define the data loaders
    data_train_loader = DataLoader(data_train, batch_size=train_bs, shuffle=True, num_workers=8)
    data_test_loader  = DataLoader(data_test, batch_size=test_bs, num_workers=8)

    # initialize the network and what else is necessary for training
    net = networks_data.get(network_name).get('network')()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=2e-3)
    # lr = 0.1
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # train and evaluate the network
    for e in range(1, num_epochs+1):
        train(e, net, criterion, optimizer, data_train_loader, len(data_train))
        accuracy = test(net, criterion, data_test_loader, len(data_test))
        # if e % 75 == 0:
        #     lr = lr/10
        #     optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    # save the state dict and the accuracy
    torch.save(net.state_dict(), save_path)
    acc_path = save_path.replace('.pt', '_accuracy.pt')
    torch.save(accuracy, acc_path)


if __name__ == "__main__":
    main()
