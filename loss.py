import torch
import torch.nn as nn
from torch.autograd import Variable

from users_definitions import networks_data
from modules import ElemWiseMultiply

'''
This file:
    - defines the loss function
'''


#########################
### Building the loss ###
#########################

def GNN_prune_loss(params_scores, orig_net, orig_net_loss, network_name, val_data, device, reduction=None, prune_factor=0.5, gamma1=1, gamma2=1, gamma3=1e2):
    ### first calculate the sparse term ###
    # sparse_term = gamma2 * torch.log2((1 / (torch.log2(params_scores.numel()/(torch.sum(params_scores) + 1e-6)) + 1e-6)))
    sparse_term = gamma2 * torch.log2(torch.sum(params_scores) + 1e-6)
    # sparse_term_v = 1/sparse_term.item()

    batch_size = 8
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=8) 
    new_net = networks_data.get(network_name).get('masked_network')(params_scores, device).to(device)

    model_dict = new_net.state_dict()
    pretrained_dict = orig_net.state_dict()    

    model_dict.update(pretrained_dict) 

    new_net.load_state_dict(model_dict)  

    ### train the new network a bit ###
    new_net.train()
    # don't train the masks
    for m in new_net.modules():
        if isinstance(m, ElemWiseMultiply):
            m.mask.requires_grad = False

    optimizer = torch.optim.SGD(new_net.parameters(), lr=0.02, momentum=0.9, nesterov=True, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    for e in range(3):
        for i, (images, labels) in enumerate(val_data_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = new_net(images)
            loss = criterion(output, labels)

            if i % 100 == 0:
                print('Retrain - Epoch %d, Batch: %d, Loss: %f' % (e, i, loss.detach().cpu().item()))
        
            loss.backward()
            optimizer.step()

    for m in new_net.modules():
        if isinstance(m, ElemWiseMultiply):
            m.mask.requires_grad = True

    ### calculate the new network's loss ###
    new_net.eval()
    avg_loss = 0.0
    acc_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    num_b = 0
    acc_data_grad = torch.zeros_like(params_scores)

    for i, (images, labels) in enumerate(val_data_loader):
        images, labels = images.to(device), labels.to(device)
        new_net.zero_grad()
        output = new_net(images)
        loss = criterion(output, labels)

        avg_loss += loss.sum()      
        # loss = gamma1 * torch.log2(loss/orig_net_loss)
        loss = gamma1 * loss
        acc_loss += loss.sum()
        # loss *= sparse_term_v        
        loss.backward()

        # get all the gradients with respect to mask
        indToSum = {}
        for m in new_net.modules():
            if isinstance(m, ElemWiseMultiply):
                nf = m.mask.grad.data.size()[1]
                ind = m.indices_list
                for i in range(nf):
                    s = torch.sum(m.mask.grad.data[:, i, :, :])
                    indToSum[ind[i]] = s
        # add those gradients the rest
        for i in range(len(params_scores)):
            acc_data_grad[i] += indToSum[i]

        num_b += 1
        if num_b == 4000:
          break

    # avg_loss /= len(val_data)
    avg_loss /= (batch_size*num_b)
    acc_loss /= (batch_size*num_b)

    new_net_loss = avg_loss
    data_term = acc_loss
    print("The data_term in the loss is: {}".format(data_term))
    print("The sparse_term in the loss is: {}".format(sparse_term))

    return sparse_term, data_term, acc_data_grad
