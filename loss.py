import torch
import torch.nn as nn

from users_definitions import networks_data

'''
This file:
    - defines the loss function
'''


#########################
### Building the loss ###
#########################

def GNN_prune_loss(params_scores, orig_net, network_name, val_data, reduction=None, prune_factor=0.5, gamma1=3e5, gamma2=1, gamma3=1e2):
    prune_th = 0.5 # prune_th, _ = torch.kthvalue(params_scores, torch.ceil(prune_factor*params_scores.numel()))
    params_mask = (params_scores >= prune_th)

    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=1024, shuffle=False, num_workers=8) 

    ### build a new network based on params_mask ###
    # new_net = Masked_LeNet5(params_mask) # We need to make the network differentiable as a function of the mask 
    # new_net = Masked_LeNet5(params_scores)
    new_net = networks_data.get(network_name).get('masked_network')(params_scores)

    model_dict = new_net.state_dict()
    pretrained_dict = orig_net.state_dict()    

    model_dict.update(pretrained_dict) 

    # for idx, layer_n in enumerate(pretrained_dict):
    #     if 'bias' in layer_n:
    #         continue

    #     if layer_n == 'convnet.c1.weight':
    #         indices = np.where(params_mask[:6])[0]
    #     elif layer_n == 'convnet.c3.weight':
    #         indices = np.where(params_mask[6:6+16])[0] 
    #     elif layer_n == 'convnet.c5.weight':
    #         indices = np.where(params_mask[6+16:6+16+120])[0] 
    #     elif layer_n == 'fc.f6.weight':
    #         indices = np.where(params_mask[6+16+120:6+16+120+84])[0] 
    #     elif layer_n == 'fc.f7.weight':
    #         indices = list(range(0, 11))
        
    #     model_dict[layer_n] = pretrained_dict[layer_n][idices,:,:]

    #     bias_n = layer_n.replace('weight', 'bias')
    #     model_dict[bias_n] = pretrained_dict[bias_n][indices] 

    new_net.load_state_dict(model_dict)



    ### calculate the new network's loss ###
    new_net.eval()
    total_correct = 0
    avg_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for i, (images, labels) in enumerate(val_data_loader):
        output = new_net(images)
        avg_loss += criterion(output, labels).sum()
        pred = output.detach().max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(val_data)
    new_net_loss = avg_loss

    ### calcualte original model loss ###
    orig_net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_data_loader):
            output = orig_net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.detach().max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()

    avg_loss /= len(val_data)
    orig_net_loss = avg_loss

    ### calculate all of the loss terms
    loss_diff = new_net_loss - orig_net_loss

    data_term = torch.mean(loss_diff * torch.abs(params_scores - 0.5))
    sparse_term = torch.norm(params_scores, 1)
    confidence_term = torch.mean(0.5 - torch.abs(params_scores - 0.5))

    print("The loss_diff is: {}".format(loss_diff))
    print("The data_term in the loss is: {}".format(gamma1*data_term))
    print("The sparse_term in the loss is: {}".format(gamma2*sparse_term))
    print("The confidence_term in the loss is: {}".format(gamma3*confidence_term))

    loss = gamma1 * data_term + gamma2 * sparse_term + gamma3 * confidence_term

    return loss
