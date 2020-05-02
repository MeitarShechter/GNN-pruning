import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torch.nn import Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
# from torch_geometric.nn import TopKPooling, global_mean_pool as gap, global_max_pool as gmp

'''
This file:
    - defines the modules (customized layers and networks) that are in use:
        Layers:
            - SAGEConv 
            - ElemWiseMultiply 
        Networks:
            - LeNet5
            - Masked_LeNet5
            - Pruned_LeNet5
            - resnet50
            - masked_resnet50
            - pruned_resnet50
            - GNNPrunningNet 
'''

##################
### Functions ####
##################

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

##################
##### Layers #####
##################

class SAGEConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(SAGEConv, self).__init__(aggr='max') #  "Max" aggregation.
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.act = torch.nn.ReLU()
        self.update_lin = torch.nn.Linear(in_channels + out_channels, out_channels, bias=False)
        # self.update_act = torch.nn.ReLU()
        
    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        
        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j):
        # x_j has shape [E, in_channels]

        x_j = self.lin(x_j)
        x_j = self.act(x_j)
        
        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]

        new_embedding = torch.cat([aggr_out, x], dim=1)
        
        new_embedding = self.update_lin(new_embedding)
        # new_embedding = self.update_act(new_embedding)
        
        return new_embedding


class ElemWiseMultiply(nn.Module):
    def __init__(self, mask, w, h, flatten=False):
        super(ElemWiseMultiply, self).__init__()
        self.mask = mask.repeat_interleave(torch.tensor(len(mask)*[h*w]), 0).view(len(mask), h, w)
        if flatten:
            self.mask = self.mask.view(-1)

    def forward(self, x):
        if self.mask.ndim < 4:
            self.mask = self.mask.unsqueeze(0)
        mask = self.mask.repeat_interleave(torch.tensor(x.shape[0]), 0).view(x.shape)
        return x * mask


####################
#### Components ####
####################

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Masked_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, mask, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, exp_s=32):
        super(Masked_Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.mask1 = ElemWiseMultiply(mask[:width], exp_s, exp_s)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.mask2 = ElemWiseMultiply(mask[width:2*width], exp_s/stride, exp_s/stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.mask3 = ElemWiseMultiply(mask[2*width:2*width+planes*self.expansion], exp_s/stride, exp_s/stride)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.mask1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.mask2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.mask3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Pruned_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, mask, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Masked_Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        conv1_numFilters = np.sum(mask[:width])
        conv2_numFilters = np.sum(mask[width:2*width])
        conv3_numFilters = np.sum(mask[2*width:2*width+planes*self.expansion])

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, conv1_numFilters)
        self.bn1 = norm_layer(conv1_numFilters)
        self.conv2 = conv3x3(conv1_numFilters, conv2_numFilters, stride, groups, dilation)
        self.bn2 = norm_layer(conv2_numFilters)
        self.conv3 = conv1x1(conv2_numFilters, conv3_numFilters)
        self.bn3 = norm_layer(conv3_numFilters)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        padd = identity.size(1) - out.size(1)
        if padd < 0:
            identity = F.pad(input=identity, pad=(0,0,0,0,0, -padd), mode='constant', value=0)
        elif padd > 0:
            out = F.pad(input=out, pad=(0,0,0,0,0, padd), mode='constant', value=0)

        out += identity
        out = self.relu(out)

        return out

####################
##### Networks #####
####################

# LeNet 5
"""
Input - 1x32x32
C1 - 6@28x28 (5x5 kernel)
tanh
S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
C3 - 16@10x10 (5x5 kernel, complicated shit)
tanh
S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
C5 - 120@1x1 (5x5 kernel)
F6 - 84
tanh
F7 - 10 (Output)
"""
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output


class Masked_LeNet5(nn.Module):
    def __init__(self, mask):
        super(Masked_LeNet5, self).__init__()

        self.c1_numFilters = 6
        self.c3_numFilters = 16
        self.c5_numFilters = 120
        self.f6_numNeurons = 84

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, self.c1_numFilters, kernel_size=(5, 5))),
            ('m1', ElemWiseMultiply(mask[:6], 28, 28)),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(self.c1_numFilters, self.c3_numFilters, kernel_size=(5, 5))),
            ('m3', ElemWiseMultiply(mask[6:6+16], 10, 10)),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(self.c3_numFilters, self.c5_numFilters, kernel_size=(5, 5))),
            ('m5', ElemWiseMultiply(mask[6+16:6+16+120], 1, 1)),
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(self.c5_numFilters, self.f6_numNeurons)),
            ('m6', ElemWiseMultiply(mask[6+16+120:6+16+120+84], 1, 1, flatten=True)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(self.f6_numNeurons, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output


class Pruned_LeNet5(nn.Module):
    def __init__(self, mask):
        super(Pruned_LeNet5, self).__init__()

        self.c1_numFilters = np.sum(mask[:6])
        self.c3_numFilters = np.sum(mask[6:6+16])
        self.c5_numFilters = np.sum(mask[6+16:6+16+120])
        self.f6_numNeurons = np.sum(mask[6+16+120:6+16+120+84])

        print("c1 = {}, c3 = {}, c5 = {}, f6 = {}".format(self.c1_numFilters, self.c3_numFilters, self.c5_numFilters, self.f6_numNeurons))
        print("Keeped {}/{} params".format(self.c1_numFilters+self.c3_numFilters+self.c5_numFilters+self.f6_numNeurons,6+16+120+84))

        if self.c1_numFilters == 0 or self.c3_numFilters == 0 or self.c5_numFilters == 0 or self.f6_numNeurons == 0:
            print("WE NEED TO PRUNE AN ENTIRE LAYER !!! TODO: NEED TO FIX") 

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, self.c1_numFilters, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(self.c1_numFilters, self.c3_numFilters, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(self.c3_numFilters, self.c5_numFilters, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(self.c5_numFilters, self.f6_numNeurons)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(self.f6_numNeurons, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output


# GNN network
class GNNPrunningNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GNNPrunningNet, self).__init__()

        self.conv1 = SAGEConv(in_channels, out_channels)
        # self.pool1 = TopKPooling(out_channels, ratio=0.8)
        self.conv2 = SAGEConv(out_channels, out_channels)
        # self.pool2 = TopKPooling(out_channels, ratio=0.8)
        self.conv3 = SAGEConv(out_channels, out_channels)
        # self.pool3 = TopKPooling(out_channels, ratio=0.8)

        # self.lin1 = torch.nn.Linear(256, 128)
        # self.lin2 = torch.nn.Linear(128, 64)
        # self.lin3 = torch.nn.Linear(64, 1)
        # self.act1 = torch.nn.ReLU()
        # self.act2 = torch.nn.ReLU()    
        self.conv4 = SAGEConv(out_channels, 1)    
  
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.squeeze(1)        

        x = F.relu(self.conv1(x, edge_index))
        # x = self.conv1(x, edge_index)

        # x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        # x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        # x = self.conv2(x, edge_index)

     
        # x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        # x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))

        # x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        # x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        # x = x1 + x2 + x3

        # x = self.lin1(x)
        # x = self.act1(x)
        # x = self.lin2(x)
        # x = self.act2(x)      
        # x = F.dropout(x, p=0.5, training=self.training)

        # x = torch.sigmoid(self.lin3(x)).squeeze(1)
        x = torch.sigmoid(self.conv4(x, edge_index)).squeeze(1)

        return x


# ResNet50
class ResNet(nn.Module):
    
    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        ## CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        ## END
        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                # elif isinstance(m, BasicBlock):
                #     nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


class Masked_ResNet(nn.Module):
    
    def __init__(self, block, layers, mask, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(Masked_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        
        ## CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.mask1 = ElemWiseMultiply(mask[:self.inplanes], 32, 32)
        ## END
        
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layer1_mask_idx_s = self.inplanes
        width = int(64 * (self.base_width / 64.)) * self.groups
        block_size = 2 * width + 64 * block.expansion
        layer1_mask_idx_t = 64 * block.expansion + block_size
        for _ in range(1, layers[0]):
            layer1_mask_idx_t = layer1_mask_idx_t + block_size

        layer2_mask_idx_s = layer1_mask_idx_s + layer1_mask_idx_t 
        width = int(128 * (self.base_width / 64.)) * self.groups
        block_size = 2 * width + 128 * block.expansion
        layer2_mask_idx_t = 128 * block.expansion + block_size
        for _ in range(1, layers[1]):
            layer2_mask_idx_t = layer2_mask_idx_t + block_size

        layer3_mask_idx_s = layer2_mask_idx_s + layer2_mask_idx_t 
        width = int(256 * (self.base_width / 64.)) * self.groups
        block_size = 2 * width + 256 * block.expansion
        layer3_mask_idx_t = 256 * block.expansion + block_size
        for _ in range(1, layers[2]):
            layer3_mask_idx_t = layer3_mask_idx_t + block_size

        layer4_mask_idx_s = layer3_mask_idx_s + layer3_mask_idx_t 
        width = int(512 * (self.base_width / 64.)) * self.groups
        block_size = 2 * width + 512 * block.expansion
        layer4_mask_idx_t = 512 * block.expansion + block_size
        for _ in range(1, layers[3]):
            layer4_mask_idx_t = layer4_mask_idx_t + block_size

        self.layer1 = self._make_masked_layer(block, 64, layers[0], mask=mask[layer1_mask_idx_s:layer1_mask_idx_s+layer1_mask_idx_t],
                                        exp_s=16)
        self.layer2 = self._make_masked_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], 
                                        mask=mask[layer2_mask_idx_s:layer2_mask_idx_s+layer2_mask_idx_t], exp_s=16)
        self.layer3 = self._make_masked_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], 
                                        mask=mask[layer3_mask_idx_s:layer3_mask_idx_s+layer3_mask_idx_t], exp_s=8)
        self.layer4 = self._make_masked_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], 
                                        mask=mask[layer4_mask_idx_s:layer4_mask_idx_s+layer4_mask_idx_t], exp_s= 4)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                # elif isinstance(m, BasicBlock):
                #     nn.init.constant_(m.bn2.weight, 0)

    def _make_masked_layer(self, block, planes, blocks, stride=1, dilate=False, mask=None, exp_s=32):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
                ElemWiseMultiply(mask[:planes * block.expansion], exp_s/stride, exp_s/stride)
            )

        layers = []
        mask_s = planes * block.expansion
        width = int(planes * (self.base_width / 64.)) * self.groups
        mask_t = 2 * width + planes * block.expansion
        layers.append(block(self.inplanes, planes, mask[mask_s:mask_s+mask_t], stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, exp_s=exp_s))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            mask_s = mask_s + mask_t
            width = int(planes * (self.base_width / 64.)) * self.groups
            mask_t = 2 * width + planes * block.expansion
            layers.append(block(self.inplanes, planes, mask[mask_s:mask_s+mask_t], groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, exp_s=exp_s/stride))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.mask1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


class Pruned_ResNet(nn.Module):
    
    def __init__(self, block, layers, mask, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(Masked_ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1_numFilters = np.sum(mask[:self.inplanes])
        
        ## CIFAR10: kernel_size 7 -> 3, stride 2 -> 1, padding 3->1
        self.conv1 = nn.Conv2d(3, self.conv1_numFilters, kernel_size=3, stride=1, padding=1, bias=False)
        ## END
        
        self.bn1 = norm_layer(self.conv1_numFilters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        layer1_mask_idx_s = self.inplanes
        width = int(64 * (self.base_width / 64.)) * self.groups
        block_size = 2 * width + 64 * block.expansion
        layer1_mask_idx_t = 64 * block.expansion + block_size
        for _ in range(1, layers[0]):
            layer1_mask_idx_t = layer1_mask_idx_t + block_size

        layer2_mask_idx_s = layer1_mask_idx_s + layer1_mask_idx_t 
        width = int(128 * (self.base_width / 64.)) * self.groups
        block_size = 2 * width + 128 * block.expansion
        layer2_mask_idx_t = 128 * block.expansion + block_size
        for _ in range(1, layers[1]):
            layer2_mask_idx_t = layer2_mask_idx_t + block_size

        layer3_mask_idx_s = layer2_mask_idx_s + layer2_mask_idx_t 
        width = int(256 * (self.base_width / 64.)) * self.groups
        block_size = 2 * width + 256 * block.expansion
        layer3_mask_idx_t = 256 * block.expansion + block_size
        for _ in range(1, layers[2]):
            layer3_mask_idx_t = layer3_mask_idx_t + block_size

        layer4_mask_idx_s = layer3_mask_idx_s + layer3_mask_idx_t 
        width = int(512 * (self.base_width / 64.)) * self.groups
        block_size = 2 * width + 512 * block.expansion
        layer4_mask_idx_t = 512 * block.expansion + block_size
        for _ in range(1, layers[3]):
            layer4_mask_idx_t = layer4_mask_idx_t + block_size

        self.layer1 = self._make_pruned_layer(block, 64, layers[0], mask=mask[layer1_mask_idx_s:layer1_mask_idx_s+layer1_mask_idx_t])
        self.layer2 = self._make_pruned_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], 
                                        mask=mask[layer2_mask_idx_s:layer2_mask_idx_s+layer2_mask_idx_t])
        self.layer3 = self._make_pruned_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], 
                                        mask=mask[layer3_mask_idx_s:layer3_mask_idx_s+layer3_mask_idx_t])
        self.layer4 = self._make_pruned_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], 
                                        mask=mask[layer4_mask_idx_s:layer4_mask_idx_s+layer4_mask_idx_t])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                # elif isinstance(m, BasicBlock):
                #     nn.init.constant_(m.bn2.weight, 0)

    def _make_pruned_layer(self, block, planes, blocks, stride=1, dilate=False, mask=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        ds_numFilters = np.sum(mask[:planes * block.expansion])

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.conv1_numFilters, ds_numFilters, stride),
                norm_layer(conv1_numFilters),
            )

        layers = []
        mask_s = planes * block.expansion
        width = int(planes * (self.base_width / 64.)) * self.groups
        mask_t = 2 * width + planes * block.expansion
        layers.append(block(self.conv1_numFilters, planes, mask[mask_s:mask_s+mask_t], stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, exp_s=exp_s))

        conv3_numFilters = np.sum(mask[mask_s+2*width:mask_s+2*width+planes*block.expansion])
        end_layer_numFilters = conv3_numFilters
        if ds_numFilters > conv3_numFilters:
            end_layer_numFilters = ds_numFilters

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            mask_s = mask_s + mask_t
            width = int(planes * (self.base_width / 64.)) * self.groups
            mask_t = 2 * width + planes * block.expansion
            layers.append(block(end_layer_numFilters, planes, mask[mask_s:mask_s+mask_t], groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, exp_s=exp_s/stride))

            conv3_numFilters = np.sum(mask[mask_s+2*width:mask_s+2*width+planes*block.expansion])
            end_layer_numFilters = conv3_numFilters
            if ds_numFilters > conv3_numFilters:
                end_layer_numFilters = ds_numFilters
        
        self.conv1_numFilters = end_layer_numFilters

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        return x


def _resnet(block, layers, device):
    model = ResNet(block, layers)
    return model

def _masked_resnet(block, layers, mask, device):
    model = Masked_ResNet(block, layers, mask)
    return model

def _pruned_resnet(block, layers, mask, device):
    model = Pruned_ResNet(block, layers, mask)
    return model

def resnet50(device='cpu'):
    return _resnet(Bottleneck, [3, 4, 6, 3], device)

def masked_resnet50(mask, device='cpu'):
    return _masked_resnet(Masked_Bottleneck, [3, 4, 6, 3], mask, device)

def pruned_resnet50(mask, device='cpu'):
    return _pruned_resnet(Pruned_Bottleneck, [3, 4, 6, 3], mask, device)



# # ResNeXt
# class Block(nn.Module):
#     '''Grouped convolution block.'''
#     expansion = 2

#     def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
#         super(Block, self).__init__()
#         group_width = cardinality * bottleneck_width
#         self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(group_width)
#         self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
#         self.bn2 = nn.BatchNorm2d(group_width)
#         self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(self.expansion*group_width)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*group_width:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*group_width)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class ResNeXt(nn.Module):
#     def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=10):
#         super(ResNeXt, self).__init__()
#         self.cardinality = cardinality
#         self.bottleneck_width = bottleneck_width
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(num_blocks[0], 1)
#         self.layer2 = self._make_layer(num_blocks[1], 2)
#         self.layer3 = self._make_layer(num_blocks[2], 2)
#         # self.layer4 = self._make_layer(num_blocks[3], 2)
#         self.linear = nn.Linear(cardinality*bottleneck_width*8, num_classes)

#     def _make_layer(self, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)
#         layers = []
#         for stride in strides:
#             layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
#             self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
#         # Increase bottleneck_width by 2 after each stage.
#         self.bottleneck_width *= 2
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         # out = self.layer4(out)
#         out = F.avg_pool2d(out, 8)
#         out = out.view(out.size(0), -1)
#         out = self.linear(out)
#         return out

# def ResNeXt29_2x64d():
#     return ResNeXt(num_blocks=[3,3,3], cardinality=2, bottleneck_width=64)

