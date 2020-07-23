import numpy as np

##################
##### LeNet5 #####
##################

def LeNet5CreateNameGraph(graph):
    ## nodes
    graph.add_node('convnet.c1.weight')
    graph.add_node('convnet.c3.weight')
    graph.add_node('convnet.c5.weight')
    graph.add_node('fc.f6.weight')
    graph.add_node('fc.f7.weight')
    ## edges
    graph.add_edge('convnet.c1.weight', 'convnet.c3.weight')
    graph.add_edge('convnet.c3.weight', 'convnet.c5.weight')
    graph.add_edge('convnet.c5.weight', 'fc.f6.weight')
    graph.add_edge('fc.f6.weight', 'fc.f7.weight')

    return graph


def LeNet5UpdatePrunedNet(new_dict, pretrained_dict, mask):
    layer_n ='convnet.c1.weight'
    bias_n = layer_n.replace('weight', 'bias')    
    indices = np.where(mask[:6])[0]
    new_dict[layer_n] = pretrained_dict[layer_n][indices,:,:,:]
    prev_indices = indices
    new_dict[bias_n] = pretrained_dict[bias_n][indices] 

    layer_n = 'convnet.c3.weight'
    bias_n = layer_n.replace('weight', 'bias')
    indices = np.where(mask[6:6+16])[0] 
    relevant_filters = pretrained_dict[layer_n][indices,:,:,:] 
    new_dict[layer_n] = relevant_filters[:, prev_indices,:,:]
    prev_indices = indices
    new_dict[bias_n] = pretrained_dict[bias_n][indices] 

    layer_n = 'convnet.c5.weight'
    bias_n = layer_n.replace('weight', 'bias')
    indices = np.where(mask[6+16:6+16+120])[0] 
    relevant_filters = pretrained_dict[layer_n][indices,:,:,:] 
    new_dict[layer_n] = relevant_filters[:, prev_indices,:,:]
    prev_indices = indices
    new_dict[bias_n] = pretrained_dict[bias_n][indices] 

    layer_n = 'fc.f6.weight'
    bias_n = layer_n.replace('weight', 'bias')
    indices = np.where(mask[6+16+120:6+16+120+84])[0] 
    relevant_filters = pretrained_dict[layer_n][indices,:] 
    new_dict[layer_n] = relevant_filters[:, prev_indices]
    prev_indices = indices
    new_dict[bias_n] = pretrained_dict[bias_n][indices] 

    layer_n = 'fc.f7.weight'
    bias_n = layer_n.replace('weight', 'bias')
    indices = list(range(0, 10))
    relevant_filters = pretrained_dict[layer_n][indices,:] 
    new_dict[layer_n] = relevant_filters[:, prev_indices]
    new_dict[bias_n] = pretrained_dict[bias_n][indices] 

    return new_dict


##################
#### resnet50 ####
##################

def resnset50CreateNameGraph(graph):
    ### add graph head ###

    ## nodes
    graph.add_node('conv1.weight')
    # graph.add_node('bn1') # indicator that there's batch norm
    ## edges
    # graph.add_edge('conv1.weight', 'bn1')


    ### add graph body ###

    num_of_blocks = [3, 4, 6, 3]
    ## nodes
    for layer_idx, num_blocks in enumerate(num_of_blocks):
        for i in range(num_blocks):
            graph.add_node('layer' + str(layer_idx+1) + '.' + str(i) + '.conv1.weight')
            # graph.add_node('layer' + str(layer_idx+1) + '.' + str(i) + '.bn1')
            graph.add_node('layer' + str(layer_idx+1) + '.' + str(i) + '.conv2.weight')
            # graph.add_node('layer' + str(layer_idx+1) + '.' + str(i) + '.bn2')
            graph.add_node('layer' + str(layer_idx+1) + '.' + str(i) + '.conv3.weight')
            # graph.add_node('layer' + str(layer_idx+1) + '.' + str(i) + '.bn3')
            if i == 0:
                # add downsample
                graph.add_node('layer' + str(layer_idx+1) + '.' + str(i) + '.downsample.0.weight')
                # graph.add_node('layer' + str(layer_idx+1) + '.' + str(i) + '.downsample.1.unnamed_bn')
    ## edges
    # connect head to body
    # graph.add_edge('bn1', 'layer1.0.conv1.weight')
    graph.add_edge('conv1.weight', 'layer1.0.conv1.weight')
    # graph.add_edge('bn1', 'layer1.0.downsample.0.weight')
    graph.add_edge('conv1.weight', 'layer1.0.downsample.0.weight')
    # connections within body
    for layer_idx, num_blocks in enumerate(num_of_blocks):
        for i in range(num_blocks):
            prefix      = 'layer' + str(layer_idx+1) + '.' + str(i) 
            conv1_name  = prefix + '.conv1.weight'
            # bn1_name    = prefix + '.bn1'
            conv2_name  = prefix + '.conv2.weight'
            # bn2_name    = prefix + '.bn2'
            conv3_name  = prefix + '.conv3.weight'
            # bn3_name    = prefix + '.bn3'

            # connect between blocks
            if i > 0:
                # downsample_name = 'layer' + str(layer_idx+1) + '.0.downsample.1.unnamed_bn'
                downsample_name = 'layer' + str(layer_idx+1) + '.0.downsample.0.weight'
                graph.add_edge(downsample_name, conv1_name)
                for j in range(i):
                    prev_prefix     = 'layer' + str(layer_idx+1) + '.' + str(j)
                    # prev_bn3_name   = prev_prefix + '.bn3'
                    prev_conv3_name   = prev_prefix + '.conv3.weight'
                    # graph.add_edge(prev_bn3_name, conv1_name)
                    graph.add_edge(prev_conv3_name, conv1_name)

            # graph.add_edge(conv1_name, bn1_name)
            # graph.add_edge(bn1_name, conv2_name)
            graph.add_edge(conv1_name, conv2_name)
            # graph.add_edge(conv2_name, bn2_name)
            # graph.add_edge(bn2_name, conv3_name)
            graph.add_edge(conv2_name, conv3_name)
            # graph.add_edge(conv3_name, bn3_name)
            # if i == 0:
                # connect downsample
                # graph.add_edge('layer' + str(layer_idx+1) + '.' + str(i) + '.downsample.0.weight', 'layer' + str(layer_idx+1) + '.' + str(i) + '.downsample.1.unnamed_bn')
        # connect between layers
        if layer_idx > 0:
            # unnamed_bn_name = 'layer' + str(layer_idx) + '.0.downsample.1.unnamed_bn' 
            ds_conv_name = 'layer' + str(layer_idx) + '.0.downsample.0.weight' 
            # graph.add_edge(unnamed_bn_name, 'layer' + str(layer_idx+1) + '.0.conv1.weight')
            # graph.add_edge(unnamed_bn_name, 'layer' + str(layer_idx+1) + '.0.downsample.0.weight')
            graph.add_edge(ds_conv_name, 'layer' + str(layer_idx+1) + '.0.conv1.weight')
            graph.add_edge(ds_conv_name, 'layer' + str(layer_idx+1) + '.0.downsample.0.weight')
            for i in range(num_of_blocks[layer_idx-1]):
                prefix = 'layer' + str(layer_idx) + '.' + str(i)
                # bn3_name = prefix + '.bn3'
                conv3_name = prefix + '.conv3.weight'
                # graph.add_edge(bn3_name, 'layer' + str(layer_idx+1) + '.0.conv1.weight')
                # graph.add_edge(bn3_name, 'layer' + str(layer_idx+1) + '.0.downsample.0.weight')
                graph.add_edge(conv3_name, 'layer' + str(layer_idx+1) + '.0.conv1.weight')
                graph.add_edge(conv3_name, 'layer' + str(layer_idx+1) + '.0.downsample.0.weight')
                

    ### add graph tail ###

    ## nodes
    graph.add_node('fc.weight')
    ## edges
    # connect body to tail
    # unnamed_bn_name = 'layer' + str(len(num_of_blocks)) + '.0.downsample.1.unnamed_bn' 
    ds_conv_name = 'layer' + str(len(num_of_blocks)) + '.0.downsample.0.weight' 
    # graph.add_edge(unnamed_bn_name, 'fc.weight')
    graph.add_edge(ds_conv_name, 'fc.weight')
    for i in range(num_of_blocks[-1]):
        prefix = 'layer' + str(len(num_of_blocks)) + '.' + str(i)
        # bn3_name = prefix + '.bn3'
        conv3_name = prefix + '.conv3.weight'
        # graph.add_edge(bn3_name, 'fc.weight')
        graph.add_edge(conv3_name, 'fc.weight')

    return graph



def resnet50UpdatePrunedNet(new_dict, pretrained_dict, mask):
    # network head
    layer_n = 'conv1.weight'
    indices = np.where(mask[:64])[0]
    new_dict[layer_n] = pretrained_dict[layer_n][indices,:,:,:]
    prev_indices = indices

    layer_n = 'bn1.weight'
    bias_n = layer_n.replace('weight', 'bias')
    rm_n = layer_n.replace('weight', 'running_mean')
    rv_n = layer_n.replace('weight', 'running_var')
    new_dict[layer_n] = pretrained_dict[layer_n][indices]
    new_dict[bias_n] = pretrained_dict[bias_n][indices] 
    new_dict[rm_n] = pretrained_dict[rm_n][indices]
    new_dict[rv_n] = pretrained_dict[rv_n][indices]

    # network body
    planes = [64, 128, 256, 512]
    n_blocks = [3, 4, 6, 3]
    m_s = 64
    prev_layer_s = 0
    # set each layer
    for k in range(1,5):
        width = planes[k-1]
        block_size = 2*width + 4*planes[k-1]
        layer_s = planes[k-1]*4 + n_blocks[k-1]*block_size

        # downsample first
        m_s = m_s + prev_layer_s
        m_e = m_s + 4*planes[k-1]

        layer_n = 'layer{}.0.downsample.0.weight'.format(k)
        ds_indices = np.where(mask[m_s:m_e])[0] 
        relevant_filters = pretrained_dict[layer_n][ds_indices,:,:,:] 
        new_dict[layer_n] = relevant_filters[:, prev_indices,:,:]

        layer_n = 'layer{}.0.downsample.1.weight'.format(k)
        bias_n = layer_n.replace('weight', 'bias')
        rm_n = layer_n.replace('weight', 'running_mean')
        rv_n = layer_n.replace('weight', 'running_var')
        new_dict[bias_n] = pretrained_dict[bias_n][ds_indices] 
        new_dict[rm_n] = pretrained_dict[rm_n][ds_indices]
        new_dict[rv_n] = pretrained_dict[rv_n][ds_indices]

        prev_ds_indices = ds_indices
        # bottleneck second
        m_s = m_e
        # set each block
        for i in range(n_blocks[k-1]):
            m_s = m_s + i*block_size
            # set each layer in each block
            for j in range(1,4):
                layer_n = 'layer{}.{}.conv{}.weight'.format(k, i, j)
                m_s = m_s + width*(j-1)
                if j == 3:
                    m_e = m_s + 4*planes[k-1]
                else:
                    m_e = m_s + width

                indices = np.where(mask[m_s:m_e])[0] 
                relevant_filters = pretrained_dict[layer_n][indices,:,:,:] 
                new_dict[layer_n] = relevant_filters[:, prev_indices,:,:]

                layer_n = 'layer{}.{}.bn{}.weight'.format(k, i, j)
                bias_n = layer_n.replace('weight', 'bias')
                rm_n = layer_n.replace('weight', 'running_mean')
                rv_n = layer_n.replace('weight', 'running_var')
                new_dict[bias_n] = pretrained_dict[bias_n][indices] 
                new_dict[rm_n] = pretrained_dict[rm_n][indices]
                new_dict[rv_n] = pretrained_dict[rv_n][indices]

                prev_indices = indices
        
            # keep the indices where we have more filters
            if len(prev_indices) < len(prev_ds_indices):
                prev_indices = prev_ds_indices

        prev_layer_s = layer_s
    
    # network tail
    layer_n = 'fc.weight'
    bias_n = layer_n.replace('weight', 'bias')
    new_dict[layer_n] = pretrained_dict[layer_n]
    new_dict[bias_n] = pretrained_dict[bias_n]

    return new_dic




##################
#### WRN_40_2 ####
##################

def WRN_40_2CreateNameGraph(graph):
    ### add graph head ###

    ## nodes
    graph.add_node('conv1.weight')
    # graph.add_node('bn1') # indicator that there's batch norm
    ## edges
    # graph.add_edge('conv1.weight', 'bn1')


    ### add graph body ###

    num_of_blocks = [6, 6, 6]
    ## nodes
    for layer_idx, num_blocks in enumerate(num_of_blocks):
        for i in range(num_blocks):
            graph.add_node('layer' + str(layer_idx+1) + '.' + str(i) + '.conv1.weight')
            # graph.add_node('layer' + str(layer_idx+1) + '.' + str(i) + '.bn1')
            graph.add_node('layer' + str(layer_idx+1) + '.' + str(i) + '.conv2.weight')
            # graph.add_node('layer' + str(layer_idx+1) + '.' + str(i) + '.bn2')
            if i == 0:
                # add downsample
                graph.add_node('layer' + str(layer_idx+1) + '.' + str(i) + '.shortcut.0.weight')
                # graph.add_node('layer' + str(layer_idx+1) + '.' + str(i) + '.downsample.1.unnamed_bn')
    ## edges
    # connect head to body
    # graph.add_edge('bn1', 'layer1.0.conv1.weight')
    graph.add_edge('conv1.weight', 'layer1.0.conv1.weight')
    # graph.add_edge('bn1', 'layer1.0.downsample.0.weight')
    graph.add_edge('conv1.weight', 'layer1.0.shortcut.0.weight')
    # connections within body
    for layer_idx, num_blocks in enumerate(num_of_blocks):
        # connect between layers
        if layer_idx > 0:
            # unnamed_bn_name = 'layer' + str(layer_idx) + '.0.downsample.1.unnamed_bn' 
            ds_conv_name = 'layer' + str(layer_idx) + '.0.shortcut.0.weight' 
            # graph.add_edge(unnamed_bn_name, 'layer' + str(layer_idx+1) + '.0.conv1.weight')
            # graph.add_edge(unnamed_bn_name, 'layer' + str(layer_idx+1) + '.0.downsample.0.weight')
            graph.add_edge(ds_conv_name, 'layer' + str(layer_idx+1) + '.0.conv1.weight')
            graph.add_edge(ds_conv_name, 'layer' + str(layer_idx+1) + '.0.shortcut.0.weight')
            for i in range(num_of_blocks[layer_idx-1]):
                prefix = 'layer' + str(layer_idx) + '.' + str(i)
                # bn2_name = prefix + '.bn2'
                conv2_name = prefix + '.conv2.weight'
                # graph.add_edge(bn2_name, 'layer' + str(layer_idx+1) + '.0.conv1.weight')
                # graph.add_edge(bn2_name, 'layer' + str(layer_idx+1) + '.0.downsample.0.weight')
                graph.add_edge(conv2_name, 'layer' + str(layer_idx+1) + '.0.conv1.weight')
                graph.add_edge(conv2_name, 'layer' + str(layer_idx+1) + '.0.shortcut.0.weight')

        for i in range(num_blocks):
            prefix      = 'layer' + str(layer_idx+1) + '.' + str(i) 
            conv1_name  = prefix + '.conv1.weight'
            # bn1_name    = prefix + '.bn1'
            conv2_name  = prefix + '.conv2.weight'
            # bn2_name    = prefix + '.bn2'

            # connect between blocks
            if i > 0:
                # downsample_name = 'layer' + str(layer_idx+1) + '.0.downsample.1.unnamed_bn'
                downsample_name = 'layer' + str(layer_idx+1) + '.0.shortcut.0.weight'
                graph.add_edge(downsample_name, conv1_name)
                for j in range(i):
                    prev_prefix     = 'layer' + str(layer_idx+1) + '.' + str(j)
                    # prev_bn2_name   = prev_prefix + '.bn2'
                    prev_conv2_name   = prev_prefix + '.conv2.weight'
                    # graph.add_edge(prev_bn3_name, conv1_name)
                    graph.add_edge(prev_conv2_name, conv1_name)

            # graph.add_edge(conv1_name, bn1_name)
            # graph.add_edge(bn1_name, conv2_name)
            graph.add_edge(conv1_name, conv2_name)
            # graph.add_edge(conv2_name, bn2_name)
            # if i == 0:
                # connect downsample
                # graph.add_edge('layer' + str(layer_idx+1) + '.' + str(i) + '.downsample.0.weight', 'layer' + str(layer_idx+1) + '.' + str(i) + '.downsample.1.unnamed_bn')
        
                

    # ### add graph tail ###

    # ## nodes
    # graph.add_node('linear.weight')
    # ## edges
    # # connect body to tail
    # # unnamed_bn_name = 'layer' + str(len(num_of_blocks)) + '.0.downsample.1.unnamed_bn' 
    # ds_conv_name = 'layer' + str(len(num_of_blocks)) + '.0.shortcut.0.weight' 
    # # graph.add_edge(unnamed_bn_name, 'fc.weight')
    # graph.add_edge(ds_conv_name, 'linear.weight')
    # for i in range(num_of_blocks[-1]):
    #     prefix = 'layer' + str(len(num_of_blocks)) + '.' + str(i)
    #     # bn2_name = prefix + '.bn2'
    #     conv2_name = prefix + '.conv2.weight'
    #     # graph.add_edge(bn2_name, 'linear.weight')
    #     graph.add_edge(conv2_name, 'linear.weight')

    return graph


def WRN_40_2UpdatePrunedNet(new_dict, pretrained_dict, mask):
    # network head

    layer_n = 'conv1.weight'
    indices = np.where(mask[:16])[0]
    if len(indices) == 0:
        indices = [0]
    new_dict[layer_n] = pretrained_dict[layer_n][indices,:,:,:]
    prev_indices = indices
    prev_ds_indices = indices
    next_bn_name = 'layer1.0.bn1.weight'


    # network body

    planes = [32, 64, 128]
    n_blocks = [6, 6, 6]
    m_s = 16
    # set each layer
    for k in range(1,4):
        # set each block
        for i in range(n_blocks[k-1]):
            # set each layer in each block


            # if we have a shortcut and no convs we need to add BN
            if i == 0 and np.sum(mask[m_s:(m_s+2*planes[k-1])]) == 0 and np.sum(mask[(m_s+2*planes[k-1]):(m_s+3*planes[k-1])]) > 0:
                curr_bn_layer_n = 'layer{}.{}.bn{}.weight'.format(k, i, 1)
                curr_bias_n     = curr_bn_layer_n.replace('weight', 'bias')
                curr_rm_n       = curr_bn_layer_n.replace('weight', 'running_mean')
                curr_rv_n       = curr_bn_layer_n.replace('weight', 'running_var')

                layer_n = next_bn_name
                bias_n  = layer_n.replace('weight', 'bias')
                rm_n    = layer_n.replace('weight', 'running_mean')
                rv_n    = layer_n.replace('weight', 'running_var')

                new_dict[curr_bn_layer_n]   = pretrained_dict[layer_n][prev_indices]
                new_dict[curr_bias_n]       = pretrained_dict[bias_n][prev_indices] 
                new_dict[curr_rm_n]         = pretrained_dict[rm_n][prev_indices]
                new_dict[curr_rv_n]         = pretrained_dict[rv_n][prev_indices]

                next_bn_name = 'layer{}.{}.bn{}.weight'.format(k, i, 2)



            for j in range(1,3):                
                m_e = m_s + planes[k-1]
                indices = np.where(mask[m_s:m_e])[0] 

                # only if keeping this layer
                if len(indices) > 0:
                    curr_bn_layer_n = 'layer{}.{}.bn{}.weight'.format(k, i, j)
                    curr_bias_n     = curr_bn_layer_n.replace('weight', 'bias')
                    curr_rm_n       = curr_bn_layer_n.replace('weight', 'running_mean')
                    curr_rv_n       = curr_bn_layer_n.replace('weight', 'running_var')

                    layer_n = next_bn_name
                    bias_n  = layer_n.replace('weight', 'bias')
                    rm_n    = layer_n.replace('weight', 'running_mean')
                    rv_n    = layer_n.replace('weight', 'running_var')

                    new_dict[curr_bn_layer_n]   = pretrained_dict[layer_n][prev_indices]
                    new_dict[curr_bias_n]       = pretrained_dict[bias_n][prev_indices] 
                    new_dict[curr_rm_n]         = pretrained_dict[rm_n][prev_indices]
                    new_dict[curr_rv_n]         = pretrained_dict[rv_n][prev_indices]

                    layer_n = 'layer{}.{}.conv{}.weight'.format(k, i, j)

                    relevant_filters = pretrained_dict[layer_n][indices,:,:,:] 
                    new_dict[layer_n] = relevant_filters[:, prev_indices,:,:]

                    if j == 2:
                        if i == n_blocks[k-1] - 1:
                            if k == 3:
                                next_bn_name = 'bn1.weight'
                            else:
                                next_bn_name = 'layer{}.{}.bn{}.weight'.format(k+1, 0, 1)
                        else:
                            next_bn_name = 'layer{}.{}.bn{}.weight'.format(k, i+1, 1)
                    else:
                        next_bn_name = 'layer{}.{}.bn{}.weight'.format(k, i, j+1)

                    prev_indices = indices
                m_s = m_s + planes[k-1]

            if i == 0:
                # shortcut last
                m_e = m_s + planes[k-1]
                ds_indices = np.where(mask[m_s:m_e])[0] 

                # only if keeping the shortcut
                if len(ds_indices) > 0:
                    layer_n = 'layer{}.0.shortcut.0.weight'.format(k)
                    relevant_filters = pretrained_dict[layer_n][ds_indices,:,:,:] 
                    new_dict[layer_n] = relevant_filters[:, prev_ds_indices,:,:]

                    prev_ds_indices = ds_indices
                m_s = m_s + planes[k-1]
            
            # keep the indices where we have more filters
            if len(prev_indices) < len(prev_ds_indices): 
                prev_indices = prev_ds_indices
            else:
                prev_ds_indices = prev_indices 

    curr_bn_layer_n = 'bn1.weight'
    curr_bias_n     = curr_bn_layer_n.replace('weight', 'bias')
    curr_rm_n       = curr_bn_layer_n.replace('weight', 'running_mean')
    curr_rv_n       = curr_bn_layer_n.replace('weight', 'running_var')
    layer_n = next_bn_name
    bias_n  = layer_n.replace('weight', 'bias')
    rm_n    = layer_n.replace('weight', 'running_mean')
    rv_n    = layer_n.replace('weight', 'running_var')
    # layer_n = 'bn1.weight'
    # bias_n = layer_n.replace('weight', 'bias')
    # rm_n = layer_n.replace('weight', 'running_mean')
    # rv_n = layer_n.replace('weight', 'running_var')
    new_dict[curr_bn_layer_n] = pretrained_dict[layer_n][prev_indices]
    new_dict[curr_bias_n]     = pretrained_dict[bias_n][prev_indices] 
    new_dict[curr_rm_n]       = pretrained_dict[rm_n][prev_indices]
    new_dict[curr_rv_n]       = pretrained_dict[rv_n][prev_indices]
            

    # network tail

    layer_n = 'linear.weight'
    bias_n = layer_n.replace('weight', 'bias')
    new_dict[layer_n] = pretrained_dict[layer_n][:, prev_indices]
    new_dict[bias_n] = pretrained_dict[bias_n]

    return new_dict