import torch
import torch.nn as nn
import torch.nn.functional as F

def smooth_crossentropy(pred, gold, smoothing=0.1):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    log_prob = F.log_softmax(pred, dim=1)
    
    out = F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)
    return out


def unflatten(gradient, parameters):
    """Change the shape of the gradient to the shape of the parameters
        Parameters:
            gradient: flattened gradient
            parameters: convert the flattened gradient to the unflattened
                        version
            tensor: convert to tonsor otherwise it will be an array
    """
    shaped_gradient = []
    begin = 0
    for layer in parameters:
        size = layer.view(-1).shape[0]
        shaped_gradient.append(
            gradient[begin:begin+size].view(layer.shape))
        begin += size
        
    return shaped_gradient