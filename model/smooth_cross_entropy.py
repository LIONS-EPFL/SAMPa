import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def smooth_crossentropy(pred, gold, smoothing=0.1, perturb_label=0.0):
    n_class = pred.size(1)

    one_hot = torch.full_like(pred, fill_value=smoothing / (n_class - 1))
    one_hot.scatter_(dim=1, index=gold.unsqueeze(1), value=1.0 - smoothing)
    one_hot += perturb_label
    one_hot = torch.clamp(one_hot, min=0.0, max=1.0)
    one_hot = F.normalize(one_hot, p=2, dim=1)
    log_prob = F.log_softmax(pred, dim=1)
    
    out = F.kl_div(input=log_prob, target=one_hot, reduction='none').sum(-1)
    return out


def perturbation_loss_tanh(pred, targets, mu=10, alpha=0.1):
    for i in range(len(targets)):
        pred[i] = pred[i] - pred[i][targets[i]]
    
    out = torch.logsumexp(mu*torch.tanh(alpha*pred),1)
    return out


def perturbation_loss_log(pred, targets, mu=1, alpha=0.1):
    for i in range(len(targets)):
        pred[i] = pred[i] - pred[i][targets[i]]
    
    out = torch.logsumexp(mu*(1+torch.nn.functional.softplus(pred-torch.log(torch.tensor(math.e-1)), beta=-1)),1)       # Add "1+"
    return out


def perturbation_loss_test(pred, targets, mu=1, alpha=0.1):
    for i in range(len(targets)):
        pred[i] = pred[i] - pred[i][targets[i]]
    
    out = torch.logsumexp(mu*pred,1)
    return out


def correction_loss_log(pred, targets, mu=1, alpha=0.1):
    for i in range(len(targets)):
        pred[i] = pred[i] - pred[i][targets[i]]
    
    out = torch.logsumexp(mu*(torch.nn.functional.softplus(pred+torch.log(torch.tensor(math.e-1)), beta=1)),1)
    return out