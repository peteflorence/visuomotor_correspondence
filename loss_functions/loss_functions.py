from __future__ import print_function
import math

import torch
import torch.nn as nn

criterion_l2 = nn.MSELoss()
criterion_l1 = nn.L1Loss()
cos = nn.CosineSimilarity(dim=1, eps=1e-6)

def l1_scaled(input,  # torch.FloatTensor of any size
              target, 
              scale=None):
    if scale is not None:
        return criterion_l1(input*scale, target*scale)
    else:
        return criterion_l1(input, target)


def l2_scaled(input, target, scale=None):
    if scale is not None:
        return criterion_l2(input*scale, target*scale)
    else:
        return criterion_l2(input, target)

    

def cosine_loss(current, target, prediction, mean=True):
    delta_target = target - current
    delta_prediction = prediction - current
    if mean:
        return -1.0*cos(delta_target, delta_prediction).mean()
    else:
        return -1.0*cos(delta_target, delta_prediction)

def acos_loss(current, target, prediction):
    delta_target = target - current
    delta_prediction = prediction - current
    dots = (delta_target*delta_prediction).sum(dim=1)
    delta_target_norms = delta_target.norm(2,dim=1)
    delta_prediction_norms = delta_prediction.norm(2,dim=1)

    # print(dots.shape, "dots.shape")
    # print(delta_target_norms.shape)
    # print(delta_prediction_norms.shape)

    acos_args = dots/(delta_target_norms*delta_prediction_norms)

    # as suggested here, use an eps 
    # https://discuss.pytorch.org/t/numerically-stable-acos-dot-product-derivative/12851
    eps = 1e-6
    acos_args = torch.clamp(acos_args,min=-1.0+eps,max=1.0-eps)

    #print(acos_args)
    acoses = torch.acos(acos_args)
    return acoses.mean()


def l2_sequence(predictions, actions, config, chunk_length=1):
    loss = l2_scaled(predictions, actions)
    return loss


def l2_l1(predictions, actions, config, chunk_length=1):
    scale = torch.FloatTensor(config['loss_scaling']).cuda()
    loss = config["truncated_backprop_length"]*(0.1*l1_scaled(predictions, actions, scale) + l2_scaled(predictions, actions, scale))
    return loss*1.0/chunk_length

def parse_mdn_params(predictions, num_gaussians, L, A):
    """
    param predictions: list of dicts... described in NLL_MDN
    returns: (L,num_gaussians), (L,num_gaussians,A), (L,num_gaussians,A)
    """

    return predictions["logpi"].squeeze(0),  \
           predictions["sigma"].squeeze(0).view(L,num_gaussians,A), \
           predictions["mu"].squeeze(0).view(L,num_gaussians,A)

def log_vectorized_norm_pdf_multivariate_torch_s(x,mu,sigma_diag):
    """
    everything is torch.FloatTensors
    for N samples, dimension dim
    x           shape: N, dim
    mu          shape: N, dim
    sigma_diag  shape: N, dim
    returns shape: N
    """
    var_diag = sigma_diag**2
    
    size = len(x[0])
    det = var_diag.prod(dim=1)
    norm_const = 1.0 / ( math.pow((2*math.pi),float(size)/2) * torch.pow(det,1.0/2) )
    x_minus_mu = (x - mu)
    var_diag_inverse = 1.0 / var_diag
    exp_factor = -0.5 * torch.sum(x_minus_mu*var_diag_inverse*x_minus_mu,dim=1)
    return torch.log(norm_const)+exp_factor


def NLL_MDN(predictions, actions, config, chunk_length=1):
    """
    predictions: dict  containing:
                     key, value: "pi"    --> (1,L,num_gaussians)
                     key, value: "sigma" --> (1,L,num_gaussians*A)
                     key, value: "my"    --> (1,L,num_gaussians*A)
    actions:     (1,L,A)
    """ 
    
    predictions = predictions
    actions     = actions.squeeze(0)      # (L,A)

    L, A = actions.shape
    num_gaussians = config["num_gaussians"] # ToDo pete: would be better to get this from shape of pi

    predictions_logpi, predictions_sigma, predictions_mu = parse_mdn_params(predictions, num_gaussians, L, A)

    """
    At this point shapes are:
    predictions_pi:    (L,num_gaussians)
    predicitons_sigma: (L,num_gaussians,A)
    predicitons_mu:    (L,num_gaussians,A)
    actions:           (L,A)
    """

    actions_reshaped  = actions.unsqueeze(1).expand_as(predictions_sigma).contiguous().view(L*num_gaussians,A)
    mu_reshaped       = predictions_mu.view(L*num_gaussians,A)
    sigma_reshaped    = predictions_sigma.view(L*num_gaussians,A)
    logpi_reshaped    = predictions_logpi.view(L*num_gaussians)
    
    g_log_probs = log_vectorized_norm_pdf_multivariate_torch_s(actions_reshaped, mu_reshaped, sigma_reshaped)+logpi_reshaped
    g_log_probs = g_log_probs.view(L,num_gaussians)
    max_log_probs = torch.max(g_log_probs,dim=-1,keepdim=True)[0].detach()
    g_log_probs = g_log_probs - max_log_probs
    #print max_log_probs
    
    g_probs = torch.exp(g_log_probs)
    sum_probs = torch.sum(g_probs, dim=1)
    result = -(max_log_probs + torch.log(sum_probs))
    
    return torch.mean(result)*1.0/chunk_length, torch.exp(predictions_logpi), predictions_sigma, predictions_mu