import torch
import numpy as np
import sys
import os
import time
import random

import spartan.utils.utils as spartanUtils

import copy

import train_vis

def make_deterministic(seed=0):
    ## Make repeatable
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    ##

def deterministic_downsample(logs_config, num_to_keep):
    random.seed(0)
    downsampled_logs_config = copy.deepcopy(logs_config)
    
    logs = downsampled_logs_config["logs"]
    random.shuffle(logs)
    
    new_logs = logs[0:num_to_keep]
    downsampled_logs_config["logs"] = new_logs
    print len(downsampled_logs_config["logs"]), "is new logs len"
    print len(logs_config["logs"]), "is original logs len"

    return downsampled_logs_config


def get_descriptor_images_detached(rgb_images, dcn):
    """
    # Option: get descriptor images, all at once for the sequence
    # This is only useful if not backpropping through the dcn

    rgb_images: N, L, 3, H, W (will assume N = 1.  L is for sequence length)
    dcn: dense correspondence network
    """
    rgb_images = rgb_images.squeeze(0) # now is N, 3, H, W
    num_minibatch_images = 10
    chunked_rgb_images = rgb_images.chunk(rgb_images.shape[0]/num_minibatch_images+1)
    descriptor_images = []

    for chunk in chunked_rgb_images:
        chunked_descriptor_images = dcn.forward(chunk.cuda(), upsample=False).detach()
        descriptor_images += [chunked_descriptor_images]
    descriptor_images = torch.cat(descriptor_images, 0)
    return descriptor_images