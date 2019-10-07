from __future__ import print_function
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import dense_correspondence_manipulation.utils.utils as pdc_utils
pdc_utils.add_dense_correspondence_to_python_path()
from dense_correspondence.network.dense_correspondence_network import DenseCorrespondenceNetwork
from dense_correspondence.correspondence_tools.correspondence_finder import random_sample_from_masked_image_torch

import numpy as np

import matplotlib.pyplot as plt


DEBUG = False
if DEBUG:
    plt.ion()
    f, a = plt.subplots(1, 1, figsize=(1, 1))


class SpatialAutoencoderWrapper(nn.Module):
    def __init__(self, config):
        super(SpatialAutoencoderWrapper, self).__init__()

        self._config = config
        if config["model"]["vision_net"] == "SpatialAutoencoder":
            self._spatial_autoencoder = torch.load(config["model"]["autoencoder_net"])
        if config["model"]["vision_net"] == "EndToEnd":
            width = (config["model"]["u_range_end"] - config["model"]["u_range_start"])/2
            self._spatial_autoencoder = SpatialAutoencoder(H=240,W=width)

        self._spatial_autoencoder = self._spatial_autoencoder.eval().cuda()
        
        # if DEBUG:
        #   plt.ion()
        #   self.f, self.a = plt.subplots(1, 1, figsize=(1, 1))


    def forward(self, x, # torch.Tensor (N,C,H,W), should already be normalized
                downsample_preprocess = True
                ): # type torch.Tensor (N, K*2) which are stacked u, then v, pixel locations
        """
        K is the number of reference descriptors
        """

        # try:
        #     if self.FIRST is False:
        #         pass
        # except:
        #     if DEBUG:
        #         plt.cla()
        #         a.imshow(np.random.randn(240,240,3))
        #         plt.draw(); plt.pause(0.001)
        #         import time; time.sleep(5)

        if downsample_preprocess:
            u_start = self._config["model"]["u_range_start"]
            u_end = self._config["model"]["u_range_end"]
            x = x[:,:,:,u_start:u_end]
            x = torch.nn.functional.interpolate(x, scale_factor=240.0/480.0, mode='bilinear', align_corners=True)

        features, decoded, _, sm_activations = self._spatial_autoencoder.forward(x)

        if DEBUG:
            plt.cla()
            
            a.imshow(x.detach().cpu().permute(0,2,3,1).numpy()[0]+0.3)
            #a.imshow(np.reshape(decoded.detach().cpu().numpy()[0], (60, 60)), cmap='gray')
            features_first_image = features[0].detach().cpu().numpy()
            K = len(features_first_image)/2
            x_features = (features_first_image[:K]+1.0)/2.0*240.0
            y_features = (features_first_image[K:]+1.0)/2.0*240.0
            for j in range(len(x_features)):
                a.scatter(x_features[j], y_features[j])
            plt.draw(); plt.pause(0.001)

        # optionally filter?

        # optionally prune?
        #features = self.prune(features, sm_activations)
        self.FIRST = False
        
        return features


    def prune(self, features, sm_activations):

        for i in range(features.shape[0]):
            features_one_image = features[i]
            
            K = len(features_one_image)/2

            x_features = (features_one_image[:K]+1.0)/2.0*109.0
            y_features = (features_one_image[K:]+1.0)/2.0*109.0
            for j in range(len(x_features)):

                y = int(y_features[j]*109.0/60)
                x = int(x_features[j]*109.0/60)

                if x < 1:
                    x += 1
                if x > 108:
                    x -= 1
                if y < 1:
                    y += 1
                if y > 108:
                    y -= 1

                sum_over_3_3_window = sm_activations[i,j,y-1:y+2,x-1:x+2].sum()

                if sum_over_3_3_window.item() > 0.2:
                    print("detected", j)
                else:
                    features[i,j] = 0.0
                    features[i,j+K] = 0.0

        return features

    def initialize_parameters_via_dataset(self,dataset):
        pass

# imlemented exactly as in Figure 3: https://arxiv.org/pdf/1509.06113.pdf
# Deep Spatial Autoencoders for Visuomotor Learning
# the batchnorm is mentioned in the text but not figure
class SpatialAutoencoder(nn.Module):
    def __init__(self, H, W, need_decoder=False, use_imagenet_pretrained=True, decoder_dropout=0.0):
        super(SpatialAutoencoder, self).__init__()
        self.need_decoder = need_decoder

        self.conv1 = nn.Conv2d(in_channels = 3,  out_channels=64, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels=32, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels=16, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(16)
        
        
        self.H = H
        self.W = W

        if self.W == 240:
            self.final_width = 109
        if self.W == 320:
            self.final_width = 149
        else:
            raise ValueError("Width should be one of above")

        self.setup_pixel_maps()

        if self.need_decoder:
            if self.W == 240:
                self.decode_linear = nn.Linear(32,60*60)
            if self.W == 320:
                self.decode_linear = nn.Linear(32,80*60)

            self.decode_drop = nn.Dropout(p=decoder_dropout)

        # only will be updated if do pose estimation
        self.pose_fc = nn.Linear(32,9)

        if use_imagenet_pretrained:
            # Levine*, Finn*, Darrel, Abeel JMLR 2016 notes using Googlenet first layer.
            # Use pytorch to download googlenet.
            # It is a little annoying that you can't do this inside of our docker environment,
            # because we have a special torchvision fork.
            # But you can just install the latest torch + torchvision outside your docker container, and then just:
            # $ import torchvision.models
            # $ gnet = torchvision.models.googlenet(pretrained=True)
            # This will download the below file (at least with torchvision 0.3.0) into the below location.
            # It will then be visible in the docker since we have mounted this folder.
            googlenet = torch.load(os.path.join(os.path.expanduser('~'),".cache/torch/checkpoints/googlenet-1378be20.pth"))
            self.conv1.weight = torch.nn.Parameter(googlenet['conv1.conv.weight'])


    def forward(self, x, downsample_preprocess=False):

        if downsample_preprocess:
            u_start = self._config["model"]["u_range_start"]
            u_end = self._config["model"]["u_range_end"]
            x = x[:, :, :, u_start:u_end]
            x = torch.nn.functional.interpolate(x, scale_factor=240.0/480.0, mode='bilinear', align_corners=True)

        x = x.view(-1, 3, self.H, self.W)
        if self.need_decoder:
            down_sampled = torch.nn.functional.interpolate(x, scale_factor=60.0/240.0, mode='bilinear', align_corners=True)
            down_sampled_gray = torch.zeros(down_sampled.shape[0], down_sampled.shape[2], down_sampled.shape[3])
            down_sampled_gray = (down_sampled[:,0,:,:] + down_sampled[:,1,:,:] + down_sampled[:,2,:,:])/3.0
            #print down_sampled.shape, "is down_sampled.shape"
            #print down_sampled_gray.shape, "is down_sampled_gray.shape"
        else:
            down_sampled_gray = torch.zeros(1)

        x = F.relu(self.bn1(self.conv1(x)))
        #print x.shape, "is shape after first conv"
        x = F.relu(self.bn2(self.conv2(x)))
        #print x.shape, "is shape after second conv"
        x = F.relu(self.bn3(self.conv3(x)))
        #print x.shape, "is shape after third conv"

        spatial_H = x.shape[2]
        spatial_W = x.shape[3]

        x = x.view(x.shape[0], x.shape[1], spatial_H*spatial_W) # 1, nm, H*W
        softmax = torch.nn.Softmax(dim=2)
        softmax_activations = softmax(x).view(x.shape[0], x.shape[1], spatial_H, spatial_W)
        #print softmax_activations.shape, "is shape after softmax"


        expected_x = torch.sum(softmax_activations*self.pos_x, dim=(2,3))
        expected_y = torch.sum(softmax_activations*self.pos_y, dim=(2,3))

        x_features_indices = ((expected_x+1.0)/2.0*self.final_width).long()  # N,16
        y_features_indices = ((expected_y+1.0)/2.0*109).long()  # N,16

        features = torch.cat((expected_x, expected_y), dim=1)
        #print features.shape, "is features.shape"

        if self.need_decoder: # for reconstruction
            reconstructed = self.decode_linear(self.decode_drop(features))
        else:
            reconstructed = torch.zeros(1)
        

        return features, reconstructed, down_sampled_gray.view(down_sampled_gray.shape[0],-1), softmax_activations

    def forward_for_pose(self,x):
        features, _, _, _ = self.forward(x, downsample_preprocess=True) # now N,32
        estimated_3d_points = self.pose_fc(features)
        return estimated_3d_points

    def setup_pixel_maps(self):
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.final_width),
            np.linspace(-1., 1., 109)
        )

        self.pos_x = torch.from_numpy(pos_x).float().cuda() # H, W
        self.pos_y = torch.from_numpy(pos_y).float().cuda() # H, w

    
    def using_old_autoencoder(self):
        self.need_decoder = True
        self.decode_linear = self.linear
        self.final_width = 109
