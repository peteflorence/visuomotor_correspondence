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

import model_based_vision

# imitation_agent
from imitation_agent.utils import utils as imitation_agent_utils

DEBUG = False

PAUSE_ON_VISUALIZE_REF_DESCRIPTORS = False

class DenseObjectNetSpatialSoftmax(nn.Module):

    def __init__(self, config):
        super(DenseObjectNetSpatialSoftmax, self).__init__()
        
        self._config = config
        self._num_ref_descriptors = config["model"]["config"]["num_ref_descriptors"]
        self.load_descriptor_net()
        self.setup_pixel_maps()
        self.unset_visualize()
        self.print_counter = 0

    def set_do_surfing(self):
        self._reference_descriptor_vec.requires_grad = True
        self._reference_descriptor_vec = nn.Parameter(self._reference_descriptor_vec)

    def unset_do_surfing(self):
        self._reference_descriptor_vec.requires_grad = False
        
    def forward(self, x, # torch.Tensor (N,C,H,W), should already be normalized
                      input_data # dict which may have depth, transforms, camera calibration,
                ): # type torch.Tensor (N, K*2) which are stacked u, then v, pixel locations
        """
        K is the number of reference descriptors
        """

        descriptor_images = self.descriptor_net.forward(x, upsample=False)
        return self.get_expectations(descriptor_images, input_data, rgb=x)

    def set_visualize(self):
        self.vis_multiple = True

    def unset_visualize(self):
        self.vis_multiple = False

    def get_expectations(self, descriptor_images, input_data, rgb=None):
        self.print_counter += 1
        if self.print_counter % 100 == 0:
            print(self._reference_descriptor_vec[0])

        expectations, sm_activations = self.spatial_expectation_of_reference_descriptors(descriptor_images)

        # if DEBUG:
        #     plt.ion()
        #     plt.cla()

        #     x_descriptor_expectations = expectations[0,:self._num_ref_descriptors]
        #     y_descriptor_expectations = expectations[0,self._num_ref_descriptors:]

        #     x_pixel_coords, y_pixel_coords = self.convert_norm_coords_to_orig_pixel_coords(x_descriptor_expectations, y_descriptor_expectations)

        #     plt.imshow(rgb[0,:].permute(1,2,0).cpu().numpy())        
        #     for i in range(self._num_ref_descriptors):
        #         plt.scatter(x_pixel_coords[i], y_pixel_coords[i])
        #     plt.draw()
        #     plt.pause(0.001)

        #     try:
        #         if self.FIRST == False:
        #             pass
        #     except:
        #         import time
        #         time.sleep(1)

        #     self.FIRST = False

        if self.vis_multiple:
            plt.ion()
            plt.cla()
            for i in range(self.N_TEST_IMG):
                x_descriptor_expectations = expectations[i,:self._num_ref_descriptors]
                y_descriptor_expectations = expectations[i,self._num_ref_descriptors:]
                x_pixel_coords, y_pixel_coords = self.convert_norm_coords_to_orig_pixel_coords(x_descriptor_expectations, y_descriptor_expectations)
                self.a[i].clear()
                self.a[i].imshow(rgb[i,:].permute(1,2,0).cpu().numpy())
                for j in range(self._num_ref_descriptors):
                    self.a[i].scatter(x_pixel_coords[j],y_pixel_coords[j])
            plt.draw()
            plt.pause(0.001)


        if self._config["use_hard_3D_unprojection"]:
            expectations = model_based_vision.hard_pixels_to_3D_world(expectations, input_data['depth'], input_data['camera_to_world'], input_data['K'])
        if self._config["use_soft_3D_unprojection"]:
            expectations = model_based_vision.soft_pixels_to_3D_world(expectations, sm_activations, input_data['depth'], input_data['camera_to_world'], input_data['K'])

        return expectations

    def load_descriptor_net(self):
        """
        Loads pre-trained descriptor network using filename specified in config
        """
        path_to_network_params = self._config["model"]["descriptor_net"]
        path_to_network_params = pdc_utils.convert_data_relative_path_to_absolute_path(path_to_network_params, assert_path_exists=True)
        model_folder = os.path.dirname(path_to_network_params)
        self.descriptor_net = DenseCorrespondenceNetwork.from_model_folder(model_folder, model_param_file=path_to_network_params)
        self.descriptor_net.eval()
        print("loaded descriptor_net")

    def initialize_parameters_via_dataset(self, 
                                          dataset, # type ImitationEpisodeDataset or ImitationEpisodeSequenceDataset
                                          ):
        """
        The use case at the moment is descriptor initialization.  This may expand.
        """
        spartan_dataset = dataset.spartan_dataset
        log_name = sorted(dataset.episodes.keys())[0]
        camera_num = self._config["camera_num"]
        index_to_sample = imitation_agent_utils.get_image_index_to_sample_from_config(self._config)

        if "reference_image_initialization" in self._config:
            log_name = self._config["reference_image_initialization"]["log_name"]
            index_to_sample = self._config["reference_image_initialization"]["index_to_sample"]


        # for logging purposes
        self.ref_log_name   = log_name
        self.ref_index      = index_to_sample
        self.ref_camera_num = camera_num

        rgb, _, mask, _ = spartan_dataset.get_rgbd_mask_pose(log_name, camera_num, index_to_sample)
        rgb_tensor = spartan_dataset.rgb_image_to_tensor(rgb).unsqueeze(0).cuda()
        #print(rgb_tensor.shape, "is rgb_tensor.shape, we want 1, D, H, W")

        mask_tensor = torch.from_numpy(np.asarray(mask))
        descriptor_image_tensor            = self.descriptor_net.forward(rgb_tensor).detach()
        #print(descriptor_image_tensor.shape, "is descriptor_image_tensor.shape, we are expecting N, D, H, W")        

        
        if "reference_vec" in self._config["model"]:
            self._reference_descriptor_vec = torch.load(self._config['model']["reference_vec"])
        else:
            print("about to sample ref descriptors")
            self.random_sample_reference_descriptors(rgb_tensor, descriptor_image_tensor, mask_tensor)


        if True:
            descriptor_image_tensor_downsample = self.descriptor_net.forward(rgb_tensor, upsample=False).detach()

            # TEST TRYING TO RECOVER THE REFERENCE DESCRIPTORS
            expectations, _ = self.spatial_expectation_of_reference_descriptors(descriptor_image_tensor_downsample)

            print(expectations.shape, "expectations.shape, should be (1, 2*K)")
            x_descriptor_expectations = expectations[0,:self._num_ref_descriptors]
            y_descriptor_expectations = expectations[0,self._num_ref_descriptors:]

            x_pixel_coords, y_pixel_coords = self.convert_norm_coords_to_orig_pixel_coords(x_descriptor_expectations, y_descriptor_expectations)

            print("these are the recomputed expectations")
            if PAUSE_ON_VISUALIZE_REF_DESCRIPTORS:
                plt.ioff()
            else:
                plt.ion()

            plt.imshow(rgb_tensor.squeeze(0).permute(1,2,0).cpu().numpy())
            for i in range(self._num_ref_descriptors):
                plt.scatter(x_pixel_coords[i], y_pixel_coords[i])

            if PAUSE_ON_VISUALIZE_REF_DESCRIPTORS:
                plt.show()
            else:
                plt.draw()
                plt.pause(0.5)

            # TEST A FEW MORE LOCATIONS
            for i in range(12):
                log_name = dataset.get_random_log_name()
                index_to_sample = dataset.episodes[log_name].get_random_idx()
                rgb, _, mask, _ = spartan_dataset.get_rgbd_mask_pose(log_name, camera_num, index_to_sample)
                rgb_tensor = spartan_dataset.rgb_image_to_tensor(rgb).unsqueeze(0).cuda()
                descriptor_image_tensor_downsample = self.descriptor_net.forward(rgb_tensor, upsample=False).detach()

                expectations, _ = self.spatial_expectation_of_reference_descriptors(descriptor_image_tensor_downsample)

                print(expectations.shape, "expectations.shape, should be (1, 2*K)")
                x_descriptor_expectations = expectations[0,:self._num_ref_descriptors]
                y_descriptor_expectations = expectations[0,self._num_ref_descriptors:]

                x_pixel_coords, y_pixel_coords = self.convert_norm_coords_to_orig_pixel_coords(x_descriptor_expectations, y_descriptor_expectations)

                print("from another view")
                plt.cla()
                plt.imshow(rgb_tensor.squeeze(0).permute(1,2,0).cpu().numpy())        
                for i in range(self._num_ref_descriptors):
                    plt.scatter(x_pixel_coords[i], y_pixel_coords[i])

                if PAUSE_ON_VISUALIZE_REF_DESCRIPTORS:
                    plt.show()
                else:
                    plt.draw()
                    plt.pause(0.5)



            print("sleeping so you can inspect the initialization")
            import time; time.sleep(3)
            plt.close()
            plt.cla()
            plt.ioff()

        # I want this to only get up after the initialization
        self.N_TEST_IMG = 5
        self.f, self.a = plt.subplots(1, self.N_TEST_IMG, figsize=(5, 2))



    def convert_norm_coords_to_orig_pixel_coords(self, norm_matches_x, norm_matches_y, to_cpu=True):
        matches_x = ((norm_matches_x/2.0 + 0.5)*640.0).detach().cpu().numpy()
        matches_y = ((norm_matches_y/2.0 + 0.5)*480.0).detach().cpu().numpy()
        return matches_x, matches_y


    def setup_pixel_maps(self):
        # shouldn't hardcode these in future
        H = 60
        W = 80

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., W),
            np.linspace(-1., 1., H)
        )

        self.pos_x = torch.from_numpy(pos_x).float().cuda()
        self.pos_y = torch.from_numpy(pos_y).float().cuda()

    def spatial_expectation_of_reference_descriptors(self, 
                                                     descriptor_images, # type torch.Tensor shape N, D, H_small, W_small
                                                     ): # type -> torch.Tensor shape N, 2*K
        N, D, H, W = descriptor_images.shape
        
        #print("N, D, H, W", N, D, H, W)
        Nref, Dref = self._reference_descriptor_vec.shape
        
        #print("Nref, Dref", Nref, Dref)
        assert Dref == D
        
        descriptor_images = descriptor_images.permute(0,2,3,1)     # N, H, W, D
        descriptor_images = descriptor_images.unsqueeze(3)         # N, H, W, 1, D
        
        #print(descriptor_images.shape, "should be N, H, W, 1, D")
        descriptor_images = descriptor_images.expand(N,H,W,Nref,D) # N, H, W, Nref, D
        #print(descriptor_images.shape, "should be N, H, W, Nref, D")

        deltas = descriptor_images - self._reference_descriptor_vec
        #print(deltas.shape, "should also be N, H, W, Nref, D?")

        neg_squared_norm_diffs = -1.0 * torch.sum(torch.pow(deltas,2), dim=4) # N, H, W, Nref
        #print(neg_squared_norm_diffs.shape, "should be N, H, W, Nref")

        ## spatial softmax
        neg_squared_norm_diffs = neg_squared_norm_diffs.permute(0,3,1,2)   # N, Nref, H, W
        #print(neg_squared_norm_diffs.shape, "should be N, Nref, H, W")
        
        neg_squared_norm_diffs_flat = neg_squared_norm_diffs.view(N, Nref, H*W) # 1, nm, H*W
        #print(neg_squared_norm_diffs_flat.shape, "should be N, Nref, H*W")
        
        softmax = torch.nn.Softmax(dim=2)
        softmax_activations = softmax(neg_squared_norm_diffs_flat).view(N, Nref, H, W) # N, Nref, H, W
        #print(softmax_activations.shape, "should be N, Nref, H, W")
        
        # softmax_attentions shape is N, Nref, H, W
        expected_x = torch.sum(softmax_activations*self.pos_x, dim=(2,3))
        #print(expected_x.shape, "expected_x.shape")

        expected_y = torch.sum(softmax_activations*self.pos_y, dim=(2,3))
        #print(expected_y.shape, "expected_y.shape")
        
        stacked_2d_features =  torch.cat((expected_x, expected_y), 1)
        
        #print(stacked_2d_features.shape, "should be N, 2*Nref")

        return stacked_2d_features, softmax_activations

    def random_sample_reference_descriptors(self, rgb_tensor, # type torch.Tensor, shape 1, C, H, W
                                            descriptor_image_tensor, # type torch.Tensor, shape 1, D, H, W
                                            mask_tensor, # type torch.Tensor, shape H, W
                                            ): # no return

        self.ref_pixels_uv = random_sample_from_masked_image_torch(mask_tensor, self._num_ref_descriptors) # tuple of (u's, v's)

        ref_pixels_flattened = self.ref_pixels_uv[1]*mask_tensor.shape[1]+self.ref_pixels_uv[0]

        # DEBUG
        if DEBUG:
        
            descriptor_image_np = descriptor_image_tensor.cpu().numpy()
            #plt.imshow(descriptor_image_np)
            #plt.show()

            plt.imshow(rgb_tensor.squeeze(0).permute(1,2,0).cpu().numpy())
            ref_pixels_uv_numpy = (self.ref_pixels_uv[0].numpy(), self.ref_pixels_uv[1].numpy())
            for i in range(self._num_ref_descriptors):
                plt.scatter(ref_pixels_uv_numpy[0][i], ref_pixels_uv_numpy[1][i])
            plt.show()
        
        ref_pixels_flattened = ref_pixels_flattened.cuda()
        
        #print(descriptor_image_tensor.shape, "should be 1, D, H, W")

        D = descriptor_image_tensor.shape[1]
        WxH = descriptor_image_tensor.shape[2]*descriptor_image_tensor.shape[3]
        

        # now view as D, H*W
        descriptor_image_tensor = descriptor_image_tensor.squeeze(0).contiguous().view(D, WxH)

        # now switch back to H*W, D
        descriptor_image_tensor = descriptor_image_tensor.permute(1,0)
        
        # self.ref_descriptor_vec is Nref, D 
        self._reference_descriptor_vec = torch.index_select(descriptor_image_tensor, 0, ref_pixels_flattened)
        #self._reference_descriptor_vec = torch.nn.Parameter(self._reference_descriptor_vec)

        #print(self._reference_descriptor_vec.shape, "should be Nref, D")
        
        # TURN OFF IF WANT TO OPTIMIZE DESCRIPTORS
        #self._reference_descriptor_vec.requires_grad = False
