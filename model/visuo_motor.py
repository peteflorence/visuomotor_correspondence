from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisuoMotorNetwork(nn.Module):

    def __init__(self, vision_net, policy_net):
        super(VisuoMotorNetwork, self).__init__()
        self._vision_net = vision_net
        self._policy_net = policy_net
        assert self._policy_net is not None
        self.unset_use_precomputed_features()
        self.unset_use_precomputed_descriptor_images()

    def forward(self, input_data, # type dict
                    ):
        """
        :param input_data: keys are 'image', 'observation'. This is batch
         data. Image data is normalized, observation data is not.
         'image' should be shape N, C, H, W
         'observation' should be shape N, O
        """
        if self._vision_net is not None:

            if self.use_precomputed_features:
                y_vision = input_data['image']
            elif self.use_precomputed_descriptor_images:
                y_vision = self._vision_net.get_expectations(input_data['image'], input_data)
            else:
                y_vision = self._vision_net(input_data['image'], input_data)
                if self._policy_net._config["freeze_vision"]:
                    y_vision = y_vision.detach()

            x_policy = torch.cat((y_vision, input_data['observation']),dim=1)
        else:
            x_policy = input_data['observation']

        return self._policy_net(x_policy)

    def forward_on_series(self, input_data, # type dict
                              ):
        """
        :param input_data: keys are 'images', 'observations'
        'images' should be shape: N=1, L, C, H, W
        'observations' should be shape N=1, L, O
        """
        

        if self._vision_net is not None:

            if self.use_precomputed_features:
                y_vision = input_data['images'].squeeze(0)
            elif self.use_precomputed_descriptor_images:
                y_vision = self._vision_net.get_expectations(input_data['images'].squeeze(0), input_data)
            else:
                y_vision = self._vision_net(input_data['images'].squeeze(0))
                if self._policy_net._config["freeze_vision"]:
                    y_vision = y_vision.detach()
                
            x_policy = torch.cat((y_vision, input_data['observations'].squeeze(0)), dim=1).unsqueeze(0)
        else:
            x_policy = input_data['observations']

        return self._policy_net.forward_on_series(x_policy)

    def set_use_precomputed_features(self):
        self.use_precomputed_features = True
        self._policy_net.set_normalization_to_include_vision_features()

    def set_use_precomputed_descriptor_images(self):
        self.use_precomputed_descriptor_images = True
        self._policy_net.set_normalization_to_include_vision_features()

    def set_do_surfing(self):
        self._vision_net.set_do_surfing()

    def unset_use_precomputed_features(self):
        self.use_precomputed_features = False

    def unset_use_precomputed_descriptor_images(self):
        self.use_precomputed_descriptor_images = False

    def set_normalization_parameters(self, *args):
        self._policy_net.set_normalization_parameters(*args)

    def set_states_zero(self):
        self._policy_net.set_states_zero()

    def set_states_initial(self):
        self._policy_net.set_states_initial()

    def detach_parameters(self):
        self._policy_net.detach_parameters()

    def initialize_parameters_via_dataset(self, dataset):
        if self._vision_net is not None:
            self._vision_net.initialize_parameters_via_dataset(dataset)

    @ property
    def action_size(self):
        return self._policy_net.action_size
