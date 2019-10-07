import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPStateless(nn.Module):

    def __init__(self, config):
        super(MLPStateless, self).__init__()
        self._config = config
        self._eps = 1e-3

        # whether or not to normalize the input
        self._normalize_input = True

        # list of fully connected layers
        self._fc_list = nn.ModuleList()


        # think about nn.Sequential

        model_config = config["model"]["config"]
        self.drop = nn.Dropout(p=model_config["dropout_prob"])

        if config["use_vision"]:
            d = 2
            if config["use_hard_3D_unprojection"] or config["use_soft_3D_unprojection"]:
                d = 3
            self._non_vision_index_start = model_config["num_ref_descriptors"]*d
            self._normalization_index_start = model_config["num_ref_descriptors"]*d
        else:
            self._non_vision_index_start = 0
            self._normalization_index_start = 0

        for i, num_units in enumerate(model_config["units_per_layer"]):

            # special care for first layer
            if i == 0:
                input_dim = model_config["num_inputs"] + self._non_vision_index_start
            else:
                input_dim = model_config["units_per_layer"][i-1]
                if model_config["use_xyz_passthrough_every_layer"]:
                    input_dim += 3

            # special care for last layer
            if i == (len(model_config["units_per_layer"]) - 1):
                output_dim = model_config["num_outputs"]
            else:
                output_dim = model_config["units_per_layer"][i]


            fc_layer = nn.Linear(input_dim, output_dim)
            self._fc_list.append(fc_layer)

    @property
    def normalize_input(self):
        return self._normalize_input

    @normalize_input.setter
    def normalize_input(self, val):
        self._normalize_input = val

    def set_normalization_parameters(self, stats,  # dict
                                     ):
        """

        :param stats: dict with keys ['observation,, 'action'] each with subkeys ['mean, 'stddev']
        :type stats:
        :return:
        :rtype:
        """
        self._dataset_stats = stats
        self._input_mean = self._dataset_stats['observation']['mean'].cuda()
        self._input_std = self._dataset_stats['observation']['std'].cuda()


    def set_normalization_to_include_vision_features(self):
        self._normalization_index_start = 0

    def forward(self, x, # type torch.Tensor (batch_size) x num_inputs
                ): # type -> torch.Tensor batch_size x num_outputs

        if self._config["model"]["config"]["use_xyz_passthrough_every_layer"]:
            xyz = x[:,self._non_vision_index_start:self._non_vision_index_start+3] * 1.0

        # need to do the normalization
        if self.normalize_input:
            x[:,self._normalization_index_start:] = self.compute_input_normalization(x[:,self._normalization_index_start:])

        num_layers = len(self._fc_list)
        for i in xrange(0, num_layers - 1):
            x = F.relu(self._fc_list[i](x))
            x = self.drop(x)
            if self._config["model"]["config"]["use_xyz_passthrough_every_layer"]:
                x = torch.cat((x,xyz), dim=1)


        # no ReLU on last layer
        x = self._fc_list[-1](x)
        #x[:,0:3] += xyz
        return x 

    def compute_input_normalization(self, x):
        """
        Normalize the input by subtracting mean and dividing by stddev
        :param x:
        :type x:
        :return:
        :rtype:
        """
        
        normed = (x-self._input_mean)
        if self._config["divide_by_std"]:
            normed = normed / (self._input_std + self._eps)
        return normed

    def set_states_zero(self):
        pass

    def set_states_initial(self):
        pass

    def detach_parameters(self):
        pass

    def set_imitation_episode(self, val):
        self._imitation_episode = val