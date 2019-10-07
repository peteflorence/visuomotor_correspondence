import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMStandard(nn.Module):
    """
    Simple LSTM
    """
    def __init__(self, config):
        super(LSTMStandard, self).__init__()
        self._config = config
        self.load_config_dict()
        
        model_config = config["model"]["config"]

        self.cell_size = model_config["RNN_CELL_SIZE"]
        self.mlp_size = model_config["MLP_SIZE"]
        
        self.n_layers = model_config["RNN_layers"]
        self.scaling_eps = 1e-6
        self._normalize_input = True # whether or not to apply input normalization

        self.drop = nn.Dropout(p=model_config["dropout_prob_MLP"])

        if config["use_vision"]:
            d = 2
            if config["use_hard_3D_unprojection"] or config["use_soft_3D_unprojection"]:
                d = 3
            self.input_size = d*model_config["num_ref_descriptors"]+self.pose_size
            self._non_vision_index_start = model_config["num_ref_descriptors"]*d
            self._normalization_index_start = model_config["num_ref_descriptors"]*d
        else:
            self.input_size =  self.pose_size
            self._non_vision_index_start = 0
            self._normalization_index_start = 0

        self.fc1 = nn.Linear(self.input_size, self.mlp_size)
        self.fc2 = nn.Linear(self.mlp_size, self.cell_size)

        if not model_config["USE_RNN"]:
            self.fc2 = nn.Linear(self.mlp_size, self.mlp_size)

        LSTM_input_size = self.cell_size
        if model_config["use_xyz_passthrough"]:
            LSTM_input_size += 3
        if not model_config["USE_MLP"]:
           LSTM_input_size = self.input_size

        self.norm1 = nn.LayerNorm(LSTM_input_size)
        self.norm2 = nn.LayerNorm(self.cell_size) # manuelli: not used?


        self.rnn = nn.LSTM(LSTM_input_size, self.cell_size, self.n_layers, dropout=model_config["dropout_prob_LSTM"],
                           batch_first=True)  # input_size, hidden_size, num_layer

        if self.regression_type == "MDN":
            self.z_pi = nn.Linear(self.cell_size, self.num_gaussians)
            self.z_sigma = nn.Linear(self.cell_size, self.num_gaussians * self.action_size)
            self.z_mu = nn.Linear(self.cell_size, self.num_gaussians * self.action_size)
        else:
            if config["model"]["config"]["RNN_bypass"]:
                linear_size = self.cell_size*2
            elif not config["model"]["config"]["USE_RNN"]:
                linear_size = self.mlp_size
            else:
                linear_size = self.cell_size
            self.linear = nn.Linear(linear_size, self.action_size)

        self.overall_input_mean = None
        self.overall_input_std = None


    def load_config_dict(self):
        self.pose_size           = self._config["model"]["config"]["pose_size"]
        self.action_size         = self._config["model"]["config"]["action_size"]
        self.regression_type = self._config["regression_type"]
        if self.regression_type == "MDN":
            self.num_gaussians = self._config["num_gaussians"]

    def set_states_zero(self):
        self.h_initial = torch.zeros(self.n_layers, 1, self.cell_size, requires_grad=True).cuda()
        self.c_initial = torch.zeros(self.n_layers, 1, self.cell_size, requires_grad=True).cuda()

    def set_states_initial(self):
        self.h0 = self.h_initial
        self.c0 = self.c_initial

    def detach_parameters(self):
        self.h_initial = self.h_initial.detach()
        self.c_initial = self.c_initial.detach()
        self.h0 = self.h0.detach()
        self.c0 = self.c0.detach()

    def set_normalization_parameters(self, mean, std):
        """
        Set parameters used to normalize the input before passing through the network
        """
        # make sure they are on the GPU
        self.overall_input_mean = mean.cuda()
        self.overall_input_std = std.cuda()

    def set_normalization_to_include_vision_features(self):
        self._normalization_index_start = 0

    def from_original_to_scaled(self, input_t):
        return (input_t - self.overall_input_mean) / (self.overall_input_std + self.scaling_eps)

    def forward(self, input_t):
        """
        input_t shape: N, d
        """
        # make sure noth trying to call one-step forward in batch
        # since this is a sequence model
        assert input_t.shape[0] == 1

        return self.forward_on_series(input_t.unsqueeze(1))

    def forward_on_series(self, input):
        """
        input should be of shape: N, L, D
        - N is batch size, should typically be 1 for sequence data
        - L is the length of the sequence
        - D is the dimension of each input in the sequence
        """
        if self._config["model"]["config"]["use_xyz_passthrough"]:
            xyz  = input[:,:,self._non_vision_index_start:self._non_vision_index_start+3] * 1.0

        try:
            if self._config["action"]["config"]["quaternion"]:
                quat = input[:,:,self._non_vision_index_start+3:self._non_vision_index_start+7] * 1.0
            if self._config["action"]["config"]["rpy"]["roll"]:
                roll = input[:,:,self._non_vision_index_start+3] * 1.0
        except:
            pass

        x = input * 1.0

        if self._normalize_input:
            for i in range(0, x.shape[1]):
                x[:, i, self._normalization_index_start:] = self.from_original_to_scaled(x[:, i, self._normalization_index_start:])

        if self._config["model"]["config"]["USE_MLP"]:
            x = F.relu(self.fc1(x))
            x = self.drop(x)
            x = F.relu(self.fc2(x))
            x = self.drop(x)

        if self._config["model"]["config"]["use_xyz_passthrough"]:
            x = torch.cat((x,xyz), dim=2)

        if self._config["model"]["config"]["USE_RNN"]:
            output, (self.h0, self.c0) = self.rnn(self.norm1(x), (self.h0, self.c0))
        else:
            output = x

        if self._config["model"]["config"]["RNN_bypass"]:
            output = torch.cat((output,x), dim=2)

        if self.regression_type == "MDN":
            logpi = F.log_softmax(self.z_pi(output), -1)
            sigma = torch.exp(self.z_sigma(output))
            mu = self.z_mu(output)
            mdn_params = dict()
            mdn_params["logpi"] = logpi
            mdn_params["sigma"] = sigma
            mdn_params["mu"] = mu
            return mdn_params

        else:
            x = self.linear(output)
            return x