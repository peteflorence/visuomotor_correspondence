from __future__ import print_function

# policy nets
from imitation_agent.model.mlpstateless import MLPStateless
from imitation_agent.model.lstm_standard import LSTMStandard

# vision nets
from imitation_agent.model.don_spatial_softmax import DenseObjectNetSpatialSoftmax
from imitation_agent.model.spatial_autoencoder import SpatialAutoencoderWrapper

from imitation_agent.model.visuo_motor import VisuoMotorNetwork

class ModelFactory(object):
    """
    Helper class to construct models from configs
    """

    @staticmethod
    def get_model(config, # dict
                  **kwargs): # -> type nn.Module object
        """
        Returns an object which subclasses nn.Module
        """
        policy_type = config["model"]["policy_net"]
        policy_net = getattr(ModelFactory, policy_type)(config, **kwargs)

        vision_type = config["model"]["vision_net"]
        vision_net = getattr(ModelFactory, vision_type)(config, **kwargs)

        return VisuoMotorNetwork(vision_net, policy_net)

    @staticmethod
    def mlp_stateless(config, # dict
                      **kwargs):
        return MLPStateless(config)


    @staticmethod
    def LSTM_standard(config, # dict
             **kwargs):
        return LSTMStandard(config)

    @staticmethod
    def none(config, # dict
             **kwargs): 
        return None

    @staticmethod
    def DonSpatialSoftmax(config, # dict
                          **kwargs):
        return DenseObjectNetSpatialSoftmax(config)

    @staticmethod
    def SpatialAutoencoder(config, # dict
                          **kwargs):
        return SpatialAutoencoderWrapper(config)

    @staticmethod
    def EndToEnd(config,
                **kwargs):
        return SpatialAutoencoderWrapper(config)






