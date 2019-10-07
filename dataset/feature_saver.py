import torch
import os

from spartan.utils import utils as spartan_utils

"""
This class will manage saving+loading
of precomputed features that you've already
precomputed.


The format is as follows:
- in the pdc/imitation/features dir, a folder will be created for every network name,
- in that folder, every new set of reference_descriptors that is used gets a new subfolder
- each subfolder contains the two files shown below.

pdc/imitation/features/
    sugar_spatial_more_masked_se2_3/
        001/
            reference_descriptors.pth -- this stores the _reference_descriptor_vec of the don_spatial_softmax
            train_features.pth -- this stores the trainset precomputed_features
            test_features.pth -- this stores the testset precomputed_features
            train_d_images.pth -- this stores the trainset precomputed d_images
            test_d_images.pth -- this stores the testset precomputed d_images
        002/
            reference_descriptors.pth
            train_features.pth
            test_features.pth
            train_d_images.pth
            test_d_images.pth

At load time:
- check if, for the network, the same reference descriptors have been saved out before
- if yes, load them


At save time:
- save the reference descriptors and features to disk

"""

class FeatureSaver():

    def __init__(self):
        data_dir = spartan_utils.get_data_dir()
        self.save_features_path = os.path.join(data_dir, "pdc/imitation/features")
        if not os.path.isdir(self.save_features_path):
            os.makedirs(self.save_features_path)


    def get_features_for_net_path(self, config):
        net_full_path = config["model"]["descriptor_net"]
        net_name_ignoring_iteration = os.path.basename(os.path.dirname(net_full_path))
        features_for_net_path = os.path.join(self.save_features_path, net_name_ignoring_iteration)
        return features_for_net_path

    def load_if_already_have(self, save_type, config, reference_descriptor_vec, episodes, train_or_test):
        """
        Return False if couldn't load
        Return True if could load, and by reference each episodes' precomputed_features dict will be loaded
        """
        features_for_net_path = self.get_features_for_net_path(config)

        if not os.path.isdir(features_for_net_path):
            print "Couldn't find any precomputed features for this network!"
            return False

        for folder in sorted(os.listdir(features_for_net_path)):
            path_to_saved_reference_descriptors = os.path.join(features_for_net_path, folder, "reference_descriptors.pth")
            saved_reference_descriptors = torch.load(path_to_saved_reference_descriptors)
            if save_type == "features":
                if not (saved_reference_descriptors.shape == reference_descriptor_vec.shape and (saved_reference_descriptors == reference_descriptor_vec).all()):
                    continue # didnt match descriptors
            print "Found folder with matching reference_descriptors! :", path_to_saved_reference_descriptors
            saved_path = os.path.join(features_for_net_path, folder, train_or_test+self.get_type(save_type))
            if os.path.exists(saved_path):
                print "Found precomputed stuff!! Loading...", saved_path
                saved_episodes = torch.load(saved_path)
                successful_load = self.load_up_saved(save_type, episodes, saved_episodes)
                if successful_load:
                    return True
            else:
                print "But didn't have precomputed things I'm looking for yet. Looked on path", saved_path
                print

        print "Couldn't find, for these reference descriptors, any precomputed features."
        return False

    def get_type(self, save_type):
        if save_type == "features":
            return "_features.pth"
        elif save_type == "d_images":
            return "_d_images.pth"
        else:
            return ValueError("I only do features or d_images!")

    def load_up_saved(self, save_type, episodes, saved_episodes):
        """
        If we are calling this function,
        We have already determined a match in terms of: network name, save_type, and reference descriptors.
        Now we check episode names, and if that's good,
        Then we load features
        """
        print "saved_episodes length", len(saved_episodes)
        print "new episodes legnth", len(episodes)

        if len(saved_episodes) < len(episodes):
            print "had less saved episodes on disk! returning"
            return False

        for log_name in episodes.keys():
            if log_name not in saved_episodes:
                print "didn't have all matching saved_episodes!"
                return False

        for log_name in episodes.keys():
            print "LOADING LOG:", log_name
            episode = episodes[log_name]
            saved_episode = saved_episodes[log_name]

            if save_type == "features":
                episode.precomputed_features = dict()
                for key in saved_episode.precomputed_features.keys():
                    # the .pth converted the int key to a tensor key.
                    # here we just one by one use ints
                    episode.precomputed_features[int(key)] = saved_episode.precomputed_features[key]
            
            elif save_type == "d_images":
                episode.precomputed_descriptor_images = dict()
                for key in saved_episode.precomputed_descriptor_images.keys():
                    # the .pth converted the int key to a tensor key.
                    # here we just one by one use ints
                    episode.precomputed_descriptor_images[int(key)] = saved_episode.precomputed_descriptor_images[key]


        print "done loading"
        return True


    def save(self, save_type, config, reference_descriptor_vec, episodes, train_or_test):

        features_for_net_path = self.get_features_for_net_path(config)

        if not os.path.isdir(features_for_net_path):
            os.makedirs(features_for_net_path)

        new_index = 0
        previously_existing = sorted(os.listdir(features_for_net_path))
        if len(previously_existing) > 0:
            new_index = int(previously_existing[-1]) + 1

        new_index_str = str(new_index).zfill(3)

        new_folder = os.path.join(features_for_net_path, new_index_str)
        os.makedirs(new_folder)

        path_to_saved_reference_descriptors = os.path.join(new_folder, "reference_descriptors.pth") 
        torch.save(reference_descriptor_vec, path_to_saved_reference_descriptors)

        save_path = os.path.join(new_folder, train_or_test+self.get_type(save_type))
        print "Saving... (might take several seconds if this is images)"
        torch.save(episodes, save_path)
        print "Saved it all!"

    # in future should check for matching in config:
    # - use_hard_3D_unprojection
    # - use_soft_3D_unprojection
    # - 