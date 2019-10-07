import torch
import numpy as np


def compute_dataset_statistics(dataset, # type ImitationEpisodeDataset
                               num_samples=200):
    """
    Compute mean and std dev for observation and action
    :param num_samples:
    :type num_samples:
    :return:
    :rtype:
    """

    data = dataset.__getitem__(0)
    obs_dim = data['observation'].shape[0]
    action_dim = data['action'].shape[0]

    obs_tensor = torch.zeros([num_samples, obs_dim])
    action_tensor = torch.zeros([num_samples, action_dim])

    for i in xrange(0, num_samples):
        idx = np.random.randint(0, len(dataset))
        data = dataset.__getitem__(idx)

        obs_tensor[i, :] = data['observation']
        action_tensor[i, :] = data['action']

    # compute mean and std_dev
    stats = dict()
    stats['observation'] = dict()
    stats['observation']['mean'] = torch.mean(obs_tensor, 0)
    stats['observation']['std'] = torch.std(obs_tensor, 0)

    stats['action'] = dict()
    stats['action']['mean'] = torch.mean(action_tensor, 0)
    stats['action']['std'] = torch.std(action_tensor, 0)

    return stats


def compute_dataset_statistics_with_precomputed_features(dataset, # type ImitationEpisodeDataset
                               num_samples=200):
    """
    Compute mean and std dev for observation and action
    :param num_samples:
    :type num_samples:
    :return:
    :rtype:
    """

    data = dataset.__getitem__(0)
    obs_dim = data['observation'].shape[0] + data['image'].shape[0]
    action_dim = data['action'].shape[0]

    obs_tensor = torch.zeros([num_samples, obs_dim])
    action_tensor = torch.zeros([num_samples, action_dim])

    for i in xrange(0, num_samples):
        idx = np.random.randint(0, len(dataset))
        data = dataset.__getitem__(idx)

        y_vision = data['image']
        obs_tensor[i, :] = torch.cat((y_vision, data['observation']))

        action_tensor[i, :] = data['action']

    # compute mean and std_dev
    stats = dict()
    stats['observation'] = dict()
    stats['observation']['mean'] = torch.mean(obs_tensor, 0)
    stats['observation']['std'] = torch.std(obs_tensor, 0)

    stats['action'] = dict()
    stats['action']['mean'] = torch.mean(action_tensor, 0)
    stats['action']['std'] = torch.std(action_tensor, 0)

    return stats

def compute_dataset_statistics_with_precomputed_descriptor_images(dataset, # type ImitationEpisodeDataset,
                               vision_network, # type DonSpatialSoftmax
                               num_samples=200):
    """
    Compute mean and std dev for observation and action
    :param num_samples:
    :type num_samples:
    :return:
    :rtype:
    """
    print "Computing statistics for normalization... (with precomputed descriptor images)"

    data = dataset.__getitem__(0)
    data['depth'] = data['depth'].unsqueeze(0)
    data['K'] = data['K'].unsqueeze(0)
    data['camera_to_world'] = data['camera_to_world'].unsqueeze(0)
    y_vision = vision_network.get_expectations(data['image'].unsqueeze(0).cuda(), data).squeeze()
    obs_dim = data['observation'].shape[0] + y_vision.shape[0]
    action_dim = data['action'].shape[0]

    obs_tensor = torch.zeros([num_samples, obs_dim])
    action_tensor = torch.zeros([num_samples, action_dim])

    for i in xrange(0, num_samples):
        idx = np.random.randint(0, len(dataset))
        data = dataset.__getitem__(idx)

        data['depth'] = data['depth'].unsqueeze(0)
        data['K'] = data['K'].unsqueeze(0)
        data['camera_to_world'] = data['camera_to_world'].unsqueeze(0)

        y_vision = vision_network.get_expectations(data['image'].unsqueeze(0).cuda(), data)
        
        obs_tensor[i, :] = torch.cat((y_vision.squeeze().detach().cpu(), data['observation']))

        action_tensor[i, :] = data['action']

    # compute mean and std_dev
    stats = dict()
    stats['observation'] = dict()
    stats['observation']['mean'] = torch.mean(obs_tensor, 0)
    stats['observation']['std'] = torch.std(obs_tensor, 0)

    stats['action'] = dict()
    stats['action']['mean'] = torch.mean(action_tensor, 0)
    stats['action']['std'] = torch.std(action_tensor, 0)

    return stats


def compute_sequence_dataset_statistics(dataset, # type ImitationEpisodeSequenceDataset
                                        precomputed_features=False,
                                        precomputed_d_images=False,
                                        num_samples=200,
                                        vision_net=None):
    """
    Compute mean and std dev for observation and action when using a sequence type dataset
    :param num_samples:
    :type num_samples:
    :return:
    :rtype:
    """

    obs_tensor = None    # will be of shape (num_samples*seq_length, obs_dim)
    action_tensor = None # will be of shape (num_samples*seq_length, action_dim)

    for i in xrange(0, num_samples):
        idx = np.random.randint(0, len(dataset))
        data = dataset.__getitem__(idx)

        new_obs_tensor = data['observations']

        if precomputed_features:
            y_vision = data['images']
            new_obs_tensor = torch.cat((y_vision,new_obs_tensor), dim=1)

        if precomputed_d_images:
            y_vision = vision_net.get_expectations(data['images'].cuda(), data)
            new_obs_tensor = torch.cat((y_vision.detach().cpu(), new_obs_tensor), dim=1)

        if obs_tensor is None:
            obs_tensor = new_obs_tensor
        else:
            obs_tensor = torch.cat((obs_tensor, new_obs_tensor))

        if action_tensor is None:
            action_tensor = data['actions']
        else:
            action_tensor = torch.cat((action_tensor, data['actions']))

    print obs_tensor.shape, "is stats obs shape"
    print action_tensor.shape, "is stats action shape"

    # compute mean and std_dev
    stats = dict()
    stats['observation'] = dict()
    stats['observation']['mean'] = torch.mean(obs_tensor, 0)
    stats['observation']['std'] = torch.std(obs_tensor, 0)

    stats['action'] = dict()
    stats['action']['mean'] = torch.mean(action_tensor, 0)
    stats['action']['std'] = torch.std(action_tensor, 0)

    return stats