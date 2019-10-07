from __future__ import print_function
import os
import numpy as np
import random
import shutil
import time


# pytorch
import torch
from torch.utils.data import Dataset, DataLoader

# tensorboard_logger
import tensorboard_logger

# spartan
import spartan.utils.utils as spartan_utils


from imitation_agent.dataset.imitation_episode_dataset import ImitationEpisodeDataset
from imitation_agent.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory

from imitation_agent.model.model_factory import ModelFactory
from imitation_agent.model.spatial_autoencoder import SpatialAutoencoder
from imitation_agent.loss_functions import loss_functions
from imitation_agent.dataset.statistics import compute_dataset_statistics, compute_dataset_statistics_with_precomputed_vision
from imitation_agent.training import train_utils
from imitation_agent.training.optimizer_schedulers import NoamOpt, StepOpt

import dense_correspondence_manipulation.utils.utils as pdc_utils



def construct_trainset_testset(config, logs_config, logs_dir_path):
    
    # Trying ee position history option
    obs_function = ObservationFunctionFactory.get_function(config)
    #obs_function = ObservationFunctionFactory.observation_from_config(config)
    
    action_function = ActionFunctionFactory.action_from_config(config)

    trainset = ImitationEpisodeDataset(logs_dir_path,
                                      logs_config,
                                      config,
                                      action_function=action_function,
                                      observation_function=obs_function)

    test_logs_config = dict()
    test_logs_config["logs"] = logs_config["test_logs"]
    testset = ImitationEpisodeDataset(logs_dir_path,
                                      test_logs_config,
                                      config,
                                      action_function=action_function,
                                      observation_function=obs_function)


    # Not yet safe to do if using debug information like gt pose
    #trainset.set_force_state_same()
    #testset.set_force_state_same()

    return trainset, testset


def construct_pose_network(config):
    H, W = 240, 240 
    network = SpatialAutoencoder(H,W)
    return network


def save_network(save_dir, minibatch_counter, start, network, opt=None):
    print("Saving after time", str(time.time() - start))
    minibatch_counter_str = str(minibatch_counter).zfill(6)
    filename = os.path.join(save_dir, "iteration-%s.pth" %(minibatch_counter_str))
    torch.save(network, filename)
    print("saved network")

def compute_test_loss(network, testset, testloader, logger, minibatch_counter, config):
    print("Computing test loss at iter: ", minibatch_counter)
    network.eval()

    """
    Compute full test loss, averaged
    """
    l2_losses = []
    l1_losses = []
    counts = []

    for idx, data in enumerate(testloader):

        data['observation'] = data['observation'].cuda()
        data['action'] = data['action'].cuda()
        data['image'] = data['image'].cuda()

        pred = network.forward_for_pose(data['image']).detach()

        y = data['observation'][:,-9:] # if configured correctly, this will grab 3D points

        # sum up loss
        l2_losses.append(loss_functions.l2_scaled(pred, y).item())
        l1_losses.append(loss_functions.l1_scaled(pred, y).item())
        counts.append(data['observation'].shape[0])

    counts = np.asarray(counts)*1.0
    counts = counts/np.sum(counts)
    l2_losses = np.asarray(l2_losses)
    l1_losses = np.asarray(l1_losses)

    l2_test_loss_average = np.dot(counts,l2_losses) 
    l1_test_loss_average = np.dot(counts,l1_losses) 

    logger.log_value("l2_test_loss_average", l2_test_loss_average, minibatch_counter)
    logger.log_value("l1_test_loss_average", l1_test_loss_average, minibatch_counter)
    

    network.train()



def train(save_dir, config, logs_config, logs_dir_path):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # make sure this is kosher for pose training
    assert not config["observation"]["config"]["gripper"]["width"]
    assert "use_dynamic_gt_object_pose" not in config["observation"]["config"]
    assert "gt_object_points" in config['observation']["config"]
    assert len(config["observation"]["config"]["gt_object_points"]) == 3
    for point in config["observation"]["config"]["gt_object_points"]:
        assert len(point) == 3

    train_utils.make_deterministic()

    spartan_utils.saveToYaml(config, os.path.join(save_dir,"config.yaml"))

    tensorboard_dir = os.path.join(save_dir, "tensorboard")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    logger = tensorboard_logger.Logger(tensorboard_dir)

    trainset, testset = construct_trainset_testset(config, logs_config, logs_dir_path)

    data = trainset.__getitem__(0)
    config["model"]["config"]["num_inputs"] = data['observation'].shape[0]
    config["model"]["config"]["num_outputs"] = data['action'].shape[0]
    print("observation size", data['observation'].shape[0])
    print("action size", data['action'].shape[0])

    network = construct_pose_network(config)

    # record some information for later
    # network._config = config I removed this since the class initialization now has the global config
    network.logs_config = logs_config

    # put network on the GPU
    network.train()
    network.cuda()

    # dataloaders
    trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    trainset_size = len(trainset)
    print("trainset_size", trainset_size)

    testloader = DataLoader(testset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    testset_size = len(testset)
    print("testset_size", testset_size)

    # learning rate scheduler
    #optimizer = torch.optim.Adam(network.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    optimizer = torch.optim.RMSprop(network.parameters(), lr=config["lr"], alpha=config["alpha"])
    #optimizer = torch.optim.SGD(network.parameters(), lr=config["lr"], momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25, 50], gamma=0.1)
    opt_manager = StepOpt(config["lr"],
                          optimizer,
                          lr_decay=config["learning_rate_decay"],
                          steps_between_lr_decay=config["steps_between_learning_rate_decay"])


    epoch_counter = 0
    minibatch_counter = 0
    log_rate = 100
    save_rate = config['save_rate_iteration']

    start = time.time()
    prev_time = time.time()

    try:
        while True: # loop over epochs
            epoch_counter += 1

            if minibatch_counter > config["global_training_steps"]:
                break

            for idx, data in enumerate(trainloader):
                opt_manager.step()
                minibatch_counter += 1

                optimizer.zero_grad()
                data['observation'] = data['observation'].cuda()
                data['action'] = data['action'].cuda()
                data['image'] = data['image'].cuda()
                y = data['observation'][:,-9:] # if configured correctly, this will grab 3D points

                pred = network.forward_for_pose(data['image'])
                l2_loss = loss_functions.l2_scaled(pred,y)
                l1_loss = loss_functions.l1_scaled(pred,y)
                
                loss = l2_loss
                
                # backprop
                loss.backward()
                optimizer.step()

                if minibatch_counter % log_rate == 0:
                    print("\n\n----------------")
                    print("Step Counter = ", minibatch_counter)
                    print("Elapsed time", time.time() - prev_time)
                    prev_time = time.time()


                    # get the learning rate
                    for p in optimizer.param_groups:
                        learning_rate = p['lr']
                        break

                    print("Learning Rate = ", learning_rate)
                    print("Loss = ", loss.item())

                    logger.log_value("learning rate", learning_rate, minibatch_counter)
                    logger.log_value("loss",    loss.item(), minibatch_counter)
                    logger.log_value("l1_loss", l1_loss.item(), minibatch_counter)
                    logger.log_value("l2_loss", l2_loss.item(), minibatch_counter)
                    

                # save the network every so often
                if (minibatch_counter % save_rate) == 0:
                    save_network(save_dir, minibatch_counter, start, network)

                # eval the network every so often
                if (minibatch_counter % config["test_loss_rate_iterations"]) == 0:
                    optimizer.zero_grad()
                    compute_test_loss(network, testset, testloader, logger, minibatch_counter, config)

                if minibatch_counter > config["global_training_steps"]:
                    save_network(save_dir, minibatch_counter, start, network)
                    break



    except KeyboardInterrupt:
        # save network if we have a KeyboardInterrupt
        save_network(save_dir, minibatch_counter, start, network)

if __name__ == "__main__":

    pdc_utils.set_cuda_visible_devices([0])

    """
    Setup Configs
    """
    spartan_source_dir = spartan_utils.getSpartanSourceDir()
    imitation_src_dir = os.path.join(spartan_source_dir, "modules/imitation_agent")
    data_dir = spartan_utils.get_data_dir()
    logs_dir_path = os.path.join(data_dir, "pdc/imitation/move_to_box_0710")


    logs_config_yaml = os.path.join(spartan_source_dir,  "modules/imitation_agent/config/task/move_to_box_0710.yaml")
    logs_config = spartan_utils.getDictFromYamlFilename(logs_config_yaml)

    config_yaml = os.path.join(imitation_src_dir, "config", "model", "mlp_stateless_position.yaml")
    config = spartan_utils.getDictFromYamlFilename(config_yaml)

    logs_config_downsampled = train_utils.deterministic_downsample(logs_config, config["num_downsampled_logs"])

    save_dir = os.path.join(data_dir, "pdc/imitation/trained_models/mlp_position", spartan_utils.get_current_YYYY_MM_DD_hh_mm_ss())
    train(save_dir, config, logs_config_downsampled, logs_dir_path) 