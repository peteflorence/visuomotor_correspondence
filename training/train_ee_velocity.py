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
from imitation_agent.loss_functions import loss_functions
from imitation_agent.dataset.statistics import compute_dataset_statistics
from imitation_agent.training import train_utils
from imitation_agent.training.optimizer_schedulers import NoamOpt, StepOpt
"""
Setup Configs
"""
spartan_source_dir = spartan_utils.getSpartanSourceDir()
imitation_src_dir = os.path.join(spartan_source_dir, "modules/imitation_agent")
data_dir = spartan_utils.get_data_dir()
logs_dir_path = os.path.join(data_dir, "pdc/imitation/logs_sugar")

# logs_config_yaml = os.path.join(spartan_source_dir,  "modules/imitation_agent/config/task/push_sugar.yaml")

#logs_config_yaml = os.path.join(spartan_source_dir,  "modules/imitation_agent/config/task/down_and_right.yaml")
logs_config_yaml = os.path.join(spartan_source_dir,  "modules/imitation_agent/config/task/move_to_box.yaml")
logs_config = spartan_utils.getDictFromYamlFilename(logs_config_yaml)


config_yaml = os.path.join(imitation_src_dir, "config", "model", "mlp_stateless_velocity.yaml")
config = spartan_utils.getDictFromYamlFilename(config_yaml)

N_EPOCH = 1000
LEARNING_RATE = 1e-4
BATCH_SIZE = 10
lambda_l1 = 1.0
WEIGHT_DECAY = 1e-4
EPS = 1e-4
RANDOM_SEED = 10

train_utils.make_deterministic()


def construct_dataset():
    action_function = ActionFunctionFactory.get_function(config)

    observation_function = ObservationFunctionFactory.get_function(config)

    dataset = ImitationEpisodeDataset(logs_dir_path,
                                      logs_config,
                                      config,
                                      action_function=action_function,
                                      observation_function=observation_function)
    return dataset


def construct_network():
    network = ModelFactory.get_model(config)
    return network


def save_network(save_dir, minibatch_counter, network, opt=None):
    minibatch_counter_str = str(minibatch_counter).zfill(6)
    filename = os.path.join(save_dir, "iteration-%s.pth" %(minibatch_counter_str))
    torch.save(network, filename)

def train(save_dir):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    tensorboard_dir = os.path.join(save_dir, "tensorboard")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    logger = tensorboard_logger.Logger(tensorboard_dir)

    dataset = construct_dataset()
    network = construct_network()
    dataset_stats = compute_dataset_statistics(dataset)

    network.set_normalization_parameters(dataset_stats)
    network.normalize_input = True

    # initialize reference_descriptors
    network.initialize_parameters_via_dataset(dataset)

    # don't require grad through vision net, if have a vision net
    if network._vision_net is not None:
        for params in network._vision_net.parameters():
            params.requires_grad = False

    # record some information for later
    # network._config = config I removed this since the class initialization now has the global config
    network.logs_config = logs_config

    # put network on the GPU
    network.cuda()

    # dataloader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    dataset_size = len(dataset)
    print("dataset_size", dataset_size)

    # learning rate scheduler
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [25, 50], gamma=0.1)
    opt_manager = StepOpt(config["lr"],
                          optimizer,
                          lr_decay=config["learning_rate_decay"],
                          steps_between_lr_decay=config["steps_between_learning_rate_decay"])

    scale_for_loss_function = 1.0/(dataset_stats['action']['std'].cuda()+EPS)

    minibatch_counter = 0
    log_rate = 100
    save_rate = config['save_rate_iteration'] # save every 100 epochs
    epoch_counter = 0

    prev_time = time.time()

    try:
        for epoch in xrange(N_EPOCH):
            epoch_counter = epoch # store it for use outside
            if epoch % 50 == 0:
                print("\n\n-----------\n---------")
                print("epoch = ", epoch)

            if minibatch_counter > config["global_training_steps"]:
                save_network(save_dir, minibatch_counter, network)
                break

            for idx, data in enumerate(dataloader):
                opt_manager.step()
                minibatch_counter += 1

                optimizer.zero_grad()
                data['observation'] = data['observation'].cuda()
                data['action'] = data['action'].cuda()
                y = data['action']
                data['image'] = data['image'].cuda()

                pred = network(data)

                # compute loss
                loss = 0

                # add L1 loss
                l1 = loss_functions.l1_scaled(pred, y, scale=scale_for_loss_function)
                loss += lambda_l1 * l1

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
                    logger.log_value("loss", loss.item(), minibatch_counter)

                    # compute loss if just predict 0
                    loss = 0
                    l1 = loss_functions.l1_scaled(y*0.0, y, scale=scale_for_loss_function)
                    loss += lambda_l1 * l1
                    logger.log_value("loss zero prediction", loss.item(), minibatch_counter)


            # save the network every so often
            if (minibatch_counter % save_rate) == 0:
                save_network(save_dir, minibatch_counter, network)

    except KeyboardInterrupt:
        # save network if we have a KeyboardInterrupt
        save_network(save_dir, minibatch_counter, network)

    save_network(save_dir, minibatch_counter, network)


if __name__ == "__main__":
    save_dir = os.path.join(data_dir, "pdc/imitation/trained_models/mlp_ee_vel", spartan_utils.get_current_YYYY_MM_DD_hh_mm_ss())
    train(save_dir)
