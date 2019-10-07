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

# imitation_agent
from imitation_agent.dataset.imitation_episode_sequence_dataset import ImitationEpisodeSequenceDataset
from imitation_agent.model.model_factory import ModelFactory
from imitation_agent.training.optimizer_schedulers import NoamOpt, StepOpt
from imitation_agent.loss_functions import loss_functions
import imitation_agent.training.train_vis as train_vis
from imitation_agent.dataset.statistics import compute_sequence_dataset_statistics
from imitation_agent.dataset.function_factory import ObservationFunctionFactory, ActionFunctionFactory
from imitation_agent.training import train_utils



def construct_trainset_testset(config, logs_config, logs_dir_path):
    # obs_function = ObservationFunctionFactory.observation_from_config(config)
    obs_function = ObservationFunctionFactory.get_function(config)
    action_function = ActionFunctionFactory.action_from_config(config)

    trainset = ImitationEpisodeSequenceDataset(logs_dir_path,
                                      logs_config,
                                      config,
                                      action_function=action_function,
                                      observation_function=obs_function)

    test_logs_config = dict()
    test_logs_config["logs"] = logs_config["test_logs"]
    testset = ImitationEpisodeSequenceDataset(logs_dir_path,
                                      test_logs_config,
                                      config,
                                      action_function=action_function,
                                      observation_function=obs_function)


    # Not yet safe to do if using debug information like gt pose
    #trainset.set_force_state_same()
    #testset.set_force_state_same()

    return trainset, testset

def construct_network(config):
    network = ModelFactory.get_model(config)
    return network

def save_network(save_dir, global_iteration, start, network):
    print('Finished Training with', str(global_iteration), 'steps')
    print("In time", str(time.time() - start))
    global_iteration_str = str(global_iteration).zfill(6)
    filename = os.path.join(save_dir, "iter-%s.pth" %(global_iteration_str))
    torch.save(network, filename)
    print("saved network")

def construct_loss_criterion(config):
    if   config["regression_type"] == "direct":
        loss_criterion = loss_functions.l2_l1
    elif config["regression_type"] == "MDN":
        loss_criterion = loss_functions.NLL_MDN
    else:
        raise ValueError("unsupported loss type")

    return loss_criterion

def compute_test_loss(network, testset, testloader, logger, minibatch_counter, loss_criterion, config):
    print("Computing test loss at iter: ", minibatch_counter)
    network.eval()

    """
    Compute full test loss, averaged
    """
    losses = []
    counts = []

    for idx, data in enumerate(testloader):
    
        network.set_states_initial()

        poses = data['observations'].cuda() # poses.shape is [1, seq_length, observation_size]
        actions = data['actions'].cuda() # action.shape is [1, seq_length, action_size]
        rgb_images = data['images'].cuda()

        predictions = []

        sequence_length = poses.shape[1]
        assert sequence_length == actions.shape[1]
        num_chunks = sequence_length / config["truncated_backprop_length"] + 1

        poses_chunked = poses.chunk(num_chunks, dim=1)
        actions_chunked = actions.chunk(num_chunks, dim=1)
        images_chunked = rgb_images.chunk(num_chunks, dim=1)

        # print("num_chunks", num_chunks)

        # Truncated backpropagation
        for pose_chunk, action_chunk, image_chunk in zip(poses_chunked, actions_chunked, images_chunked):

            chunk_input_dict = dict()
            chunk_input_dict["observations"] = pose_chunk
            chunk_input_dict["images"] = image_chunk
            prediction_chunk = network.forward_on_series(chunk_input_dict).detach()

            # sum up loss
            chunk_length = data['observations'].shape[1]

            loss = loss_criterion(prediction_chunk, action_chunk, config, chunk_length)

            losses.append(loss.item())
            counts.append(data['observations'].shape[1])

    counts = np.asarray(counts)*1.0
    counts = counts/np.sum(counts)
    losses = np.asarray(losses)

    test_loss_average = np.dot(counts,losses) 

    logger.log_value("test_loss_average", test_loss_average, minibatch_counter)

    network.train()


def train(save_dir, config, logs_config, logs_dir_path):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    tensorboard_dir = os.path.join(save_dir, "tensorboard")
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    spartan_utils.saveToYaml(config, os.path.join(save_dir,"config.yaml"))

    logger = tensorboard_logger.Logger(tensorboard_dir)

    loss_criterion = construct_loss_criterion(config)
    test_loss_criterion = construct_loss_criterion(config)
    trainset, testset = construct_trainset_testset(config, logs_config, logs_dir_path)

    # LENGTH CHECK
    for log_name in trainset.episodes.keys():
        episode = trainset.episodes[log_name]
        if len(episode) < 50:
            print("No way this log should be so short!")
            print(log_name, len(episode))
            import sys; sys.exit(0)

    data = trainset.__getitem__(0)
    config["model"]["config"]["pose_size"] = data['observations'].shape[1]
    config["model"]["config"]["action_size"] = data['actions'].shape[1]
    print("observation size", data['observations'].shape[1])
    print("action size", data['actions'].shape[1])

    net = construct_network(config)
    net.initialize_parameters_via_dataset(trainset)


    # record some information for later
    #net._config = config
    net.logs_config = logs_config

    # put network on the GPU
    net.cuda()

    if net._vision_net is not None and config["precompute_features"]:
        print("precomputing trainset features")
        trainset.precompute_all_features(net._vision_net, "train")
        print("precomputing testset features")
        testset.precompute_all_features(net._vision_net, "test")
        
        net.set_use_precomputed_features()
        stats = compute_sequence_dataset_statistics(trainset, precomputed_features=True)
    elif net._vision_net is not None and config["precompute_descriptor_images"]:
        print("precomputing trainset descriptor images")
        trainset.precompute_all_descriptor_images(net._vision_net, "train")
        print("precomputing testset descriptor images")
        testset.precompute_all_descriptor_images(net._vision_net, "test")

        net.set_use_precomputed_descriptor_images()
        stats = compute_sequence_dataset_statistics(trainset, precomputed_d_images=True, vision_net=net._vision_net)
    else:
        stats = compute_sequence_dataset_statistics(trainset)

    # set the normalization parameters
    net.set_normalization_parameters(stats['observation']['mean'], stats['observation']['std'])

    # if surfing though, unfreeze the reference descriptors
    if config["freeze_vision"] and net._vision_net is not None and config["surf_descriptors"]:
        net.set_do_surfing()

    net.set_states_zero()

    # set network to train
    net.train()

    # dataloader
    trainloader = DataLoader(trainset, batch_size=config["batch_size"], shuffle=True, num_workers=config["num_workers"])
    trainset_size = len(trainset)
    print("trainset_size", trainset_size)

    testset.set_length(200)
    testloader = DataLoader(testset, batch_size=config["batch_size"], shuffle=False, num_workers=config["num_workers"])
    testset_size = len(testset)
    print("testset_size", testset_size)
    

    # learning rate scheduler
    optimizer = torch.optim.RMSprop(net.parameters(), lr=config["lr"], alpha=config["alpha"])
    opt_manager = StepOpt(config["lr"],
                          optimizer,
                          lr_decay=config["learning_rate_decay"],
                          steps_between_lr_decay=config["steps_between_learning_rate_decay"])


    start = time.time()
    global_iteration = 0
    prev_time = time.time()

    try:
        epoch = -1
        while True:
            epoch += 1

            if global_iteration > config["global_training_steps"]:
                save_network(save_dir, global_iteration, start, net)
                break

            for i, data in enumerate(trainloader, 0):

                global_iteration += 1

                net.set_states_initial()
                poses = data['observations'].cuda() # poses.shape is [1, seq_length, observation_size]
                actions = data['actions'].cuda() # action.shape is [1, seq_length, action_size]
                rgb_images = data['images'].cuda()

                predictions = []

                # diffs = actions - poses
                # Note: need to have same shape actions / poses to use diffs

                sequence_length = poses.shape[1]
                assert sequence_length == actions.shape[1]
                num_chunks = sequence_length / config["truncated_backprop_length"] + 1

                poses_chunked = poses.chunk(num_chunks, dim=1)
                actions_chunked = actions.chunk(num_chunks, dim=1)
                images_chunked = rgb_images.chunk(num_chunks, dim=1)

                # print("num_chunks", num_chunks)

                # Truncated backpropagation
                for pose_chunk, action_chunk, image_chunk in zip(poses_chunked, actions_chunked, images_chunked):
                    start_iter = time.time()
                    net.detach_parameters()

                    # zero the parameter gradients
                    opt_manager.optimizer.zero_grad()

                    # print("pose_chunk.shape", pose_chunk.shape)
                    # print("action_chunk.shape", action_chunk.shape)

                    # forward + backward + optimize

                    chunk_input_dict = dict()
                    chunk_input_dict["observations"] = pose_chunk
                    chunk_input_dict["images"] = image_chunk
                    prediction_chunk = net.forward_on_series(chunk_input_dict)


                    chunk_length = pose_chunk.shape[1]

                    if config["regression_type"] == "MDN":
                        loss, pi, sigma, mu = loss_criterion(prediction_chunk, action_chunk, config,
                                                             chunk_length)
                    elif config["regression_type"] == "direct":
                        loss = loss_criterion(prediction_chunk, action_chunk, config, chunk_length)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), config["clip_grad_norm"])
                    opt_manager.step()

                    end_iter = time.time()

                    if config["regression_type"] == "MDN":

                        pi_np, sigma_np, mu_np = pi.cpu().detach().numpy(), sigma.cpu().detach().numpy(), mu.detach().cpu().numpy()

                        def gumbel_sample(x, axis=1):
                            """
                            x shape: N, num_gaussians (numpy.ndarray)
                            returns: N,               (numpy.ndarray)
                            """
                            z = np.random.gumbel(loc=0, scale=1, size=x.shape)
                            return (np.log(x) + z).argmax(axis=axis)

                        k = gumbel_sample(pi_np ** 8)

                        L = pi_np.shape[0]
                        indices = (np.arange(L), k)
                        rn = np.random.randn(L, net.action_size)
                        sampled = rn * (sigma_np ** 8)[indices] + mu_np[indices]

                        predictions += [torch.from_numpy(sampled).unsqueeze(0)]
                    elif config["regression_type"] == "direct":
                        predictions += [prediction_chunk]


                predictions = torch.cat(predictions, 1)


                # print statistics
                if global_iteration % 50 == 0:  # print every 1000 mini-batches
                    print('[%d, %5d] loss: %.8f' % (epoch + 1, i + 1, loss.item()))

                    # get the learning rate
                    for p in optimizer.param_groups:
                        learning_rate = p['lr']
                        break

                    # log things to tensorboard
                    logger.log_value("learning rate", learning_rate, global_iteration)
                    logger.log_value("loss", loss.item(), global_iteration)


                if global_iteration % 100 == 0:  # plot
                    print('single-iter: ' + str(end_iter - start_iter) + ' seconds. plot iter:' + str(global_iteration))
                    print('lr: ' + str(opt_manager.rate()))

                    # train_vis.draw_signal_fitting_plots(poses, actions, predictions, config)

                if (global_iteration % config["test_loss_rate_iterations"]) == 0 and config["regression_type"] != "MDN":
                    opt_manager.optimizer.zero_grad()
                    compute_test_loss(net, testset, testloader, logger, global_iteration, test_loss_criterion, config)

                if global_iteration % config["save_rate_iteration"] == 0:
                    save_network(save_dir, global_iteration, start, net)

                if global_iteration > config["global_training_steps"]:
                    # just break here, will be saved in the outer break
                    break


    except KeyboardInterrupt:
        save_network(save_dir, global_iteration, start, net)


if __name__ == "__main__":
    """
    Setup Configs
    """
    import dense_correspondence_manipulation.utils.utils as pdc_utils
    pdc_utils.set_cuda_visible_devices([0])

    spartan_source_dir = spartan_utils.getSpartanSourceDir()
    imitation_src_dir = os.path.join(spartan_source_dir, "modules/imitation_agent")
    data_dir = spartan_utils.get_data_dir()


    logs_dir_path = os.path.join(data_dir, "pdc/imitation/grab_plate")

    logs_config_yaml = os.path.join(spartan_source_dir, "modules/imitation_agent/config/task/grab_plate.yaml")

    logs_config = spartan_utils.getDictFromYamlFilename(logs_config_yaml)


    #config_yaml = os.path.join(imitation_src_dir, "config", "model", "lstm_sequence_push_box.yaml")
    config_yaml = os.path.join(imitation_src_dir, "experiments", "lstm_grab_plate.yaml")
    config = spartan_utils.getDictFromYamlFilename(config_yaml)


    train_utils.make_deterministic()

    save_dir = os.path.join(data_dir, "pdc/imitation/trained_models/lstm_standard", spartan_utils.get_current_YYYY_MM_DD_hh_mm_ss())
    train(save_dir, config, logs_config, logs_dir_path)
