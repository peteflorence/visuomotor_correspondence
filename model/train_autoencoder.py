import os
import dense_correspondence_manipulation.utils.utils as utils
utils.add_dense_correspondence_to_python_path()
utils.set_cuda_visible_devices([1])
from dense_correspondence.dataset.dynamic_spartan_dataset import DynamicSpartanDataset
from dense_correspondence.training.training import DenseCorrespondenceTraining

import torchvision

from imitation_agent.model.spatial_autoencoder import SpatialAutoencoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim.lr_scheduler import ExponentialLR,MultiStepLR

import torchvision
import matplotlib.pyplot as plt

from matplotlib import cm
import numpy as np
import random

import copy

import spartan.utils.utils as spartan_utils
import tensorboard_logger


torch.manual_seed(1)    # reproducible
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1)
random.seed(1)

"""
CONFIG
"""

# Whether or not to use slow loss
USE_SLOW_FEATURE_LOSS = True

# Whether or not the target is masked
USE_MASKED_DECODE_TARGET = False

# Whether to use dropout on the decoder
DECODER_DROPOUT = 0.1

CAMERA_NUM = 0

# Hyper Parameters
EPOCH = 50
BATCH_SIZE = 16
LR = 0.001
N_TEST_IMG = 5
DECAY_LR = 0.5

H, W = 240, 320 


config_filename = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence',
                                       'dataset', 'composite',
                                       'dynamic_sugar_move_to_box_se2_box_in_frame.yaml')
config = utils.getDictFromYamlFilename(config_filename)


"""
END CONFIG
"""


config["provide_gt_depths"] = True
train_data = DynamicSpartanDataset(config=config)
train_data.set_small_image_size(H,W)
train_data._domain_randomize = False


## this is not actually training config, but is needed to set some parameters in the dataset...
train_config_file = os.path.join(utils.getDenseCorrespondenceSourceDir(), 'config', 'dense_correspondence', 
                               'training', 'training.yaml')
train_config = utils.getDictFromYamlFilename(train_config_file)
train = DenseCorrespondenceTraining(dataset=train_data, config=train_config)
train.setup()
train_data._domain_randomize = False
train_data.set_as_autoencoder_image_loader(use_masked_decode_target=USE_MASKED_DECODE_TARGET, use_slow_loss=USE_SLOW_FEATURE_LOSS, camera_num=CAMERA_NUM)
print len(train_data[0])

test_data = copy.deepcopy(train_data)
test_data.set_test_mode()

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=12)
test_loader  = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=12)

autoencoder = SpatialAutoencoder(H,W,need_decoder=True,decoder_dropout=DECODER_DROPOUT)
autoencoder = autoencoder.cuda()

optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
lr_stepper = ExponentialLR(optimizer, gamma=DECAY_LR)
loss_func = nn.MSELoss()

save_dir = (spartan_utils.get_current_YYYY_MM_DD_hh_mm_ss())
tensorboard_dir = os.path.join(save_dir, "tensorboard")
if not os.path.exists(tensorboard_dir):
    os.makedirs(tensorboard_dir)

logger = tensorboard_logger.Logger(tensorboard_dir)


# original data (first row) for viewing
view_data = torch.zeros(N_TEST_IMG,3,H*W)
for i in range(N_TEST_IMG):
    print i
    img = train_data[0][0][0] # 3, 480, 640
    img = img.view(3,-1)
    view_data[i] = img

# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
plt.ion()   # continuously plot

# 5,flat
for i in range(N_TEST_IMG):
    a[0][i].imshow(np.reshape(view_data.data.permute(0,2,1).numpy()[i], (H, W, 3))); a[0][i].set_xticks(()); a[0][i].set_yticks(())
    plt.draw(); plt.pause(0.05)


overall_iter = 0

torch.save(autoencoder, os.path.join(save_dir,"dsae_start.pth"))

def compute_test_loss():
    autoencoder.eval()
    optimizer.zero_grad()
    test_iter = 0
    NUM_TO_AVERAGE = 100
    loss = 0
    for i, data, in enumerate(test_loader):
        test_iter += 1
        raw_imgs, recon_targets = data
        b_x = raw_imgs.view(-1, 3, H*W).cuda()  

        encoded, decoded, b_y, sm_activations = autoencoder(b_x)
        recon_targets = recon_targets.view(b_x.shape[0], -1).cuda()
        print decoded.shape, recon_targets.shape

        loss += loss_func(decoded.detach(), recon_targets)

        if test_iter >= NUM_TO_AVERAGE:
            break

    avg_test_loss =  loss.item() / (NUM_TO_AVERAGE*1.0)

    logger.log_value("mse_loss_autoencode_test", avg_test_loss, overall_iter)
    
    optimizer.zero_grad()
    autoencoder.train()



for epoch in range(EPOCH):
    print epoch, "is epoch"
    for i, data in enumerate(train_loader):
        overall_iter += 1

        raw_imgs, recon_targets = data

        batch_size = raw_imgs.shape[0]

        # the data comes in as triplets so we can potentially 
        # apply slow feature loss
        # this stacks them up so they are just one batch
        # but we can recover the pairs after forwarding
        raw_imgs      = torch.cat((raw_imgs[:,0,:,:,:], raw_imgs[:,1,:,:,:], raw_imgs[:,2,:,:,:]))
        recon_targets = torch.cat((recon_targets[:,0,:,:], recon_targets[:,1,:,:], recon_targets[:,2,:,:]))

        b_x = raw_imgs.view(-1, 3, H*W).cuda()  

        encoded, decoded, b_y, sm_activations = autoencoder(b_x)

        recon_targets = recon_targets.view(b_x.shape[0], -1).cuda()

        mse_loss = loss_func(decoded, recon_targets)
        logger.log_value("mse_loss_autoencode", mse_loss.item(), overall_iter)

        loss = mse_loss

        if USE_SLOW_FEATURE_LOSS:
            # unstack in a way so that we can apply slow feature loss
            unstacked_features_first  = encoded[0:batch_size]
            unstacked_features_second = encoded[batch_size:(2*batch_size)]
            unstacked_features_third  = encoded[(2*batch_size):]

            # each of the above is shape (N, K*2)

            delta_second = unstacked_features_third - unstacked_features_second
            delta_first  = unstacked_features_second - unstacked_features_first

            slow_loss = torch.norm(delta_second - delta_first, p=2)
            logger.log_value("slow_loss", slow_loss.item(), overall_iter)
            loss += slow_loss


        logger.log_value("total_loss_autoencode", loss.item(), overall_iter)

        optimizer.zero_grad()
        loss.backward()                   
        optimizer.step()

        if overall_iter % 250 == 1:
            print "do test eval"
            compute_test_loss()


        if i % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy())

            encoded_data, decoded_data = encoded, decoded


            plt.cla()
            for i in range(N_TEST_IMG):
                a[0][i].imshow(raw_imgs.data[i].cpu().permute(1,2,0).numpy())
                a[1][i].clear()
                a[1][i].imshow(np.reshape(decoded_data.data.cpu().numpy()[i], (60, 80)), cmap='gray')
                features = encoded_data[i].detach().cpu().numpy()
                print(features.shape, "is features shape")
                x_features = (features[:len(features)/2]+1.0)/2.0*80
                y_features = (features[len(features)/2:]+1.0)/2.0*60
                for j in range(len(x_features)):
                    y = int(y_features[j]*109.0/60)
                    x = int(x_features[j]*autoencoder.final_width/80)
                    sum_over_3_3_window = sm_activations[i,j,y-1:y+2,x-1:x+2].sum()
                    print(sum_over_3_3_window.item(), torch.max(sm_activations[i,j]).item())
                    if sum_over_3_3_window.item() > 0.2:
                        s = 20.0
                        marker='X'
                    else:
                        s = 2.0
                        marker='x'
                    a[1][i].scatter(x_features[j], y_features[j], s=s, marker=marker)
                a[1][i].set_xticks(()); a[1][i].set_yticks(())
            print(torch.max(sm_activations).item(), "overall max")
            plt.draw(); plt.pause(0.05)
    lr_stepper.step()
    torch.save(autoencoder, os.path.join(save_dir,"dsae_"+str(epoch)+".pth"))

print "done training"
plt.ioff()
plt.show()

print "saved"