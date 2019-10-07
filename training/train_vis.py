import matplotlib.pyplot as plt
import numpy as np

def draw_signal_fitting_plots(poses, actions, predictions, config):
    """
    poses:       torch Tesnsor, shape: (N, L, pose_size) 
    actions:     torch Tesnsor, shape: (N, L, action_size)
    predicitons: torch Tesnsor, shape: (N, L, action_size)
    """
    plt.cla() # clear plot
    poses = poses[0,:].detach().cpu()
    x_hist = poses[:,0].numpy()
    y_hist = poses[:,1].numpy()
    z_hist = poses[:,2].numpy()

    actions   = actions[0,:].detach().cpu()
    x_actions = actions[:,0].numpy()
    y_actions = actions[:,1].numpy()
    z_actions = actions[:,2].numpy()

    predictions = predictions[0,:].detach().cpu()
    x_predictions = predictions[:,0].numpy()# + y_hist
    y_predictions = predictions[:,1].numpy()# + y_hist
    z_predictions = predictions[:,2].numpy()# + z_hist

    t = np.arange(0,predictions.shape[0],1)

    plt.plot(t,x_predictions, color="red",   ls="--", label="x_d_pred")
    plt.plot(t,x_actions,     color="red",   ls="-",  label="x_d")
    plt.plot(t,x_hist,        color="red",   ls=":",  label="x")

    plt.plot(t,y_predictions, color="blue",  ls="--", label="y_d_pred")
    plt.plot(t,y_actions,     color="blue",  ls="-",  label="y_d")
    plt.plot(t,y_hist,        color="blue",  ls=":",  label="y")
    
    plt.plot(t,z_predictions, color="green", ls="--", label="z_d_pred")
    plt.plot(t,z_actions,     color="green", ls="-",  label="z_d")
    plt.plot(t,z_hist,        color="green", ls=":",  label="z")
    
    plt.draw()
    plt.legend()
    plt.pause(0.001)


def draw_2d_trajectory_plots(poses, actions, predictions):
    """
    poses:       torch Tesnsor, shape: (N, L, pose_size) 
    actions:     torch Tesnsor, shape: (N, L, action_size)
    predicitons: torch Tesnsor, shape: (N, L, action_size)
    """
    plt.cla() # clear plot
    poses = poses[0,:].detach().cpu()
    y_hist = poses[:,0].numpy()
    z_hist = poses[:,1].numpy()
    plt.plot(y_hist, z_hist, color="green", lw=2)

    actions = actions[0,:].detach().cpu()
    y_actions = actions[:,0].numpy()
    z_actions = actions[:,1].numpy()
    plt.plot(y_actions, z_actions, color="black")

    predictions = predictions[0,:].detach().cpu()
    y_predictions = predictions[:,0].numpy()# + y_hist
    z_predictions = predictions[:,1].numpy()# + z_hist
    plt.plot(y_predictions, z_predictions, color="blue", ls="--")

    plt.draw()
    plt.pause(0.001)