# system
import numpy

# torch
import torch

# ros
import rospy

# spartan
from robot_msgs.msg import CartesianGoalPoint

# imitation_agent
import imitation_agent.config.parameters as parameters

import imitation_agent.dataset.dataset_utils as dataset_utils


class EEVelocityAgent(object):

    def __init__(self, network, observation_function, imitation_episode):
        self.network = network
        self.network.eval()
        self.network.cuda()

        self._observation_function = observation_function
        self.imitation_episode = imitation_episode
        self.spartan_dataset = dataset_utils.build_spartan_dataset("")

    def compute_control_action(self, agent, #type ROSTaskSpaceControlAgent
                               ): # type -> (robot_msgs.msg.CartesianGoalPoint, float)

        data_list = agent.data_list
        self.imitation_episode.set_state_dict(data_list)


        idx = len(data_list) - 1
        obs = self._observation_function(self.imitation_episode, idx)
        y = None

        # extract RGB image
        image_data = agent.get_latest_images()
        rgb_tensor = self.spartan_dataset.rgb_image_to_tensor(image_data['rgb_image'])

        with torch.no_grad():
            # now unsqueeze (so it has batch_size 1) and push through the network
            x = torch.unsqueeze(obs, 0).cuda()

            input_data = {"observation": x}
            input_data["image"] = rgb_tensor.unsqueeze(0).cuda()
            # push it through the network
            y = self.network.forward(input_data)
            y = y.squeeze(0).cpu().numpy() # should be shape [6] (ang_vel, lin_vel)

        msg = CartesianGoalPoint()
        msg.ee_frame_id = parameters.ee_frame_id
        msg.use_end_effector_velocity_mode = True

        msg.linear_velocity.header.frame_id = "base"
        msg.linear_velocity.vector.x = y[3]
        msg.linear_velocity.vector.y = y[4]
        msg.linear_velocity.vector.z = y[5]

        msg.angular_velocity.header.frame_id = "base" # maybe should be world . . .
        msg.angular_velocity.vector.x = y[0]
        msg.angular_velocity.vector.y = y[1]
        msg.angular_velocity.vector.z = y[2]

        gripper_width = 0.0
        return msg, gripper_width




