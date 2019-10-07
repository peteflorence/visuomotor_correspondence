from __future__ import print_function

import dense_correspondence_manipulation.utils.utils as pdc_utils
pdc_utils.add_dense_correspondence_to_python_path()
from dense_correspondence.correspondence_tools import correspondence_finder

import torch
import numpy as np

def hard_pixels_to_3D_world(y_vision,  # N, 2*k, where x features are stacked on top of y features, cuda tensor
	                   depth,     # N, H, W, on cpu
	                   camera_to_world, # N, 4, 4 , on cpu
	                   K):              # N, 3, 3, on cpu
	N = y_vision.shape[0]
	k = y_vision.shape[1]/2

	y_vision_3d = torch.zeros(N, k*3).cuda()

	for i in range(N):
		for j in range(k):
			u_normalized_coordinate = y_vision[i,j]
			v_normalized_coordinate = y_vision[i,k+j]
			u_pixel_coordinate = ((u_normalized_coordinate/2.0 + 0.5)*640.0).long().detach().cpu().numpy()
			v_pixel_coordinate = ((v_normalized_coordinate/2.0 + 0.5)*480.0).long().detach().cpu().numpy()
			uv = (u_pixel_coordinate,v_pixel_coordinate)

			z = depth[i,v_pixel_coordinate,u_pixel_coordinate].numpy()

			K_one = K[i,:,:].numpy() 
			camera_to_world_one = camera_to_world[i,:,:].numpy()

			#print(uv, z, K_one, camera_to_world_one)
			point_3d_world = correspondence_finder.pinhole_projection_image_to_world_coordinates(uv, z, K_one, camera_to_world_one)
			y_vision_3d[i,j+0*k] = point_3d_world[0]
			y_vision_3d[i,j+1*k] = point_3d_world[1]
			y_vision_3d[i,j+2*k] = point_3d_world[2]
	
	return y_vision_3d

def compute_expected_z(softmax_activations, depth):
        """
        softmax_activations: N, nm, H, W
        depth: N, C=1, H, W
        """
        N = softmax_activations.shape[0]
        num_matches = softmax_activations.shape[1]
        expected_z = torch.zeros(N, num_matches).cuda()


        downsampled_depth = torch.nn.functional.interpolate(depth, scale_factor=1.0/8, mode='bilinear', align_corners=True)
        # N, C=1, H/8, W/8

        for i in range(N):
            one_expected_z = torch.sum((softmax_activations[i]*downsampled_depth[i,0]).unsqueeze(0), dim=(2,3)) #1, nm
            expected_z[i,:] = one_expected_z
        
        return expected_z

def soft_pixels_to_3D_world(y_vision,  # N, 2*k, where x features are stacked on top of y features, cuda tensor
					   sm_activations, # N, k, H, W (small H, W size)
	                   depth,     # N, H, W, on cpu (full H, W size)
	                   camera_to_world, # N, 4, 4 , on cpu
	                   K):              # N, 3, 3, on cpu
	N = y_vision.shape[0]
	k = y_vision.shape[1]/2

	y_vision_3d = torch.zeros(N, k*3).cuda()

	for i in range(N):
		for j in range(k):
			u_normalized_coordinate = y_vision[i,j]
			v_normalized_coordinate = y_vision[i,k+j]
			u_pixel_coordinate = ((u_normalized_coordinate/2.0 + 0.5)*640.0).long().detach().cpu().numpy()
			v_pixel_coordinate = ((v_normalized_coordinate/2.0 + 0.5)*480.0).long().detach().cpu().numpy()
			uv = (u_pixel_coordinate,v_pixel_coordinate)


			this_depth = (depth[i]).unsqueeze(0).unsqueeze(1).cuda() # convert to meters, from millimeters
			z = compute_expected_z(sm_activations[i,j].unsqueeze(0).unsqueeze(0), this_depth).cpu().detach().numpy()

			K_one = K[i,:,:].numpy() 
			camera_to_world_one = camera_to_world[i,:,:].numpy()

			# HACK TO MAKE THIS IN CAMERA FRAME!
			camera_to_world_one = np.eye(4)

			#print(uv, z, K_one, camera_to_world_one)
			point_3d_world = correspondence_finder.pinhole_projection_image_to_world_coordinates(uv, z, K_one, camera_to_world_one)
			y_vision_3d[i,j+0*k] = point_3d_world[0]
			y_vision_3d[i,j+1*k] = point_3d_world[1]
			y_vision_3d[i,j+2*k] = point_3d_world[2]
	
	return y_vision_3d