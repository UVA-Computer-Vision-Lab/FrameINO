'''
    Evaluate the metrics
'''
import os, sys, shutil
import random
import cv2
import pickle
import math
import imageio
import time
import numpy as np
from PIL import Image, ImageDraw
import gc
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
import torch.nn.functional as F
import torchvision.transforms as transforms
# from sam2.sam2_video_predictor import SAM2VideoPredictor 



def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance


def visualize_track(frames, pred_tracks, instance_idx, dot_radius_resize = 6):
    
    # Define the color code
    all_color_codes = [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), 
                        (255, 0, 255), (0, 0, 255), (128, 128, 128), (64, 224, 208),
                        (233, 150, 122)]
    for _ in range(100):        # Should not be over 100 colors
        all_color_codes.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))


    # Prediction Tracks
    drawn_frames = []
    for frame_idx in range(len(pred_tracks)):
        frame = frames[frame_idx].copy()
        original_height, original_width, _ = frame.shape

        for points_idx in range(len(pred_tracks[0])):
            coord = (pred_tracks[frame_idx, points_idx, 0], pred_tracks[frame_idx, points_idx, 1])

            if coord[0] != 0 and coord[1] != 0:
                
                # Prepare
                color_code = all_color_codes[points_idx % len(all_color_codes)]
                horizontal, vertical = coord
                
                # Draw square around the target position
                vertical_start = min(original_height, max(0, vertical - dot_radius_resize))
                vertical_end = min(original_height, max(0, vertical + dot_radius_resize))       # Diameter, used to be 10, but want smaller if there are too many points now
                horizontal_start = min(original_width, max(0, horizontal - dot_radius_resize))
                horizontal_end =  min(original_width, max(0, horizontal + dot_radius_resize))
                
                frame[vertical_start:vertical_end, horizontal_start:horizontal_end, :] = color_code    
        
        drawn_frames.append(frame)

    imageio.mimsave("merge_visual" + str(instance_idx) + ".mp4",  drawn_frames, fps=12)



def INO_Traj_evaluation(data_parent_path, region_target_height, region_target_width, test_num_frames):
    '''
        Args:
            region_target_height (int):     Height without the padding, the padding will be scaled together
            region_target_width (int):      Width without the padding, the padding will be scaled together
            test_num_frames (int): Sampled number of frames from the GT and GEN generated results
            
    '''

    # Init the Co-Tracker Model (Offline mode)
    device = "cuda"
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)



    # Calculate the MAX nubmer of frames in video folder
    total_gen_num_frames_one_video, total_gt_num_frames_one_video = 0, 0
    for file_name in sorted(os.listdir(os.path.join(data_parent_path, "instance0"))):
        if file_name.find("gen_frame") != - 1:
            total_gen_num_frames_one_video += 1
    
        if file_name.find("gt_frame") != - 1:
            total_gt_num_frames_one_video += 1
    print("We have total gen and gt number in one video of frames of ", total_gen_num_frames_one_video, total_gt_num_frames_one_video)

    
    # Get the index
    gen_indices = np.linspace(0, total_gen_num_frames_one_video - 1, test_num_frames, dtype=int)     
    gt_indices = np.linspace(0, total_gt_num_frames_one_video - 1, test_num_frames, dtype=int)
    assert(len(gen_indices) == test_num_frames)
    assert(len(gt_indices) == test_num_frames)
    


    # Iterate each sub folder
    all_video_score = []
    start_time = time.time()
    for instance_idx in range(len(os.listdir(data_parent_path))):

        
        # Define the path
        sub_folder_path = os.path.join(data_parent_path, "instance"+str(instance_idx))


        # Read the Important information from processed_meta_data store inside the folder
        processed_meta_data_store_path = os.path.join(sub_folder_path, "processed_meta_data.pkl")
        assert(os.path.exists(processed_meta_data_store_path))
        with open(processed_meta_data_store_path, 'rb') as file:
            processed_meta_data = pickle.load(file)

        # Fetch information
        GT_track_traj = processed_meta_data["full_pred_tracks"]
        original_height = int(processed_meta_data["original_height"])
        original_width = int(processed_meta_data["original_width"])
        resized_mask_region_box = processed_meta_data["resized_mask_region_box"]
        
        # Read sample
        sample_GT_frame = cv2.imread(os.path.join(sub_folder_path, "gt_padded_frame0.png"))
        canvas_height, canvas_width, _  = sample_GT_frame.shape 

        # NOTE: Define the new size, which is based on the region box to the target width and height
        (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) = resized_mask_region_box        # already resized
        region_height = bottom_right_y - top_left_y
        region_width = bottom_right_x - top_left_x
        scale_w = region_target_width / region_width
        scale_h = region_target_height / region_height
        scaled_canvas_width = int(canvas_width * scale_w)
        scaled_canvas_height = int(canvas_height * scale_h)


        # Resize the initial query points to the scaled canvas size
        first_frame_first_obj_points = GT_track_traj[0][0]        # We only consider the first frame and the first object points
        if len(first_frame_first_obj_points) == 0:
            print("Skip this one, because there is not points capable to be tracked from GT source")
            continue
        queries = [[float(0), int(scaled_canvas_width * cord_x/original_width), int(scaled_canvas_height * cord_y/original_height)] for (cord_x, cord_y) in first_frame_first_obj_points]
        queries = torch.tensor(queries).to(device)



        # Iterate the sampled frame
        gen_padded_frames, gt_padded_frames = [], []
        for order_idx in range(test_num_frames):

            # Read the path
            gen_padded_frame_path = os.path.join(sub_folder_path, "gen_padded_frame"+str(gen_indices[order_idx])+".png")       # This will be arbitrary resolution and needs to be resized
            assert(os.path.exists(gen_padded_frame_path))
            gt_padded_frame_path = os.path.join(sub_folder_path, "gt_padded_frame"+str(gt_indices[order_idx])+".png")       # This will be arbitrary resolution and needs to be resized
            assert(os.path.exists(gt_padded_frame_path))


            # Read and Transforms
            gen_padded_frame = cv2.cvtColor(cv2.imread(gen_padded_frame_path), cv2.COLOR_RGB2BGR)
            gt_padded_frame = cv2.cvtColor(cv2.imread(gt_padded_frame_path), cv2.COLOR_RGB2BGR)

            # Resize
            gen_padded_frame = cv2.resize(gen_padded_frame, (scaled_canvas_width, scaled_canvas_height))        # The height and width for tracking, all methods is uniformed on the same
            gt_padded_frame = cv2.resize(gt_padded_frame, (scaled_canvas_width, scaled_canvas_height))


            # Updates
            gen_padded_frames.append(gen_padded_frame)
            gt_padded_frames.append(gt_padded_frame)


            
        # Tracking based on the initial points and all generated video frames
        gen_padded_video_tensor = torch.tensor(gen_padded_frames).permute(0, 3, 1, 2)[None].float().to(device)
        pred_tracks, pred_visibility = cotracker(gen_padded_video_tensor, queries = queries[None], backward_tracking = False)          # B T N 2,  B T N 1
        pred_tracks = pred_tracks[0].long().detach().cpu().numpy()      # return shape is (temporal, sptial_points, 2)
        pred_visibility = pred_visibility[0].detach().cpu().numpy()     # return shape is (temporal, sptial_points)
        assert(len(pred_tracks) == len(pred_visibility))

        # Tracking based on the initial points and all GT video frames
        gt_padded_video_tensor = torch.tensor(gt_padded_frames).permute(0, 3, 1, 2)[None].float().to(device)
        gt_tracks, gt_visibility = cotracker(gt_padded_video_tensor, queries = queries[None], backward_tracking = False)          # B T N 2,  B T N 1
        gt_tracks = gt_tracks[0].long().detach().cpu().numpy()      # return shape is (temporal, sptial_points, 2)
        gt_visibility = gt_visibility[0].detach().cpu().numpy()     # return shape is (temporal, sptial_points)
        assert(len(gt_tracks) == len(gt_visibility))



        # Visualize the tracking results
        # visualize_track(gen_padded_frames, pred_tracks, instance_idx)


        # Calculate the traj error
        all_frames_distances = []
        for temporal_idx, pred_points in enumerate(pred_tracks):
            
            # Read
            gt_points = gt_tracks[temporal_idx]      # Only check the first object

            # Iterate
            per_frame_distances = []
            for point_idx in range(len(gt_points)):
                gt_point = gt_points[point_idx]
                pred_point = pred_points[point_idx]

                # Calculate Euclidean Distance
                distance = euclidean_distance(pred_point, gt_point)
                per_frame_distances.append(distance)
            
            all_frames_distances.append( sum(per_frame_distances) / len(per_frame_distances))



        # Per Video Updates
        score = sum(all_frames_distances) / len(all_frames_distances)
        all_video_score.append(score)

        print("Finish Instance ", instance_idx, " with score ", score)
        print("Time spent from the beginning is ", (time.time() - start_time)/60, "Min")


    print("Number of lists effective is",len(all_video_score))
    average_score = sum(all_video_score) / len(all_video_score)
    return average_score
