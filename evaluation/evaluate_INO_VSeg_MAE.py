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
from sam2.sam2_video_predictor import SAM2VideoPredictor  




def INO_VSeg_MAE_evaluation(data_parent_path, region_target_height, region_target_width, test_num_frames):
    '''
        Args:
            region_target_height (int):     Height without the padding, the padding will be scaled together
            region_target_width (int):      Width without the padding, the padding will be scaled together
            test_num_frames (int): Sampled number of frames from the GT and GEN generated results
    ''' 

    # Init the Co-Tracker Model (Offline mode)
    device = "cuda"
    sam2_predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large")     # Large model has 224.4M Param at 39.5 FPS
    tmp_SAM2_gen_path = "tmp_SAM2_gen"
    tmp_SAM2_gt_path = "tmp_SAM2_gt"


    # Prepare the tmp folder
    if os.path.exists(tmp_SAM2_gen_path):
        shutil.rmtree(tmp_SAM2_gen_path)
    os.makedirs(tmp_SAM2_gen_path)
    if os.path.exists(tmp_SAM2_gt_path):
        shutil.rmtree(tmp_SAM2_gt_path)
    os.makedirs(tmp_SAM2_gt_path)
    

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
        sub_folder_path = os.path.join(data_parent_path, "instance"+str(instance_idx))


        # Prepare the folder needed
        tmp_store_folder_gen = os.path.join(tmp_SAM2_gen_path, "instance" + str(instance_idx))
        tmp_store_folder_gt = os.path.join(tmp_SAM2_gt_path, "instance" + str(instance_idx))
        if os.path.exists(tmp_store_folder_gen):
            shutil.rmtree(tmp_store_folder_gen)
        os.makedirs(tmp_store_folder_gen)
        if os.path.exists(tmp_store_folder_gt):
            shutil.rmtree(tmp_store_folder_gt)
        os.makedirs(tmp_store_folder_gt)


        # Read the main reference img
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

        # Scale
        scaled_top_left_x = int(top_left_x * scale_w)
        scaled_top_left_y = int(top_left_y * scale_h)
        scaled_bottom_right_x = int(bottom_right_x * scale_w)
        scaled_bottom_right_y = int(bottom_right_y * scale_h)
        scaled_canvas_width = int(canvas_width * scale_w)
        scaled_canvas_height = int(canvas_height * scale_h)


        # Resize the initial query points to the scaled canvas size
        first_frame_first_obj_points = GT_track_traj[0][0]        # We only consider the first frame and the first object points
        if len(first_frame_first_obj_points) == 0:
            print("Skip this one, because there is not points capable to be tracked from GT source")
            continue
        resized_first_frame_first_obj_points = [[int(scaled_canvas_width * cord_x/original_width), int(scaled_canvas_height * cord_y/original_height)] for (cord_x, cord_y) in first_frame_first_obj_points]


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

            # Add to list
            gen_padded_frames.append(gen_padded_frame)
            gt_padded_frames.append(gt_padded_frame)


            # Write to the tmp folder for the SAM2 tracking
            gen_store_path = os.path.join(tmp_store_folder_gen, str(order_idx).zfill(4) + ".jpg")
            cv2.imwrite(gen_store_path, cv2.cvtColor(gen_padded_frame, cv2.COLOR_BGR2RGB))

            gt_store_path = os.path.join(tmp_store_folder_gt, str(order_idx).zfill(4) + ".jpg")
            cv2.imwrite(gt_store_path, cv2.cvtColor(gt_padded_frame, cv2.COLOR_BGR2RGB))



        ############################################################## DO SAM for the generated video ##############################################################
        
        segmentation_masks_gen = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

            # Init the state
            state = sam2_predictor.init_state(tmp_store_folder_gen)
            sam2_predictor.reset_state(state)

            # Add new prompts and instantly get the output on the same frame
            labels = np.array([1] * len(resized_first_frame_first_obj_points), np.int32)  # All are the same label
            frame_idx, object_ids, masks = sam2_predictor.add_new_points_or_box(
                                                                                    state, 
                                                                                    frame_idx = 0,
                                                                                    obj_id = 1,     # Only consider single isntance now
                                                                                    points = resized_first_frame_first_obj_points,  # Use points in the first frame
                                                                                    labels = labels,
                                                                                )

             # Iterate all frames and Recognize multiple masks
            for frame_idx, object_ids, masks in sam2_predictor.propagate_in_video(state, start_frame_idx=0):
                for obj_idx, out_obj_id in enumerate(object_ids):

                    # Convert to boolean mask and 3 channels
                    segmentation_mask_raw = (masks[obj_idx] > 0.0).cpu().numpy().astype(np.uint8)
                    _, height, width = segmentation_mask_raw.shape
                    segmentation_masks_gen.append(segmentation_mask_raw[0])
                    

                    # Visualize
                    segmentation_mask_cat = np.stack([segmentation_mask_raw, segmentation_mask_raw, segmentation_mask_raw], axis=-1)[0]         # Range (0,1)  With 3 channels
                    merged_mask_gen = segmentation_mask_cat * cv2.cvtColor(gen_padded_frames[frame_idx], cv2.COLOR_BGR2RGB)
                    # cv2.imwrite("SAM2_gen"+str(frame_idx)+".png", merged_mask_gen)
                    
                    # We only set to have one object
                    break
                    
        # Clean folder
        shutil.rmtree(tmp_store_folder_gen)

        ##########################################################################################################################################################




        ############################################################## DO SAM for the GroundTruth video ##############################################################
        
        segmentation_masks_gt = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

            # Init the state
            state = sam2_predictor.init_state(tmp_store_folder_gt)
            sam2_predictor.reset_state(state)

            # Add new prompts and instantly get the output on the same frame
            labels = np.array([1] * len(resized_first_frame_first_obj_points), np.int32)  # All are the same label
            frame_idx, object_ids, masks = sam2_predictor.add_new_points_or_box(
                                                                                    state, 
                                                                                    frame_idx = 0,
                                                                                    obj_id = 1,     # Only consider single isntance now
                                                                                    points = resized_first_frame_first_obj_points,  # Use points in the first frame
                                                                                    labels = labels,
                                                                                )

             # Iterate all frames and Recognize multiple masks
            for frame_idx, object_ids, masks in sam2_predictor.propagate_in_video(state, start_frame_idx=0):
                for obj_idx, out_obj_id in enumerate(object_ids):

                    # Convert to boolean mask and 3 channels
                    segmentation_mask_raw = (masks[obj_idx] > 0.0).cpu().numpy().astype(np.uint8)
                    _, height, width = segmentation_mask_raw.shape
                    segmentation_masks_gt.append(segmentation_mask_raw[0])
                    

                    # Visualize
                    segmentation_mask_cat = np.stack([segmentation_mask_raw, segmentation_mask_raw, segmentation_mask_raw], axis=-1)[0]         # Range (0,1)  With 3 channels
                    merged_mask_gt = segmentation_mask_cat * cv2.cvtColor(gt_padded_frames[frame_idx], cv2.COLOR_BGR2RGB)
                    # cv2.imwrite("SAM2_gt"+str(frame_idx)+".png", merged_mask_gt)
                    
                    # We only set to have one object
                    break
                    
        # Clean folder
        shutil.rmtree(tmp_store_folder_gt)

        ##########################################################################################################################################################


        

        # Calculate the SAM area-based metrics
        all_frames_distances = []
        for temporal_idx in range(len(segmentation_masks_gen)):
            
            # Read
            gen_sam_mask = segmentation_masks_gen[temporal_idx]      # Only check the first object
            gt_sam_mask = segmentation_masks_gt[temporal_idx]

            # Crop based on Region Box 
            gen_sam_cropped_mask = gen_sam_mask[scaled_top_left_y:scaled_bottom_right_y, scaled_top_left_x:scaled_bottom_right_x]
            gt_sam_cropped_mask = gt_sam_mask[scaled_top_left_y:scaled_bottom_right_y, scaled_top_left_x:scaled_bottom_right_x]
        

            # Calcualte the mean absolute error
            num_gen_pixels_iniside_region = int(np.sum(gen_sam_cropped_mask))
            num_gt_pixels_inside_region = int(np.sum(gt_sam_cropped_mask))
            mae = abs(num_gen_pixels_iniside_region - num_gt_pixels_inside_region)
            mae_ratio = mae / (region_target_height * region_target_width)

            # Updates
            all_frames_distances.append(mae_ratio)


        # Per Video Updates
        total_score = sum(all_frames_distances)/len(all_frames_distances)
        print("The score in this video instance", instance_idx, " is ", total_score)
        print("Time spent is ", (time.time() - start_time)/60)
        all_video_score.append(total_score)




    # Clean the tmp folder
    shutil.rmtree(tmp_SAM2_gen_path)
    shutil.rmtree(tmp_SAM2_gt_path)



    # Calculate the average and then return 
    print("Effective case has", len(all_video_score))
    average_score = sum(all_video_score) / len(all_video_score)
    print("The total average score is ", average_score)



    return average_score


