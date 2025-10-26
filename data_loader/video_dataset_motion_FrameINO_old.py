import os, sys, shutil
from typing import List, Optional, Tuple, Union
from pathlib import Path
import csv
import random
import numpy as np
import ffmpeg
import json
import imageio
import collections
import cv2
import pdb
import math
import PIL.Image as Image
csv.field_size_limit(13107200)       # Default setting is 131072, 100x expand should be enough

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from utils.optical_flow_utils import flow_to_image, filter_uv, bivariate_Gaussian

# Init paramter and global shared setting

# Blurring Kernel
blur_kernel = bivariate_Gaussian(45, 3, 3, 0, grid = None, isotropic = True)

# Color
all_color_codes = [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), 
                    (255, 0, 255), (0, 0, 255), (128, 128, 128), (64, 224, 208),
                    (233, 150, 122)]
for _ in range(100):        # Should not be over 100 colors
    all_color_codes.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))

# Data Transforms
train_transforms = transforms.Compose(
                                        [
                                            transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
                                        ]
                                    )


class VideoDataset_Motion_FrameINO(Dataset):
    def __init__(
        self,
        config,
        csv_folder_path,
        FrameOut_only = False,
        one_point_one_obj = False,
        strict_validation_match = False,
    ) -> None:
        super().__init__()

        # Fetch the Fundamental Setting
        self.dataset_folder_path = config["dataset_folder_path"]
        if not FrameOut_only:   # Frame In mode
            self.ID_folder_path = config["ID_folder_path"]
        self.target_height = config["height"]
        self.target_width = config["width"]
        # self.ref_cond_size = config["ref_cond_size"]
        self.preset_decode_fps = config["preset_decode_fps"]        # Set to be 16
        self.train_frame_num = config["train_frame_num"]
        self.empty_text_prompt = config["empty_text_prompt"]
        self.start_skip = config["start_skip"]
        self.end_skip = config["end_skip"]
        self.dot_radius = int(config["dot_radius"])                 # Set to be 6
        self.point_keep_ratio_ID = config["point_keep_ratio_ID"]
        self.point_keep_ratio_regular = config["point_keep_ratio_regular"]
        self.faster_motion_prob = config["faster_motion_prob"]
        self.FrameOut_only = FrameOut_only  
        self.one_point_one_obj = one_point_one_obj      # Currently, this only open when FrameOut_only = True
        self.strict_validation_match = strict_validation_match
        self.config = config

        # Sanity Check
        assert(self.point_keep_ratio_ID <= 1.0)
        assert(self.point_keep_ratio_regular <= 1.0)


        # Read the CSV files
        info_lists = []
        for csv_file_name in os.listdir(csv_folder_path):       # Read all csv files
            csv_file_path  = os.path.join(csv_folder_path, csv_file_name)
            with open(csv_file_path) as file_obj: 
                reader_obj = csv.reader(file_obj) 
                
                # Iterate over each row in the csv  
                for idx, row in enumerate(reader_obj): 
                    if idx == 0:
                        elements = dict()
                        for element_idx, key in enumerate(row):
                            elements[key] = element_idx
                        continue

                    # Read the important information
                    info_lists.append(row)


        # Organize
        self.info_lists = info_lists
        self.element_idx_dict = elements

        # Log
        print("The number of videos for ", csv_folder_path, " is ", len(self.info_lists))
        # print("The memory cost is ", sys.getsizeof(self.info_lists))


    def __len__(self):
        return len(self.info_lists)


    @staticmethod
    def prepare_traj_tensor(full_pred_tracks, original_height, original_width, selected_frames, 
                                dot_radius, target_width, target_height, region_box, idx = 0, first_frame_img = None):

        # Prepare the color and other stuff
        target_color_codes = all_color_codes[:len(full_pred_tracks[0])]        # This means how many objects in total we have
        (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) = region_box

        # Prepare the traj image
        traj_img_lists = []

        # Set a new dot radius based on the resolution fluctuating
        dot_radius_resize = int( dot_radius * original_height / 384 )     # This is set with respect to default 384 height, will be adjust based on the height change

        # Prepare base draw image if there is
        if first_frame_img is not None:
            img_with_traj = first_frame_img.copy()

        # Iterate all object instance
        merge_frames = []
        for temporal_idx, obj_points in enumerate(full_pred_tracks): # Iterate all downsampled frames, should be 13

            # Init the base img for the traj figures
            base_img = np.zeros((original_height, original_width, 3)).astype(np.float32)      # Use the original image size
            base_img.fill(255)      # Whole white frames

            # Iterate for the per object
            for obj_idx, points in enumerate(obj_points):

                # Basic setting
                color_code = target_color_codes[obj_idx]        # Color across frames should be consistent


                # Process all points in this current object
                for (horizontal, vertical) in points:
                    if horizontal < 0 or horizontal >= original_width or vertical < 0 or vertical >= original_height:
                        continue    # If the point is already out of the range, Don't draw

                    # Draw square around the target position
                    vertical_start = min(original_height, max(0, vertical - dot_radius_resize))
                    vertical_end = min(original_height, max(0, vertical + dot_radius_resize))       # Diameter, used to be 10, but want smaller if there are too many points now
                    horizontal_start = min(original_width, max(0, horizontal - dot_radius_resize))
                    horizontal_end =  min(original_width, max(0, horizontal + dot_radius_resize))

                    # Paint
                    base_img[vertical_start:vertical_end, horizontal_start:horizontal_end, :] = color_code    

                    # Draw the visual of traj if needed
                    if first_frame_img is not None:  
                        img_with_traj[vertical_start:vertical_end, horizontal_start:horizontal_end, :] = color_code

            # Resize frames  Don't use negative and don't resize in [0,1]
            base_img = cv2.resize(base_img, (target_width, target_height), interpolation = cv2.INTER_CUBIC)
    
            # Dilate (Default to be True)
            base_img = cv2.filter2D(base_img, -1, blur_kernel).astype(np.uint8)

            # Append selected_frames and the color together for visualization
            merge_frame = selected_frames[temporal_idx].copy()
            merge_frame = cv2.rectangle(merge_frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 0, 0), 5)          # Draw the Region Box Area
            merge_frame[base_img < 250] = base_img[base_img < 250]
            merge_frames.append(merge_frame)


            # Append to the temporal index
            traj_img_lists.append(base_img)
        
        # Convert to tensor
        traj_imgs_np = np.array(traj_img_lists)
        traj_tensor = torch.tensor(traj_imgs_np) 

        # Transform
        traj_tensor = traj_tensor.float()
        traj_tensor = torch.stack([train_transforms(traj_frame) for traj_frame in traj_tensor], dim=0)
        traj_tensor = traj_tensor.permute(0, 3, 1, 2).contiguous()  # [F, C, H, W]


        # Write to video (For Debug Purpose)
        # imageio.mimsave("merge_cond" + str(idx) + ".mp4",  merge_frames, fps=12)


        # Return
        merge_frames = np.array(merge_frames)
        if first_frame_img is not None: 
            return traj_tensor, traj_imgs_np, merge_frames, img_with_traj
        else:
            return traj_tensor, traj_imgs_np, merge_frames        # Need to return traj_imgs_np for other purpose 




    def __getitem__(self, idx):

        while True: # Iterate until there is a valid video read

            try:

                # Fetch the information
                info = self.info_lists[idx]
                video_path = os.path.join(self.dataset_folder_path, info[self.element_idx_dict["video_path"]])
                original_height = int(info[self.element_idx_dict["height"]])
                original_width = int(info[self.element_idx_dict["width"]])
                num_frames = int(info[self.element_idx_dict["num_frames"]])
                fps = float(info[self.element_idx_dict["fps"]])

                # Fetch all panoptic frames
                FrameIN_info_all = json.loads(info[self.element_idx_dict["FrameIN_info"]])      
                Track_Traj_all = json.loads(info[self.element_idx_dict["Track_Traj"]])         
                text_prompt_all = json.loads(info[self.element_idx_dict["Improved_Text_Prompt"]])  
                ID_info_all = json.loads(info[self.element_idx_dict["ID_info"]])

                
                # Randomly Choose one available
                panoptic_idx = random.choice(range(len(FrameIN_info_all)))
                FrameIN_info = FrameIN_info_all[panoptic_idx]
                Track_Traj = Track_Traj_all[panoptic_idx]
                text_prompt = text_prompt_all[panoptic_idx]
                ID_info_panoptic = ID_info_all[panoptic_idx]


                # Organize
                resolution = str(self.target_width) + "x" + str(self.target_height)
                fps_scale = self.preset_decode_fps / fps
                downsample_num_frames = int(num_frames * fps_scale)


                # FrameIn drop
                if self.FrameOut_only or random.random() < self.config["drop_FrameIn_prob"]:
                    drop_FrameIn = True
                else:
                    drop_FrameIn = False


                    
                # Sanity check
                if not os.path.exists(video_path):
                    raise Exception("This video path ", video_path, " doesn't exists!")
                

                # Not all objects is ideal FrameIn, we need to select
                if not self.strict_validation_match:
                    effective_obj_idxs = []
                    for obj_idx, obj_info in enumerate(ID_info_panoptic):
                        if obj_info != []:
                            effective_obj_idxs.append(obj_idx)
                    main_target_obj_idx = random.choice(effective_obj_idxs)     # NOTE: I think we should only has one object to be processed for now
                else:
                    main_target_obj_idx = 0     # Always choose the first one

                #################################################### Fetch FrameIn ID information ###############################################################

                # Fetch the FrameIn ID info
                segmentation_info, useful_region_box = ID_info_panoptic[main_target_obj_idx]       # There might be multiple objects ideal, but we just randomly choose one
                if not self.FrameOut_only:
                    _, first_frame_reference_path, _ = segmentation_info     # bbox_info, first_frame_reference_path, store_img_path_lists
                    first_frame_reference_path = os.path.join(self.ID_folder_path, first_frame_reference_path)

                ##################################################################################################################################################



                ############ Randomly choose one mask inside the multiple choice available (Resolution is respect to the origional resolution) ############
                useful_region_box.sort(key=lambda x: x[0])

                # Choose one region box
                if not self.strict_validation_match:
                    mask_region = random.choice(useful_region_box[-5:])[1:]         # Choose in the largest 5 available
                else:
                    mask_region = useful_region_box[-1][1:]     # Choose the last one

                # Fetch
                (top_left_x_raw, top_left_y_raw), (bottom_right_x_raw, bottom_right_y_raw) = mask_region        # As Original Resolution

                # Resize the mask based on the CURRENT Target resolution (现在的384x480的resolution了)
                top_left_x = int(top_left_x_raw * self.target_width / original_width) 
                top_left_y = int(top_left_y_raw * self.target_height / original_height)
                bottom_right_x = int(bottom_right_x_raw * self.target_width / original_width) 
                bottom_right_y = int(bottom_right_y_raw * self.target_height / original_height)
                resized_mask_region_box = (top_left_x, top_left_y), (bottom_right_x, bottom_right_y)

                ###########################################################################################################################################



                ############################################## Read the video by ffmpeg #############################################################

                # Read the video by ffmpeg in the needed decode fps and resolution
                video_stream, err = ffmpeg.input(
                                                    video_path
                                                ).filter(
                                                    'fps', fps = self.preset_decode_fps, round = 'up'
                                                ).output(
                                                    "pipe:", format = "rawvideo", pix_fmt = "rgb24", s = resolution
                                                ).run(
                                                    capture_stdout = True, capture_stderr = True
                                                )      # The resize is already included
                video_np_raw = np.frombuffer(video_stream, np.uint8).reshape(-1, self.target_height, self.target_width, 3)

                # Sanity Check
                if len(video_np_raw) - self.start_skip - self.end_skip < self.train_frame_num:
                    raise Exception("The number of frames from the video is not enough")

                # Crop the tensor with all Non-interest region becomes blank(black-0 value); The region is target resolution in training with VAE step size adjustment
                video_np_masked = np.zeros(video_np_raw.shape, dtype = np.uint8)
                video_np_masked[:, top_left_y:bottom_right_y, top_left_x:bottom_right_x, :] = video_np_raw[:, top_left_y:bottom_right_y, top_left_x:bottom_right_x, :]

                #########################################################################################################################################



                ######################################### Define the text prompt #######################################################

                # Whether empty text prompt; Text Prompt already exists above 
                if self.empty_text_prompt or random.random() < self.config["text_mask_ratio"]:
                    text_prompt = ""

                ########################################################################################################################



                ###################### Prepare the Tracking points for each object (each object has different color) #################################
                
                # Make sure that the frame from the FrameIN_info has enough number of frames
                _, original_start_frame_idx, fps_scale = FrameIN_info[main_target_obj_idx]      # This is expected to be all the same in the video
                downsample_start_frame_idx = max(0, int(original_start_frame_idx * fps_scale))


                # Check the max number of frames available (NOTE: Recommended to use Full Text Prompt Version)
                max_step_num = (downsample_num_frames - downsample_start_frame_idx) // self.train_frame_num
                if max_step_num == 0:
                    print("This video is ", video_path)
                    raise Exception("The video is too short!")
                elif max_step_num >= 2 and random.random() < self.faster_motion_prob:
                    iter_gap = 2       # Maximum Setting now is 2x; else, the VAE might not works well
                else:
                    iter_gap = 1
                

                # Iterate all the Segmentation Info
                full_pred_tracks = [[] for _ in range(self.train_frame_num)]   # The dim should be: (temporal, object, points, xy) The fps should be fixed to 12 fps, which is the same as training decode fps

                # Iterate all objects but not the main objects
                for obj_idx in range(len(ID_info_panoptic)):

                    # Prepare the tracjectory
                    pred_tracks = Track_Traj[obj_idx]
                    pred_tracks = pred_tracks[downsample_start_frame_idx : downsample_start_frame_idx + iter_gap * self.train_frame_num : iter_gap]   
                    if len(pred_tracks) != self.train_frame_num:
                        raise Exception("The len of pre_track does not match")


                    # For Non-main obj idx, we must ensure all points inside the region box; If it is main obj, the ID must be outside the region box
                    if obj_idx != main_target_obj_idx or self.FrameOut_only:

                        # Randomly select the points based on the prob given, here, the number of points is different for each objeects
                        kept_point_status = random.choices([True, False], weights = [self.point_keep_ratio_regular, 1 - self.point_keep_ratio_regular], k = len(pred_tracks[0]))
                    
                        # Check witht the first frame, No need to check for following frames (allowed to have FrameOut effect)
                        first_frame_points = pred_tracks[0]
                        for point_idx in range(len(first_frame_points)):
                            (horizontal, vertical) = first_frame_points[point_idx]
                            if horizontal < top_left_x_raw or horizontal >= bottom_right_x_raw or vertical < top_left_y_raw or vertical >= bottom_right_y_raw: 
                                kept_point_status[point_idx] = False
                        
                    else:   # For main object

                        # Randomly select the points based on the prob given, here, the number of points is different for each objeects
                        if drop_FrameIn:
                            # No motion provided on ID for Drop FrameIn cases
                            kept_point_status = random.choices([False], k = len(pred_tracks[0]))
                
                        else:   # Regular FrameIn case
                            kept_point_status = random.choices([True, False], weights = [self.point_keep_ratio_ID, 1 - self.point_keep_ratio_ID], k = len(pred_tracks[0]))


                    # Sanity Check
                    if len(kept_point_status) != len(pred_tracks[-1]):
                        raise Exception("The number of points filterred is not match with the dataset")

                    # Iterate and add all temporally
                    for temporal_idx, pred_track in enumerate(pred_tracks):

                        # Iterate all point one by one
                        left_points = []
                        for point_idx in range(len(pred_track)):
                            # Select kept points
                            if kept_point_status[point_idx]:
                                left_points.append(pred_track[point_idx])

                        # Append the left points to the list
                        full_pred_tracks[temporal_idx].append(left_points)    # pred_tracks will be 49 frames, and each one represent all tracking points for single objects; only one object here

                # Fetch One Point
                if self.one_point_one_obj:
                    one_track_point = []
                    for full_pred_track_per_frame in full_pred_tracks:
                        one_track_point.append( [[full_pred_track_per_frame[0][0]]])

                #######################################################################################################################################



                ############################### Process the Video Tensor (based on info fetched from traj) ############################################

                # Select Frames based on the panoptic range (No Mask here)
                selected_frames = video_np_raw[downsample_start_frame_idx : downsample_start_frame_idx + iter_gap * self.train_frame_num : iter_gap]

                # Prepare the Video Tensor; NOTE: in this branch, video tensor is full image without mask
                video_tensor = torch.tensor(selected_frames)   # Convert to tensor
                if len(video_tensor) != self.train_frame_num:
                    raise Exception("The len of train frames does not match")

                # Training transforms for the Video and condition 
                video_tensor = video_tensor.float() 
                video_tensor = torch.stack([train_transforms(frame) for frame in video_tensor], dim=0)
                video_tensor = video_tensor.permute(0, 3, 1, 2).contiguous()  # [F, C, H, W]
                


                if drop_FrameIn:
                    main_reference_img = np.uint8(np.zeros((self.target_height, self.target_width, 3)))     # Whole Black (0-value) pixel placeholder

                else:

                    # Fetch the reference and resize
                    main_reference_img = np.asarray(Image.open(first_frame_reference_path))

                    # Resize to the same size as the video 
                    ref_h, ref_w = main_reference_img.shape[:2]
                    scale_h = self.target_height / max(ref_h, ref_w)
                    scale_w = self.target_width / max(ref_h, ref_w)
                    new_h, new_w = int(ref_h * scale_h), int(ref_w * scale_w)
                    main_reference_img = cv2.resize(main_reference_img, (new_w, new_h), interpolation = cv2.INTER_AREA)

                    # Calculate padding amounts on all direction
                    pad_height1 = (self.target_height - main_reference_img.shape[0]) // 2
                    pad_height2 = self.target_height - main_reference_img.shape[0] - pad_height1
                    pad_width1 = (self.target_width - main_reference_img.shape[1]) // 2
                    pad_width2 = self.target_width - main_reference_img.shape[1] - pad_width1

                    # Apply padding to same resolution as the training farmes
                    main_reference_img = np.pad(
                                                    main_reference_img, 
                                                    ((pad_height1, pad_height2), (pad_width1, pad_width2), (0, 0)), 
                                                    mode = 'constant', 
                                                    constant_values = 0
                                                )
                    # cv2.imwrite("main_reference_img_padded"+str(idx)+".png", cv2.cvtColor(main_reference_img, cv2.COLOR_BGR2RGB))


                # Convert to tensor
                main_reference_tensor = torch.tensor(main_reference_img)
                main_reference_tensor = train_transforms(main_reference_tensor).permute(2, 0, 1).contiguous()


                # Fetch the first frame and then do ID merge for this branch of training
                first_frame_np = video_np_masked[downsample_start_frame_idx]         # Needs to return for Validation
                # cv2.imwrite("first_frame"+str(idx)+".png", cv2.cvtColor(first_frame_np, cv2.COLOR_BGR2RGB))
                
                # Convert to Tensor and then Transforms
                first_frame_tensor = torch.tensor(first_frame_np)
                first_frame_tensor = train_transforms(first_frame_tensor).permute(2, 0, 1).contiguous()
                
                #######################################################################################################################################



                ############################################## Draw the Traj Points and Transform to Tensor #############################################

                # Draw the dilated points 
                if self.one_point_one_obj:
                    target_pred_tracks = one_track_point        # For this case, we only has one point per one object
                else:
                    target_pred_tracks = full_pred_tracks

                traj_tensor, traj_imgs_np, merge_frames = self.prepare_traj_tensor(target_pred_tracks, original_height, original_width, selected_frames, 
                                                                                    self.dot_radius, self.target_width, self.target_height, resized_mask_region_box, idx)

                #########################################################################################################################################


                # Write some processed meta data
                processed_meta_data = {
                                            "full_pred_tracks": full_pred_tracks,
                                            "original_width": original_width,
                                            "original_height": original_height,
                                            "mask_region": mask_region,
                                            "resized_mask_region_box": resized_mask_region_box,
                                        }

            except Exception as e:
                print("The exception is ", e)
                old_idx = idx
                idx = random.randint(0, len(self.info_lists))
                print("We cannot process the video", old_idx, " and we choose a new idx of ", idx)
                continue     # For any error occurs, we run it again with new idx proposed (a random int less than current value)


            # If everything is ok, we should break at the end
            break
        

        # Return the information
        return {
                    "video_tensor": video_tensor,
                    "traj_tensor": traj_tensor,
                    "first_frame_tensor": first_frame_tensor,
                    "main_reference_tensor": main_reference_tensor,
                    "text_prompt": text_prompt,

                    # The rest are auxiliary data for the validation/testing purposes
                    "video_gt_np": selected_frames,
                    "first_frame_np": first_frame_np,
                    "main_reference_np": main_reference_img,
                    "processed_meta_data": processed_meta_data,
                    "traj_imgs_np": traj_imgs_np,
                    "merge_frames" : merge_frames,
                    "gt_video_path": video_path,
                }
    



    