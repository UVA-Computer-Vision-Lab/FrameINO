import os, sys, shutil
from typing import List, Optional, Tuple, Union
from pathlib import Path
import csv
import random
import math
import numpy as np
import ffmpeg
import json
import imageio
import collections
import cv2
import pdb
csv.field_size_limit(sys.maxsize)      # Default setting is 131072, 100x expand should be enough

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



class VideoDataset_Motion(Dataset):

    def __init__(
        self,
        config,
        download_folder_path,
        csv_relative_path,
        video_relative_path,
        is_diy_test = False,
    ) -> None:
        super().__init__()

        # Gen Size Settings
        # self.height_range = config["height_range"]
        # self.max_aspect_ratio = config["max_aspect_ratio"]
        self.target_height = config["target_height"]
        self.target_width = config["target_width"]
        self.sample_accelerate_factor = config["sample_accelerate_factor"]
        self.train_frame_num_range = config["train_frame_num_range"]

        # Condition Settings (Text, Motion, etc.)
        self.empty_text_prompt = config["empty_text_prompt"]
        self.dot_radius = int(config["dot_radius"])                 
        self.point_keep_ratio = config["point_keep_ratio"]          # Point selection mechanism
        self.faster_motion_prob = config["faster_motion_prob"]

        # Other Settings
        self.download_folder_path = download_folder_path
        self.is_diy_test = is_diy_test
        self.config = config
        self.video_folder_path = os.path.join(download_folder_path, video_relative_path)
        csv_folder_path = os.path.join(download_folder_path, csv_relative_path)


        # Sanity Check
        assert(os.path.exists(csv_folder_path))
        assert(self.point_keep_ratio <= 1.0)
        


        # Read the CSV files
        info_lists = []
        for csv_file_name in os.listdir(csv_folder_path):       # Read all csv files
            csv_file_path = os.path.join(csv_folder_path, csv_file_name)
            
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
                                dot_radius, target_width, target_height, idx = 0, first_frame_img = None):

        # Prepare the color
        target_color_codes = all_color_codes[:len(full_pred_tracks[0])]        # This means how many objects in total we have

        # Prepare the traj image
        traj_img_lists = []

        # Set a new dot radius based on the resolution fluctuating
        dot_radius_resize = int( dot_radius * original_height / 384 )     # This is set with respect to default 384 height, will be adjust based on the height change

        # Prepare base draw image if there is
        if first_frame_img is not None:
            img_with_traj = first_frame_img.copy()

        # Iterate all temporal sequence
        merge_frames = []
        for temporal_idx, points_per_frame in enumerate(full_pred_tracks): # Iterate all downsampled frames, should be 13

            # Init the base img for the traj figures
            base_img = np.zeros((original_height, original_width, 3)).astype(np.float32)      # Use the original image size
            base_img.fill(255)      # Whole white frames

            # Iterate all points in each object
            for obj_idx, points_per_obj in enumerate(points_per_frame):

                # Basic setting
                color_code = target_color_codes[obj_idx]        # Color across frames should be consistent

                # Process all points in this current object
                for (horizontal, vertical) in points_per_obj:
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
            if len(selected_frames) != 0:
                merge_frame = selected_frames[temporal_idx].copy()
                merge_frame[base_img < 250] = base_img[base_img < 250]
                merge_frames.append(merge_frame)
            # cv2.imwrite("Video"+str(idx) + "_traj" + str(temporal_idx).zfill(2) + ".png", cv2.cvtColor(merge_frame, cv2.COLOR_RGB2BGR))       # Comment Out Later


            # Append to the temporal index
            traj_img_lists.append(base_img)
        

        # Convert to tensor
        traj_imgs_np = np.array(traj_img_lists)
        traj_tensor = torch.tensor(traj_imgs_np)

        # Transform
        traj_tensor = traj_tensor.float()
        traj_tensor = torch.stack([train_transforms(traj_frame) for traj_frame in traj_tensor], dim=0)
        traj_tensor = traj_tensor.permute(0, 3, 1, 2).contiguous()  # [F, C, H, W]

        
        # Write to video (Comment Out Later)
        # imageio.mimsave("merge_cond" + str(idx) + ".mp4",  merge_frames, fps=12)


        # Return
        merge_frames = np.array(merge_frames)
        if first_frame_img is not None: 
            return traj_tensor, traj_imgs_np, merge_frames, img_with_traj
        else:
            return traj_tensor, traj_imgs_np, merge_frames        # Need to return traj_imgs_np for other purpose 



    def __getitem__(self, idx):

        while True: # Iterate until there is a valid video read

            # try:

            # Fetch the information
            info = self.info_lists[idx]
            video_path = os.path.join(self.video_folder_path, info[self.element_idx_dict["video_path"]])
            original_height = int(info[self.element_idx_dict["height"]])
            original_width = int(info[self.element_idx_dict["width"]])
            # num_frames = int(info[self.element_idx_dict["num_frames"]])       # Deprecated, this is about the whole frame duration, not just one

            valid_duration = json.loads(info[self.element_idx_dict["valid_duration"]])  
            All_Frame_Panoptic_Segmentation = json.loads(info[self.element_idx_dict["Panoptic_Segmentation"]]) 
            text_prompt_all = json.loads(info[self.element_idx_dict["Structured_Text_Prompt"]])            
            Track_Traj_all = json.loads(info[self.element_idx_dict["Track_Traj"]])           # NOTE: Same as above, only consider the first panoptic segmented frame 
            Obj_Info_all = json.loads(info[self.element_idx_dict["Obj_Info"]]) 


            # Sanity check
            if not os.path.exists(video_path):
                raise Exception("This video path", video_path, "doesn't exists!")


            ########################################## Mangage Resolution and selected Clip Setting ##########################################

            # Option1: Variable Resolution Gen
            # # Check the resolution size
            # aspect_ratio = min(self.max_aspect_ratio, original_width / original_height)
            # target_height_raw = min(original_height, random.randint(*self.height_range))
            # target_width_raw = min(original_width, int(target_height_raw * aspect_ratio))
            # # Must be the multiplier of 32
            # target_height = (target_height_raw // 32) * 32
            # target_width = (target_width_raw // 32) * 32
            # print("New Height and Width are ", target_height, target_width)

            # Option2: Fixed Resolution Gen (Assume that the provided is 32x valid)
            target_width = self.target_width
            target_height = self.target_height


            # Only choose the first clip
            Obj_Info = Obj_Info_all[0]      # For the Motion Training, we have enough dataset, so we just choose the first panoptic section
            Track_Traj = Track_Traj_all[0]
            text_prompt = text_prompt_all[0]
            resolution = str(target_width) + "x" + str(target_height)       # Used for ffmpeg load
            frame_start_idx = Obj_Info[0][1]       # NOTE: If there is multiple objects Obj_Info[X][1] should be the same


            ##############################################################################################################################



            ############################################## Read the video by ffmpeg #################################################

            # Read the video by ffmpeg in the needed decode fps and resolution
            video_stream, err = ffmpeg.input(
                                                video_path
                                            ).output(
                                                "pipe:", format = "rawvideo", pix_fmt = "rgb24", s = resolution, vsync = 'passthrough',
                                            ).run(
                                                capture_stdout = True, capture_stderr = True    # If there is bug, command capture_stderr
                                            )    # The resize is already included
            video_np_full = np.frombuffer(video_stream, np.uint8).reshape(-1, target_height, target_width, 3)

            # Fetch the valid duration
            video_np = video_np_full[valid_duration[0] : valid_duration[1]]
            valid_num_frames = len(video_np)      # Update the number of frames


            # Decide the accelerate factor
            train_frame_num_raw = random.randint(*self.train_frame_num_range)
            if frame_start_idx + 3 * train_frame_num_raw < valid_num_frames and random.random() < self.faster_motion_prob:      # Should be (1) have enough frames and (2) in 10% probability
                sample_accelerate_factor = self.sample_accelerate_factor + 1       # Hard Code
            else:
                sample_accelerate_factor = self.sample_accelerate_factor


            # Check the number of frames needed this time
            frame_end_idx = min(valid_num_frames, frame_start_idx + sample_accelerate_factor * train_frame_num_raw)
            frame_end_idx = frame_start_idx + 4 * math.floor(( (frame_end_idx-frame_start_idx) - 1) / 4) + 1       # Rounded to the closest 4N + 1 size


            # Select Frames and Convert to Tensor
            selected_frames = video_np[ frame_start_idx : frame_end_idx : sample_accelerate_factor]       # NOTE: start from the first frame
            video_tensor = torch.tensor(selected_frames)   # Convert to tensor
            first_frame_np = selected_frames[0]    # Needs to return for Validation
            train_frame_num = len(video_tensor)      # Read the actual number of frames from the video (Must be 4N+1)


            # Data transforms and shape organize
            video_tensor = video_tensor.float() 
            video_tensor = torch.stack([train_transforms(frame) for frame in video_tensor], dim=0)
            video_tensor = video_tensor.permute(0, 3, 1, 2).contiguous()  # [F, C, H, W]
            

            #############################################################################################################################



            ######################################### Define the text prompt #######################################################

            # NOTE: text prompt is fetched above; here, we just decide if we you empty string
            if self.empty_text_prompt or random.random() < self.config["text_mask_ratio"]:
                text_prompt = ""
            # print("Text Prompt for Video", idx, " is ", text_prompt)
            
            ########################################################################################################################



            ###################### Prepare the Tracking points for each object (each object has different color) #################################

            # Iterate all the segmentation info
            full_pred_tracks = [[] for _ in range(train_frame_num)]   # The dim should be: (temporal, object, points, xy) The fps should be fixed to 12 fps, which is the same as training decode fps
            for track_obj_idx in range(len(Obj_Info)):

                # Read the basic info
                text_name, frame_idx_raw = Obj_Info[track_obj_idx]      # This is expected to be all the same in the video
                

                # Sanity Check: make sure that the number of frames is consistent
                if track_obj_idx > 0:
                    if frame_idx_raw != previous_frame_idx_raw:
                        raise Exception("The panoptic_frame_idx cannot pass the sanity check")


                # Prepare the tracjectory
                pred_tracks_full = Track_Traj[track_obj_idx]
                pred_tracks = pred_tracks_full[ frame_start_idx : frame_end_idx : sample_accelerate_factor]   
                if len(pred_tracks) != train_frame_num:
                    raise Exception("The length of tracking images does not match the video GT.")


                # Randomly select the points based on the prob given, here, the number of points is different for each objeects
                kept_point_status = random.choices([True, False], weights = [self.point_keep_ratio, 1 - self.point_keep_ratio], k = len(pred_tracks[0]))
                if len(kept_point_status) != len(pred_tracks[-1]):
                    raise Exception("The number of points filterred is not match with the dataset")


                # Iterate and add all temporally
                for temporal_idx, pred_track in enumerate(pred_tracks):

                    # Iterate all point one by one
                    left_points = []
                    for point_idx in range(len(pred_track)):
                        if kept_point_status[point_idx]:
                            left_points.append(pred_track[point_idx])
                    # Append the left points to the list
                    full_pred_tracks[temporal_idx].append(left_points)    # pred_tracks will be 49 frames, and each one represent all tracking points for single objects; only one object here


                # Other update
                previous_frame_idx_raw = frame_idx_raw


            # Draw the dilated traj points 
            traj_tensor, traj_imgs_np, merge_frames = self.prepare_traj_tensor(full_pred_tracks, original_height, original_width, selected_frames, 
                                                                                self.dot_radius, target_width, target_height, idx)

            # Sanity Check to make sure that the traj tensor and ground truth has the same number of frames
            if len(traj_tensor) != len(video_tensor):        # If this two cannot match, the torch.cat on latents will fail
                raise Exception("Traj length and Video length does not matched!")

            #########################################################################################################################################


            # except Exception as e:        # Note: You can uncomment this part to jump failure cases in mass training.
            #     print("The exception is ", e)
            #     old_idx = idx
            #     idx = (idx + 1) % len(self.info_lists)
            #     print("We cannot process the video", old_idx, " and we choose a new idx of ", idx)
            #     continue     # For any error occurs, we run it again with new idx proposed (a random int less than current value)


            # If everything is ok, we should break at the end
            break
        

        # Return the information
        return {
                    "video_tensor": video_tensor,
                    "traj_tensor": traj_tensor,
                    "text_prompt": text_prompt,

                    # The rest are auxiliary data for the validation/testing purposes
                    "video_gt_np": selected_frames,
                    "first_frame_np": first_frame_np,
                    "traj_imgs_np": traj_imgs_np,
                    "merge_frames": merge_frames,
                    "gt_video_path": video_path,
                }
    



    