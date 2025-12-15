
'''
    This file is trying to use Co-Tracker to track random points and then filter
'''

import os, sys, shutil
import argparse
import numpy as np
from PIL import Image, ImageDraw
import cv2
import random
import ffmpeg
import time
import math
import pandas as pd
import json
import csv
import imageio
import torch
csv.field_size_limit(sys.maxsize)


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from utils.optical_flow_utils import flow_to_image, filter_uv, bivariate_Gaussian


# Blurring Kernel
blur_kernel = bivariate_Gaussian(45, 3, 3, 0, grid = None, isotropic = True)

# Color
all_color_codes = [(255, 0, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), 
                    (255, 0, 255), (0, 0, 255), (128, 128, 128), (64, 224, 208),
                    (233, 150, 122)]
for _ in range(100):        # Should not be over 100 colors
    all_color_codes.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))




def prepare_traj_tensor(full_pred_tracks, full_visibility_tracks, original_height, original_width, selected_frames, 
                                dot_radius, target_width, target_height, encode_fps, name = ""):

    # Prepare the color
    target_color_codes = all_color_codes[:len(full_pred_tracks[0])]        # This means how many objects in total we have
    invisible_color_code = (0, 0, 0)


    # Set a new dot radius based on the resolution fluctuating
    dot_radius_resize = int( dot_radius * original_height / 384 )     # This is set with respect to default 384 height, will be adjust based on the height change


    # Iterate all object instance
    imgs = []
    for temporal_idx, obj_points in enumerate(full_pred_tracks): # Iterate all downsampled frames, should be 13

        # Init the base img for the traj figures
        base_img = np.zeros((original_height, original_width, 3)).astype(np.float32)      # Use the original image size
        base_img.fill(255)      # Whole white frames

        # Prepare visibility points
        visible_masks = full_visibility_tracks[temporal_idx]


        # Iterate for the per object
        for obj_idx, points in enumerate(obj_points):

            # Basic setting
            color_code = target_color_codes[obj_idx]        # Color across frames should be consistent
            visible_mask = visible_masks[obj_idx]

            # Process all points in this current object
            for point_idx, (horizontal, vertical) in enumerate(points):
                if horizontal < 0 or horizontal >= original_width or vertical < 0 or vertical >= original_height:
                    continue    # If the point is already out of the range, Don't draw
                
                is_visible = visible_mask[point_idx]

                # Draw square around the target position
                vertical_start = min(original_height, max(0, vertical - dot_radius_resize))
                vertical_end = min(original_height, max(0, vertical + dot_radius_resize))       # Diameter, used to be 10, but want smaller if there are too many points now
                horizontal_start = min(original_width, max(0, horizontal - dot_radius_resize))
                horizontal_end =  min(original_width, max(0, horizontal + dot_radius_resize))

                # Paint
                if is_visible:
                    base_img[vertical_start:vertical_end, horizontal_start:horizontal_end, :] = color_code  
                else:
                    base_img[vertical_start:vertical_end, horizontal_start:horizontal_end, :] = invisible_color_code  


        # Resize frames  Don't use negative and don't resize in [0,1]
        base_img = cv2.resize(base_img, (target_width, target_height), interpolation = cv2.INTER_CUBIC)

        # Dilate (Default to be True)
        # base_img = cv2.filter2D(base_img, -1, blur_kernel).astype(np.uint8)


        # Append selected_frames and the color together for visualization
        merge_frame = selected_frames[temporal_idx].copy()
        merge_frame[base_img < 250] = base_img[base_img < 250]
        # cv2.imwrite("Video" + name + "_traj" + str(temporal_idx).zfill(2) + ".png", cv2.cvtColor(merge_frame, cv2.COLOR_RGB2BGR)) # The base_img is designed in RGB form
        imgs.append(merge_frame)
    
        
    # Write to video (For Debug Purpose)
    # os.system("ffmpeg -r " + str(encode_fps) + " -f image2 -i Video"+name+"_traj%02d.png -vcodec libx264 -crf 30 -pix_fmt yuv420p gen_video" + name + ".mp4")

    # Clean frames
    # os.system("rm Video" + name + "_traj*.png")


    # Imageio save
    imageio.mimsave("gen_video"+name+".mp4", imgs, fps = encode_fps)



def visualize_motion(Track_Traj, Object_of_Interest, Track_Visibility, panoptic_frame_idx, process_last_idx, 
                        height, width, fps, video_np, dot_radius, name):

    
    # Convert Track_Traj to the data form of (temporal, object, points, xy)
    train_frame_num = process_last_idx - panoptic_frame_idx
    full_pred_tracks = [[] for _ in range(train_frame_num)]   # The dim should be: (temporal, object, points, xy) The fps should be fixed to 12 fps, which is the same as training decode fps
    full_visibility_tracks = [[] for _ in range(train_frame_num)]

    # Iterate all objects in current panoptic frame
    for obj_idx in range(len(Object_of_Interest)):

        # Read the basic info
        text_name, original_start_frame_idx = Object_of_Interest[obj_idx]      # This is expected to be all the same in the video
        
        # Sanity Check: make sure that the number of frames is consistent
        if obj_idx > 0:
            if original_start_frame_idx != previous_panoptic_frame_idx:
                raise Exception("The panoptic_frame_idx cannot pass the sanity check")
        panoptic_frame_idx = original_start_frame_idx


        # Prepare the tracjectory
        pred_tracks = Track_Traj[obj_idx]
        visibility_tracks = Track_Visibility[obj_idx]

        # Fetch the ideal range needed
        pred_tracks = pred_tracks[panoptic_frame_idx : panoptic_frame_idx + train_frame_num ]   
        visibility_tracks = visibility_tracks[panoptic_frame_idx : panoptic_frame_idx + train_frame_num]
        if len(pred_tracks) != train_frame_num:
            raise Exception("The len of pre_track does not match")
        for temporal_idx, pred_track in enumerate(pred_tracks):
            full_pred_tracks[temporal_idx].append(pred_track)    # pred_tracks will be 49 frames, and each one represent all tracking points for single objects; only one object here
            full_visibility_tracks[temporal_idx].append(visibility_tracks[temporal_idx])

        # Other update
        previous_panoptic_frame_idx = original_start_frame_idx

    # Choose the selected frame range
    selected_frames = video_np[panoptic_frame_idx : panoptic_frame_idx + train_frame_num]

    # Call the Drawing function     (Will also do ffmpeg encode inside)
    prepare_traj_tensor(full_pred_tracks, full_visibility_tracks, height, width, selected_frames, dot_radius, width, height, fps, name = name)




@torch.no_grad()
def single_process(csv_folder_path, store_folder_path, GPU_offset, min_process_frame, max_process_frame):
    
    # Setting
    store_freq = 10
    device = 'cuda'


    # Read the csv file
    csv_idx = GPU_offset
    csv_file_path = os.path.join(csv_folder_path, "sub" + str(csv_idx) + ".csv")
    print("CSV file we read is ", csv_file_path)


    # Prepare the store file path
    store_file_path = os.path.join(store_folder_path, "sub" + str(csv_idx) + ".csv")
    if not resume and os.path.exists(store_file_path):
        # Remove existing csv
        os.remove(store_file_path)


    # Resume the store csv  
    find_resume = True
    if resume:      # Read the last store row
        find_resume = False
        with open(store_file_path, 'r') as file:
            reader = csv.reader(file)
            store_rows = list(reader)
            last_store_row = store_rows[-1]
            print("The number of rows we have processed in the store csv is ", len(store_rows))
            

    # Init the Co-Tracker Model (Offline mode)
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)


    # Read all row in the csv file
    start_time = time.time()
    info_lists = []       # The order will be follow automatically
    with open(csv_file_path) as file_obj:
        # Read the csv
        reader_obj = csv.reader(file_obj) 
        
        # Iterate over each row in the csv  
        for row_idx, row in enumerate(reader_obj): 

            # For the first row case (With all title content)
            if row_idx == 0:    # The first line is the title of content
                elements = dict()
                for element_idx, key in enumerate(row):
                    elements[key] = element_idx

                print("The first row is ", row + ["Obj_Info", "Track_Traj", "Track_Visibility"])

                # Store the csv
                if not resume:
                    with open(store_file_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows([row + ["Obj_Info", "Track_Traj", "Track_Visibility"]])
                continue
            

            # Read important information
            original_idx = row[elements["ID"]]
            video_path = row[elements["video_path"]]
            # num_frames = int(row[elements["num_frames"]])
            fps = float(row[elements["fps"]])
            height = int(row[elements["height"]])
            width = int(row[elements["width"]])
            All_Frame_Panoptic_Segmentation = json.loads(row[elements["Panoptic_Segmentation"]])    
            valid_duration = json.loads(row[elements["valid_duration"]])



            # Resume mode will execute until we have the last store row matched
            if resume:
                if video_path == last_store_row[elements["video_path"]] and original_idx == last_store_row[elements["ID"]]:     # Check ID and video Path (Should all)
                    find_resume = True      # We find the exact row we want
                    print("We find the resume position from row ", row_idx)
            if not find_resume:
                continue


            # Read the video with the original resolution but much denser FPS
            resolution = str(VIDEO_INPUT_RESO[1]) + "x" + str(VIDEO_INPUT_RESO[0])
            video_stream, err = ffmpeg.input(
                                                video_path
                                            ).output(
                                                "pipe:", format = "rawvideo", pix_fmt = "rgb24", s = resolution, vsync = 'passthrough',
                                            ).run(
                                                capture_stdout = True, capture_stderr = True    # If there is bug, command capture_stderr
                                            )    # The resize is already included
            video_full_np = np.frombuffer(video_stream, np.uint8).reshape(-1, VIDEO_INPUT_RESO[0], VIDEO_INPUT_RESO[1], 3)

            # Fetch the valid duration; NOTE: For now, we fetch all frames for tracking
            video_np = video_full_np[valid_duration[0] : valid_duration[1]]            
            video_tensor = torch.tensor(video_np).permute(0, 3, 1, 2)[None].float().to(device)  # B T C H W
            num_frames = len(video_np)      # Update the number of frames


            # Init
            Obj_Info, Track_Traj, Track_Visiblity = [], [], []      # Data Structure is: panoptic -> object -> Num of Points -> x,y
            

            # Iterate All Frames inside the All_Frame_Panoptic_Segmentation
            for panoptic_idx, Per_Frame_Panoptic_Segmentation in enumerate(All_Frame_Panoptic_Segmentation):    # Across different frame idx in the video
                
                # Sometimes this is empty due to historical bug
                if len(Per_Frame_Panoptic_Segmentation) == 0:
                    continue


                # Fetch and Init
                panoptic_frame_idx, sample_point_info = Per_Frame_Panoptic_Segmentation     # NOTE: panoptic_frame_idx is respect to orignal fps, not downsampled fps setting
                Object_of_Interest, Track_Traj_panoptic, Track_Visiblity_panoptic = [], [], []


                # Check if the left frame is enough (Consider the store downsample ratio)
                if num_frames - panoptic_frame_idx <= min_process_frame:   # The remaining frames is not enough
                    print("The remaining frame is not enough")
                    continue
                process_last_idx = min(panoptic_frame_idx + max_process_frame, num_frames - 1)
                

              
                # Iterate the sampled points
                all_queries = []        # TODO: This query is not bounded with the ID, this will be hard to know if this point is human A or other object
                ID_info = []
                cumulative_idx = 0
                for (id_num, text_name, reference_points) in sample_point_info:    # This will collect all object in the current frame
                    queries = []
                    for reference_point in reference_points:

                        # Resize to the target process Resolution
                        cord_y = int(reference_point[0] * VIDEO_INPUT_RESO[0] / height )
                        cord_x = int(reference_point[1] * VIDEO_INPUT_RESO[1] / width )

                        # Append reference point      
                        queries.append([float(panoptic_frame_idx), cord_x, cord_y])   # The order is [frame_idx, width, height]   # First Width then height

                    # Extend all sampled points
                    all_queries.extend(queries)

                    # Update the corresponding ID
                    ID_info.append((id_num, text_name, cumulative_idx, cumulative_idx+len(queries)))     # The range is [start, end)
                    cumulative_idx += len(queries)
          
                # Merge all to torch tensor
                all_queries = torch.tensor(all_queries).to(device)



                ############################################## CoTracker Model Inference 1 for FORWARD Tracking##############################################
                try:    # Execute the cotracker in the uniform resized resolution
                    pred_tracks, pred_visibility = cotracker(video_tensor, queries = all_queries[None], backward_tracking = True)   # B T N 2,  B T N 1
                    pred_tracks = pred_tracks[0].long().detach().cpu().numpy()      # return shape is (temporal, sptial_points, 2)
                    pred_visibility = pred_visibility[0].detach().cpu().numpy()     # return shape is (temporal, sptial_points)
                    assert(len(pred_tracks) == len(pred_visibility))

                except Exception:
                    print("The execution of CoTracker has error, we skip ", video_path)
                    continue
                #############################################################################################################################################
                


                # Fetch the last point and prepare Traking task again
                last_queries = []
                for (cord_x, cord_y) in pred_tracks[process_last_idx]:
                    last_queries.append([float(process_last_idx), float(cord_x), float(cord_y)])        # Mask the last frame
                last_queries = torch.tensor(last_queries).to(device)



                ############################################# CoTracker Model Inference 2 for BACKWARD Tracking ##################################################
                try:    # Execute the cotracker in the uniform resized resolution
                    pred_tracks_backward, pred_visibility_backward = cotracker(video_tensor, queries = last_queries[None], backward_tracking = True) # B T N 2,  B T N 1
                    pred_tracks_backward = pred_tracks_backward[0].long().detach().cpu().numpy()      # return shape is (temporal, sptial_points, 2)
                    pred_visibility_backward = pred_visibility_backward[0].detach().cpu().numpy()     # return shape is (temporal, sptial_points)
                    assert(len(pred_tracks_backward) == len(pred_visibility_backward))
                    assert(len(pred_tracks) == len(pred_tracks_backward))

                except Exception:
                    print("The execution of CoTracker has error, we skip ", video_path)
                    continue
                ###################################################################################################################################################



                # Split pred_tracks back to each object and then store
                for id_number, text_name, cumulative_start, cumulative_end in ID_info:     # All track is split based on the ID

                    store_len = math.ceil(len(pred_tracks) / store_downsample_ratio)
                    Track_Traj_obj = [[] for _ in range(store_len)]  # This list should have same temporal frame number
                    Track_visibility_obj = [[] for _ in range(store_len)]
                    point_idx = cumulative_start        
                    skip_point_num = 0
                    while point_idx < cumulative_end:

                        # Correct the points for those with motion error that is too large (Only test at panoptic frame idx places)
                        # if pred_visibility[process_last_idx][point_idx]:       # Must be a visible points for the Backward Anchor frame points (process_last_idx)
                        cord_x, cord_y = pred_tracks[panoptic_frame_idx][point_idx].tolist()
                        cord_x_backward, cord_y_backward = pred_tracks_backward[panoptic_frame_idx][point_idx].tolist()
                        error_distance = np.sqrt((cord_x - cord_x_backward)**2 + (cord_y - cord_y_backward)**2)
                        if error_distance > VIDEO_INPUT_RESO[0] * motion_error_tolerate:    # With respect to the resized resolution
                            skip_point_num += 1
                            point_idx += 1
                            continue


                        # Store for all points temporally available
                        for temporal_idx in range(len(pred_tracks)):
                            
                            if temporal_idx % store_downsample_ratio != 0:
                                continue        # We store the tracking in downsample ways to avoid adjust the other preprocesing code

                            # Resize back the points
                            cord_x_raw, cord_y_raw = pred_tracks[temporal_idx][point_idx].tolist()
                            traj_point = [int(cord_x_raw * width / VIDEO_INPUT_RESO[1]), int(cord_y_raw * height / VIDEO_INPUT_RESO[0])]
                            
                            # Append to the list for each object
                            store_temporal_idx = temporal_idx // store_downsample_ratio
                            Track_Traj_obj[store_temporal_idx].append(traj_point)
                            Track_visibility_obj[store_temporal_idx].append(pred_visibility[temporal_idx][point_idx].tolist())     # Visibility has size of shape relation, we ignore

                        # Update
                        point_idx += 1


                    # If more than half of the points is eliminated, we should discard this objects / all objects (whole instances)
                    skip_ratio = skip_point_num / (cumulative_end - cumulative_start)
                    print("skip point ratio is ", skip_ratio)
                    if skip_ratio > max_skip_ratio_allowed:
                        print("The tracking has too many fails, we gave up this object!")
                        continue


                    # Append to the larger lists
                    if len(Track_Traj_obj[0]) != 0:     # Check if the first frame is not empty
                        Object_of_Interest.append([text_name, panoptic_frame_idx])      
                        Track_Traj_panoptic.append(Track_Traj_obj)                      # NOTE: update the Track_Traj together
                        Track_Visiblity_panoptic.append(Track_visibility_obj)                    # This is corresponding to the Track_Traj



                # Store the result Panoptic Frame Wise
                if len(Object_of_Interest) != 0:
                    
                    # Append to the full lists
                    Obj_Info.append(Object_of_Interest)
                    Track_Traj.append(Track_Traj_panoptic)
                    Track_Visiblity.append(Track_Visiblity_panoptic)
                    

                    # Visualize (Comment Out Later)
                    # video_stream, err = ffmpeg.input(
                    #                             video_path
                    #                         ).output(
                    #                             "pipe:", format = "rawvideo", pix_fmt = "rgb24", vsync = 'passthrough',
                    #                         ).run(
                    #                             capture_stdout = True, capture_stderr = True
                    #                         )             # As original Resolution and 12 FPS
                    # video_np2 = np.frombuffer(video_stream, np.uint8).reshape(-1, height, width, 3)       
                    # visualize_motion(Track_Traj_panoptic, Object_of_Interest, Track_Visiblity_panoptic, panoptic_frame_idx, process_last_idx, height, width, fps,
                    #                     video_np2, dot_radius=6, name = "Row"+str(row_idx)+"_Panoptic"+str(panoptic_idx)+"_"+str(int(fps))+"FPS_Cycle_NoVisibleFilter")


            # Update to the info lists (One instance finished!)
            if len(Obj_Info) != 0:
                info_lists.append(row + [json.dumps(Obj_Info), json.dumps(Track_Traj), json.dumps(Track_Visiblity)])  
            

            # Log update (The update will be quite random for this file, because we may skip earlier on)
            if row_idx % store_freq == 0:
                
                print("We have processed ", float(row_idx/1000), "K video")
                print("The number of valid videos we found in this iter is ", len(info_lists))
                full_time_spent = int(time.time() - start_time)
                print("Time spent is %d min %d s" %(full_time_spent//60, full_time_spent%60))

                # Store the csv
                with open(store_file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(info_lists)

                # Restart the info_lists
                info_lists = []    


        # Final Log update
        with open(store_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(info_lists)




if __name__ == "__main__": 


    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_offset', type=int, default=0)
    args = parser.parse_args()


    # Fundamental Setting
    csv_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_caption"      # Input
    store_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_TrackMotion"      # Output
    resume = False                  # Whether we read from the middle
    GPU_offset = args.GPU_offset



    # Tracking Setting
    VIDEO_INPUT_RESO = (384, 512)           # Resolution of the input video to be resized to (based on the Co-Tracker Demo)
    min_process_frame = 50                  # Respect to the default fps; 50 is 2*25
    max_process_frame = 170                 # Respect to the default fps; 170 is set for max 81 frames at 12FPS in training
    store_downsample_ratio = 1              # 
    motion_error_tolerate = 0.04            # This is with respect to the height
    max_skip_ratio_allowed = 0.33           # If more than 1/3 of the points is not valid, we should discard this object case


    # Prepare the folder
    if not os.path.exists(store_folder_path):
        os.makedirs(store_folder_path)


    # Process
    single_process(csv_folder_path, store_folder_path, GPU_offset, min_process_frame, max_process_frame)


    print("Finished!")