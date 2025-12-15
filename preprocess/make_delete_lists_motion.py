'''
    This file is to estimate the motion to all objects and filter in object level.
'''

import os, sys, shutil
import time
from multiprocessing import Process
import multiprocessing
import csv
import json
import collections
import matplotlib.pyplot as plt
import random
import numpy as np
import cv2
import ffmpeg
from torchvision import transforms
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
                                dot_radius, target_width, target_height, name = ""):

    # Prepare the color
    target_color_codes = all_color_codes[:len(full_pred_tracks[0])]        # This means how many objects in total we have
    invisible_color_code = (0, 0, 0)

    # Prepare the traj image
    traj_img_lists = []

    # Set a new dot radius based on the resolution fluctuating
    dot_radius_resize = int( dot_radius * original_height / 384 )     # This is set with respect to default 384 height, will be adjust based on the height change


    # Iterate all object instance
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
        cv2.imwrite("Video"+name + "_traj" + str(temporal_idx).zfill(2) + ".png", cv2.cvtColor(merge_frame, cv2.COLOR_RGB2BGR)) # The base_img is designed in RGB form


        # Append to the temporal index
        traj_img_lists.append(base_img)
    
        
    # Write to video (For Debug Purpose)
    os.system("ffmpeg -r 12 -f image2 -i Video"+name+"_traj%02d.png -vcodec libx264 -crf 30 -pix_fmt yuv420p gen_video" + name + ".mp4")

    # Clean frames
    os.system("rm Video" + name + "_traj*.png")



def visualize_motion(Track_Traj, Obj_Info, Track_Visibility, height, width, video_np, dot_radius, name):

    
    # Convert Track_Traj to the data form of (temporal, object, points, xy)
    train_frame_num = 49
    iter_gap = 1
    full_pred_tracks = [[] for _ in range(train_frame_num)]   # The dim should be: (temporal, object, points, xy) The fps should be fixed to 12 fps, which is the same as training decode fps
    full_visibility_tracks = [[] for _ in range(train_frame_num)]

    # Iterate all objects in current panoptic frame
    for obj_idx in range(len(Obj_Info)):

        # Read the basic info
        text_name, original_start_frame_idx, fps_scale = Obj_Info[obj_idx]      # This is expected to be all the same in the video
        
        # Sanity Check: make sure that the number of frames is consistent
        if obj_idx > 0:
            if original_start_frame_idx != previous_original_frame_idx:
                raise Exception("The panoptic_frame_idx cannot pass the sanity check")
        downsample_panoptic_frame_idx = int(fps_scale * original_start_frame_idx)

        # Prepare the tracjectory
        pred_tracks = Track_Traj[obj_idx]
        visibility_tracks = Track_Visibility[obj_idx]

        # Fetch the ideal range needed
        pred_tracks = pred_tracks[downsample_panoptic_frame_idx : downsample_panoptic_frame_idx + iter_gap * train_frame_num : iter_gap]   
        visibility_tracks = visibility_tracks[downsample_panoptic_frame_idx : downsample_panoptic_frame_idx + iter_gap * train_frame_num : iter_gap]
        if len(pred_tracks) != train_frame_num:
            raise Exception("The len of pre_track does not match")
        for temporal_idx, pred_track in enumerate(pred_tracks):
            full_pred_tracks[temporal_idx].append(pred_track)    # pred_tracks will be 49 frames, and each one represent all tracking points for single objects; only one object here
            full_visibility_tracks[temporal_idx].append(visibility_tracks[temporal_idx])

        # Other update
        previous_original_frame_idx = original_start_frame_idx

    # Choose the selected frame range
    selected_frames = video_np[downsample_panoptic_frame_idx : downsample_panoptic_frame_idx + iter_gap * train_frame_num : iter_gap]

    # Call the Drawing function
    prepare_traj_tensor(full_pred_tracks, full_visibility_tracks, height, width, selected_frames, dot_radius, width, height, name = name)



def curate_all_object_speed(csv_folder_path, process_id):
    # This function will curate the speed of all objects in all panoptic frames


    # Read the csv file
    csv_file_path = os.path.join(csv_folder_path, "sub" + str(process_id) + ".csv")


    # Init
    obj_motion_strength = []


    # Iterate the CSV files
    start_time = time.time()
    with open(csv_file_path) as file_obj: 
    
        reader_obj = csv.reader(file_obj) 
        
        # Iterate over each row in the csv  
        for idx, row in enumerate(reader_obj): 
            if idx == 0:    # The first line is the title of content
                print("The first row is ", row)
                elements = dict()
                for element_idx, key in enumerate(row):
                    elements[key] = element_idx
                continue
            

            # Read important information
            video_path = row[elements["video_path"]]
            num_frames = int(row[elements["num_frames"]])
            fps = float(row[elements["fps"]])
            height = int(row[elements["height"]])
            width = int(row[elements["width"]])
            Obj_Info_raw = json.loads(row[elements["Obj_Info"]])            # Data Struct is: panoptic -> object -> 2 [id_type, frame_idx]
            Track_Traj_raw = json.loads(row[elements["Track_Traj"]])        # Data Structure is: panoptic -> object -> Num of Points -> x,y
            


            # Iterate all Panoptic Frames
            for panoptic_list_idx, per_panoptic_Track_Traj in enumerate(Track_Traj_raw):

                
                if len(per_panoptic_Track_Traj) == 0:     # Sometimes there is no info inside
                    print("No per_panoptic_Track_Traj Found!")
                    continue
                

                # Iterate all objects
                for obj_list_idx, per_object_Track_Traj in enumerate(per_panoptic_Track_Traj):


                    # Calculate the average motion distances between the FIRST and LAST (avoid flickering)
                    moving_distances = []
                    start_process_idx = Obj_Info_raw[panoptic_list_idx][obj_list_idx][1]    
                    last_process_idx = min(start_process_idx + process_frame_num, len(per_object_Track_Traj) - 1)
                    process_len = last_process_idx - start_process_idx

                    # Iterate all points
                    for point_idx in range(len(per_object_Track_Traj[0])):          

                        # First point
                        first_x, first_y = per_object_Track_Traj[start_process_idx][point_idx]

                        # Last point
                        last_x, last_y = per_object_Track_Traj[last_process_idx][point_idx]

                        # Calculate the Euclidian Distance
                        moving_distance = np.sqrt( (last_x - first_x) ** 2 + (last_y - first_y) ** 2 ) / process_len       # Scale with respect to the number of frames consider
                        moving_distance = 100000 * moving_distance / (height * width)    # Scale with respect to the resolution area of the video
                        moving_distances.append(float(moving_distance))

                    # Taking the average per object (across all points)
                    average_movement_per_obj = sum(moving_distances) / len(moving_distances)
                    obj_motion_strength.append([average_movement_per_obj, video_path, start_process_idx, last_process_idx])


            # Update log
            if idx % 1000 == 0:
                
                print("We have processed ", idx//1000, "K video")
                full_time_spent = int(time.time() - start_time)
                print("Time spent is %d min %d s" %(full_time_spent//60, full_time_spent%60))

    return obj_motion_strength



def discard_video_in_range(csv_folder_path, store_folder_path, process_id, lower_threshold, higher_threshold):

    # Discard video that is too slow

    # Read the csv file
    csv_file_path = os.path.join(csv_folder_path, "sub" + str(process_id) + ".csv")

    # Prepare the store file path
    store_file_path = os.path.join(store_folder_path, "sub" + str(process_id) + ".csv")
    if os.path.exists(store_file_path):
        # Remove existing csv
        os.remove(store_file_path)


    # Init
    info_lists = []
    effective_obj_tracking = 0
    tracking_point_left_less_than_expected = 0
    store_freq = 1000
    

    # Iterate the CSV files
    start_time = time.time()
    with open(csv_file_path) as file_obj: 
    
        reader_obj = csv.reader(file_obj) 
        
        # Iterate over each row in the csv  
        for row_idx, row in enumerate(reader_obj): 

            if row_idx == 0:    # The first line is the title of content
                print("The first row is ", row)
                elements = dict()
                for element_idx, key in enumerate(row):
                    elements[key] = element_idx
                
                # Store the csv
                with open(store_file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows([row])
                continue
            

            # Read important information
            video_path = row[elements["video_path"]]
            num_frames = int(row[elements["num_frames"]])
            fps = float(row[elements["fps"]])
            height = int(row[elements["height"]])
            width = int(row[elements["width"]])
            Obj_Info_raw = json.loads(row[elements["Obj_Info"]])
            Track_Traj_raw = json.loads(row[elements["Track_Traj"]])
            Track_Visibility_raw = json.loads(row[elements["Track_Visibility"]])
            if "Text_Prompt" in elements:
                modify_text = True
                text_prompt_all = json.loads(row[elements["Text_Prompt"]])


            # Init; This is because we will switch new ones
            Obj_Info_new, Track_Traj_new, Track_Visibility_new, Text_Prompt_new = [], [], [], []


            # Read the Video Frame for the Visual Purposes
            # video_stream, err = ffmpeg.input(
            #                                     video_path
            #                                 ).filter(
            #                                     'fps', fps = decode_fps_training, round = 'up'
            #                                 ).output(
            #                                     "pipe:", format = "rawvideo", pix_fmt = "rgb24"
            #                                 ).run(
            #                                     capture_stdout = True, capture_stderr = True
            #                                 )      # The resize is already included
            # video_np = np.frombuffer(video_stream, np.uint8).reshape(-1, height, width, 3)


            # Iterate all Panoptic Frames
            for panoptic_list_idx, panoptic_Track_Traj in enumerate(Track_Traj_raw):

                if len(panoptic_Track_Traj) == 0:     # Sometimes there is no info inside
                    print("No panoptic_Track_Traj Found!")
                    continue


                # Fetch
                panoptic_Track_Visibility = Track_Visibility_raw[panoptic_list_idx]

                # Init the new sub info to new Tracking info
                Obj_Info_objects, Track_Traj_objects, Track_Visibility_objects = [], [], []


                # Draw the Tracking
                # visualize_motion(panoptic_Track_Traj, panoptic_Obj_Info, panoptic_Track_Visibility, height, width, video_np, 7, name = "Row"+str(row_idx)+"_Panoptic"+str(panoptic_idx)+"_Begin")


                # Iterate all objects
                num_point_before = 0        # Calculate change of number of tracking points
                num_point_after = 0
                max_move_speed = 0 
                for obj_list_idx, per_object_Track_Traj in enumerate(panoptic_Track_Traj):

                    # Calculate the average motion distances from 
                    moving_distances = []
                    start_process_idx = Obj_Info_raw[panoptic_list_idx][obj_list_idx][1]    
                    last_process_idx = min(start_process_idx + process_frame_num, len(per_object_Track_Traj) - 1)
                    process_len = last_process_idx - start_process_idx

                    # Update the number of tracking points available
                    num_point_before += len(per_object_Track_Traj[0])

                    # Itearte all points
                    for point_idx in range(len(per_object_Track_Traj[0])):

                        # First point
                        first_x, first_y = per_object_Track_Traj[start_process_idx][point_idx]

                        # Last point
                        last_x, last_y = per_object_Track_Traj[last_process_idx][point_idx]

                        # Calculate the Euclidian Distance      # Same as function above
                        moving_distance = np.sqrt( (last_x - first_x) ** 2 + (last_y - first_y) ** 2 ) / process_len       # Scale with respect to the number of frames consider
                        moving_distance = 100000 * moving_distance / (height * width)    # Scale with respect to the resolution area of the video
                        moving_distances.append(float(moving_distance))


                    # Taking the average per object
                    average_movement_per_obj = sum(moving_distances) / len(moving_distances)
                    max_move_speed = max(average_movement_per_obj, max_move_speed)

                    # Check whether we reach the threshold
                    if average_movement_per_obj >= lower_threshold and average_movement_per_obj <= higher_threshold:

                        # Valid Cases and store Frame Info + Traj Info + Visibility
                        Obj_Info_objects.append(Obj_Info_raw[panoptic_list_idx][obj_list_idx])
                        Track_Traj_objects.append(Track_Traj_raw[panoptic_list_idx][obj_list_idx])
                        Track_Visibility_objects.append(Track_Visibility_raw[panoptic_list_idx][obj_list_idx])
                        
                        # uUpdate the nubmer of tracking points that will be stored
                        num_point_after += len(per_object_Track_Traj[0])


                # Append to the full lists
                if len(Obj_Info_objects) != 0:


                    # Judge wheter too many tracking points are lost
                    tracking_point_left_ratio = num_point_after / num_point_before
                    if tracking_point_left_ratio < min_tracking_point_left_ratio:
                        # print("Too many tracking points disappear!")
                        tracking_point_left_less_than_expected += 1
                        continue

                    # Update
                    effective_obj_tracking += 1
                    Obj_Info_new.append(Obj_Info_objects)
                    Track_Traj_new.append(Track_Traj_objects)
                    Track_Visibility_new.append(Track_Visibility_objects)
                    if modify_text:
                        Text_Prompt_new.append(text_prompt_all[panoptic_list_idx])        # NOTE: text prompt is the same per panoptic frame


                    # Debug Check
                    # if max_move_speed > 700:
                    #     video_stream, err = ffmpeg.input(
                    #                             video_path
                    #                         ).filter(
                    #                             'fps', fps = decode_fps_training, round = 'up'
                    #                         ).output(
                    #                             "pipe:", format = "rawvideo", pix_fmt = "rgb24"
                    #                         ).run(
                    #                             capture_stdout = True, capture_stderr = True
                    #                         )      # The resize is already included
                    #     video_np = np.frombuffer(video_stream, np.uint8).reshape(-1, height, width, 3)
                    #     visualize_motion(Track_Traj_objects, Obj_Info_objects, Track_Visibility_objects, height, width, video_np, 7, name = "Row"+str(row_idx)+"_Panoptic"+str(panoptic_idx)+"_End" + str(max_move_speed))



            # Replace the Original ones
            if len(Obj_Info_new) != 0:

                # Rewrite Obj_Info and Track_Traj
                row[elements["Obj_Info"]] = json.dumps(Obj_Info_new)
                row[elements["Track_Traj"]] = json.dumps(Track_Traj_new)
                row[elements["Track_Visibility"]] = json.dumps(Track_Visibility_new)
                if modify_text:
                    row[elements["Text_Prompt"]] = json.dumps(Text_Prompt_new)
                
                # Append to info_lists
                info_lists.append(row)
            

            # Log update (The update will be quite random for this file, because we may skip earlier on)
            if row_idx % store_freq == 0:
                
                print("We have processed ", float(row_idx/1000), "K video")
                print("The number of valid videos we found in this iter is ", len(info_lists), " out of ", store_freq)
                full_time_spent = int(time.time() - start_time)
                print("Time spent is %d min %d s" %(full_time_spent//60, full_time_spent%60))

                # Store the csv
                with open(store_file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(info_lists)

                # Restart the info_lists
                info_lists = []    


        # Write to the store path
        with open(store_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(info_lists)

    print("The number of effective element is ", effective_obj_tracking)
    print("The number of panoptic frame that is filterred because of the tracking point left ratio policy is ", tracking_point_left_less_than_expected)




if __name__ == "__main__":
    

    # Fundamental Setting
    csv_folder_paths = [
                            "/PATH/TO/CSV_FOLDER/general_dataset_TrackMotion",
                        ]   # Inputs of all datasets (motion tracking is balanced across the whole dataset)
    store_folder_paths = [
                            "/PATH/TO/CSV_FOLDER/general_dataset_TrackMotion_left",
                        ]   # Outputs
    process_frame_num = 170                     # 170 is about 81 Frames*2 in training 
    motion_select_range = [0.5, 1.0]            # Kept Range
    min_tracking_point_left_ratio = 0.5         # At least this amount of tracking points in each panoptic segmentation should have



    ##################################### Find the Top and Bottom Threshold of the Per Object Motion ########################################

    # Init
    obj_motion_strength_all = []        # This is per object level

    # Iterate all datasets csv
    for csv_folder_path in csv_folder_paths:

        # Get the name
        essential_name = csv_folder_path.split('/')[-1]

        # Read all videos and then curate the average motion (first + last / 2)
        obj_motion_strength_per_dataset = []
        for process_id in range(len(os.listdir(csv_folder_path))):

            # Curate the motion strength from each sub csv
            obj_motion_strength_list = curate_all_object_speed(csv_folder_path, process_id)
            obj_motion_strength_per_dataset.extend(obj_motion_strength_list)
            print("We have finished csv file ", str(process_id))

            # Fast Check (Comment Out Later)
            # break

            # Visualize per instance
            # plt.clf()
            # plt.hist(obj_motion_strength_list, color='lightgreen', ec='black', bins=1000)
            # plt.savefig("csv"+str(process_id)+".png")
        
        # Sort from the smallest motion to the strongest motion
        obj_motion_strength_per_dataset.sort(key = lambda x: x[0])      # Only the first key is about the motion strength, the rest are video information
        pure_value_lists = [value[0] for value in obj_motion_strength_per_dataset]
        plt.clf()
        plt.hist(pure_value_lists, bins=1000)
        plt.savefig("motion_distance_" + essential_name + ".png")

        # Extend to the main list
        obj_motion_strength_all.extend(obj_motion_strength_per_dataset)


    # Sort All from the smallest motion to the strongest motion
    obj_motion_strength_all.sort(key = lambda x: x[0])      # Only the first key is about the motion strength, the rest are video information
    pure_value_lists = [value[0] for value in obj_motion_strength_all]
    plt.clf()
    plt.hist(pure_value_lists, bins=1000)
    plt.savefig("motion_distance_all.png")


    # Calculate a threshold that is used to filter
    length = len(obj_motion_strength_all)
    start_range, end_range = int(length * motion_select_range[0]),  min(int(length * motion_select_range[1]), length - 1)
    lower_threshold = obj_motion_strength_all[start_range][0]
    higher_threshold = obj_motion_strength_all[end_range][0]



    ########################################################################################################################################



    # Fast Debug Setting (Comment Out Later)
    # lower_threshold, higher_threshold = 0.012, 0.725         # 50% - 100%

    # Process all folder one by one
    print("Lower and Higher motion threshold is ", lower_threshold, higher_threshold)
    for idx, store_folder_path in enumerate(store_folder_paths):

        # Prepare the folder
        if os.path.exists(store_folder_path):
            shutil.rmtree(store_folder_path)
        os.makedirs(store_folder_path)

        # Fetch
        csv_folder_path = csv_folder_paths[idx]

        # Discard all those object that is out of the range (Store inside)
        for process_id in range(len(os.listdir(csv_folder_path))):
            discard_video_in_range(csv_folder_path, store_folder_path, process_id, lower_threshold, higher_threshold)

