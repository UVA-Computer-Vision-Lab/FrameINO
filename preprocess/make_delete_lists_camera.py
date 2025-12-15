'''
    This file is to get a data distribution of the dataset, like fps
'''

import os, sys, shutil
import time
from multiprocessing import Process
import multiprocessing
import csv
import math
import collections
import json
import numpy as np
import ast
import matplotlib.pyplot as plt
import random
import imageio
import ffmpeg

csv.field_size_limit(sys.maxsize)



def compute_pose_error(R1: np.ndarray, t1: np.ndarray, prev_focal: float, R2: np.ndarray, t2: np.ndarray, cur_focal: float):
    """
    Compute the translation and rotation error between two camera poses.

    Parameters:
    - R1, R2: (3, 3) numpy arrays representing rotation matrices.
    - t1, t2: (3,) numpy arrays representing translation vectors.

    Returns:
    - translation_error: Euclidean distance between translations.
    - rotation_error_deg: Rotation difference in degrees.
    """

    # Compute translation error (Euclidean distance)
    translation_error = np.linalg.norm(t1 - t2)

    # Compute rotation error (Geodesic distance)
    R_diff = R1.T @ R2
    trace_value = np.trace(R_diff)
    
    # Ensure numerical stability
    trace_value = np.clip((trace_value - 1) / 2, -1.0, 1.0)
    
    # Compute rotation error in radians and convert to degrees
    rotation_error_rad = np.arccos(trace_value)
    rotation_error_deg = np.degrees(rotation_error_rad)


    # Check the change of focus
    focal_difference = abs(cur_focal - prev_focal)



    return float(translation_error), float(rotation_error_deg), focal_difference




def calculate_camera_motion(camera_info):

    # Fetch information
    Rotation_matrix_list = camera_info["rotation"]
    Translation_list = camera_info["translation"]
    Focal_list_x = camera_info['focal_x']
    Focal_list_y = camera_info['focal_y']

    # Focal combine
    Focal_list = [math.sqrt(Focal_list_x[idx] * Focal_list_y[idx]) for idx in range(len(Focal_list_x))]


    total_rotation_error, total_translation_error, total_focal_change = 0, 0, 0
    # Iterate all temporally to comapre the motion and angle difference
    for idx, rotation_matrix in enumerate(Rotation_matrix_list):
        
        # Fetch information
        cur_rotation_matrix = np.array(rotation_matrix)
        cur_translation = np.array(Translation_list[idx])
        cur_focal = Focal_list[idx]
        

        # Skip the first case
        if idx == 0:
            prev_rotation_matrix = cur_rotation_matrix
            prev_translation = cur_translation
            prev_focal = cur_focal
            continue
        

        # Calculate the pose error
        translation_error, rotation_error_deg, focal_difference = compute_pose_error(prev_rotation_matrix, prev_translation, prev_focal, cur_rotation_matrix, cur_translation, cur_focal)

        # Update
        total_translation_error += translation_error
        total_rotation_error += rotation_error_deg
        total_focal_change += focal_difference


        # Update Previous info at the end
        prev_rotation_matrix = cur_rotation_matrix
        prev_translation = cur_translation
        prev_focal = cur_focal


    # Calculate the average
    average_rotation_error = total_rotation_error / (len(Rotation_matrix_list) - 1)
    average_translation_error = total_translation_error / (len(Rotation_matrix_list) - 1)
    average_focal_change = total_focal_change / (len(Rotation_matrix_list) - 1)

    

    return average_rotation_error, average_translation_error, average_focal_change




if __name__ == "__main__":
    
    
    # Basic Setting
    csv_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_camera_estimation"               # Input
    store_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_camera_estimation_left"        # Output
    target_column_name = "Camera_Pose"
    store_name_postfix = "Visual"      # For visualization


    # Store Samples Setting (For Visualization)
    collected_range = [0.0, 1.0] 
    sample_num = 50
    store_width = 480
    store_height = 320


    # Delete Range
    delete_ranges = {
                        "rotation": [0.6, 1.0],
                        "translation": [0.6, 1.0],
                        "focal": [0.85, 1.0],
                    }       # 1.0 refers to the strongest level

    # Default Setting:
    #   "rotation": [0.6, 1.0],
    #   "translation": [0.6, 1.0],
    #   "focal": [0.85, 1.0],

    # VidGen Extra Filter:
    #   "rotation": [0.42, 1.0],
    #   "translation": [0.5, 1.0],
    #   "focal": [0.85, 1.0],

    

    # Set the parallel num
    parallel_num = len(os.listdir(csv_folder_path))        # The number of storage is the same as original number of files


    # Prepare the folder
    if os.path.exists(store_folder_path):
        shutil.rmtree(store_folder_path)
    os.makedirs(store_folder_path)


    # Prepare return items
    manager = multiprocessing.Manager()
    all_process_video_paths = manager.dict()



    # Collect the data in a whole
    all_video_info = []
    camera_rotation_error_info, camera_translation_error_info, camera_focal_change_info = [], [], []
    for process_id in range(parallel_num):
        # Read the csv file

        csv_file_path = os.path.join(csv_folder_path, "sub" + str(process_id) + ".csv")
        print("We are processing ", csv_file_path)


        # analysis_names should all be float type
        with open(csv_file_path) as file_obj: 
        
            reader_obj = csv.reader(file_obj) 
            
            # Iterate over each row in the csv  
            for idx, row in enumerate(reader_obj): 

                if idx == 0:    # The first line is the title of content
                    
                    elements = dict()
                    for element_idx, key in enumerate(row):
                        elements[key] = element_idx
                    first_row_info = row
                    continue
                
                # Curate all the videos together.
                video_path = row[elements["video_path"]] 
                valid_duration = row[elements["valid_duration"]]
                max_sec_consider, speedup_factor, camera_info = ast.literal_eval(row[elements[target_column_name]])



                # Calculate the camera motion score
                average_rotation_error, average_translation_error, average_focal_change = calculate_camera_motion(camera_info)


                # Update
                camera_rotation_error_info.append((average_rotation_error, video_path, valid_duration))
                camera_translation_error_info.append((average_translation_error, video_path, valid_duration))
                camera_focal_change_info.append((average_focal_change, video_path, valid_duration))


                # Curate all the videos together
                info = dict()
                for key in elements:
                    element_idx = elements[key]
                    info[key] = row[element_idx]     # HACK: We must use delete lists to split the value 
                all_video_info.append(info)


    total_length = len(camera_rotation_error_info)
    print("Full number of videos we have is ", total_length)



    # Sort all elements
    camera_rotation_error_info.sort(key=lambda x: x[0])
    camera_translation_error_info.sort(key=lambda x: x[0])
    camera_focal_change_info.sort(key=lambda x: x[0])




    #################################################### HACK: COPY some sample videos ##################################################


    # Prepare Camera Rotation
    target_video_infos = [camera_rotation_error_info, camera_translation_error_info, camera_focal_change_info] 
    tmp_store_folders = [
                            "sample_rotation_error_" + store_name_postfix, 
                            "sample_translation_error_" + store_name_postfix, 
                            "sample_focal_change_" + store_name_postfix, 
                        ]

    # Iterate all 
    for idx, target_video_info in enumerate(target_video_infos):

        # Prepare folder
        tmp_store_folder = tmp_store_folders[idx]
        if os.path.exists(tmp_store_folder):
            shutil.rmtree(tmp_store_folder)
        os.makedirs(tmp_store_folder)
        print("Prepareing ", tmp_store_folder)


        # Define the collect range videos
        length = len(target_video_info)
        pool_video_paths = target_video_info[int(collected_range[0] * length) : int(collected_range[1] * length)]
        selected_video_paths = random.sample(pool_video_paths, min(sample_num, len(pool_video_paths)))  # First random sample
        selected_video_paths.sort(key=lambda x: x[0])   # Then, Sort


        # Store the videos
        for idx, video_combo in enumerate(selected_video_paths):

            # Fetch
            score, video_path, valid_duration = video_combo
            valid_duration = json.loads(valid_duration)
            video_extension = video_path.split(".")[-1]
            store_path = os.path.join(tmp_store_folder, str(idx) + "_" + str(score) + "." + video_extension)


            # Read the video and then crop the used duration
            resolution = str(store_width) + "x" + str(store_height)
            video_stream, err = ffmpeg.input(
                                                video_path
                                            ).output(
                                                "pipe:", format = "rawvideo", pix_fmt = "rgb24", s = resolution, vsync = 'passthrough',
                                            ).run(
                                                capture_stdout = True, capture_stderr = True    # If there is bug, command capture_stderr
                                            )    # The resize is already included
            video_full_np = np.frombuffer(video_stream, np.uint8).reshape(-1, store_height, store_width, 3)
            video_np = video_full_np[valid_duration[0] : valid_duration[1]]
            imageio.mimsave(store_path, video_np, fps = 24)



    #####################################################################################################################################




    ######################################## Filter the CSV files and merge to a csv #####################################################

    # Sort 
    delete_sets = set()
    for idx, target_type in enumerate(delete_ranges.keys()):

        # Find for each target type: rotation, translation, focal
        if target_type == "rotation":
            combo = camera_rotation_error_info

        elif target_type == "translation":
            combo = camera_translation_error_info

        elif target_type == "focal":
            combo = camera_focal_change_info

        else:
            raise NotImplementedError("We don't have this type")
        
        # Form the delete set
        start_ratio, end_ratio = delete_ranges[target_type]
        combo_len = len(combo)
        delete_list = combo[int(combo_len * start_ratio) : int(combo_len * end_ratio)]
        print(target_type, "Delete list has len", len(delete_list))
        for (score, delete_video_path, valid_duration) in delete_list:
            delete_sets.add(delete_video_path)
        print("delete_sets has increased to ", len(delete_sets))


    # Merge the left csv to one lists
    left_video_info = []
    for info in all_video_info:
        video_path = info["video_path"]
        if video_path in delete_sets:       # Could not be in the delete set
            continue

        row = []
        for key in first_row_info:
            row.append(info[key])
        left_video_info.append(row)
    print("Total Left video num has ", len(left_video_info))
    print("Final left ratio is ", len(left_video_info) / total_length)


    # Also, prepapre a left list (Equal split)
    effective_length = len(left_video_info)
    for division_idx in range(parallel_num):
        sub_video_lists = left_video_info[ int(effective_length * division_idx/parallel_num) : int(effective_length * (division_idx+1)/parallel_num)]
        sub_video_lists.insert(0, first_row_info)
        sub_csv_path = os.path.join(store_folder_path, "sub"+str(division_idx) + ".csv")

        with open(sub_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(sub_video_lists)

    print("Finished!")

    #######################################################################################################################