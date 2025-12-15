import os, sys, shutil
import pandas as pd
import time
import csv
import ffmpeg
import imageio
import json
import copy
import numpy as np
from multiprocessing import Process
import multiprocessing
import cv2
import argparse
import subprocess




def single_process( csv_folder_path,
                    store_folder_path,
                    process_idx, 
                    min_num_frames_needed, 
                    max_num_frames_needed,
                    valid_aspect_ratio,
                    min_width_threshold,
                    ):

    # Setting
    store_freq = 50


    # Read the csv file
    csv_file_path = os.path.join(csv_folder_path, "sub" + str(process_idx) + ".csv")
    store_file_path = os.path.join(store_folder_path, "sub" + str(process_idx) + ".csv")
    print("We are processing ", csv_file_path)


    # Prepare the folder
    if os.path.exists(store_file_path):
        os.remove(store_file_path)


    # Read all row in the csv file
    info_lists = []
    start_time = time.time()
    invalid_num_frames_too_small, invalid_num_frames_too_many, invalid_aspect_ratio, invalid_resolution, invalid_fps, invalid_duration = [], [], [], [], [], []
    with open(csv_file_path) as file_obj: 
    
        reader_obj = csv.reader(file_obj) 
        
        # Iterate over each row in the csv  
        for idx, row in enumerate(reader_obj): 

            if idx == 0:    # The first line is the title of content

                elements = dict()
                for element_idx, key in enumerate(row):
                    elements[key] = element_idx

                # Append the first row with all index information
                row.extend(["height", "width", "num_frames", "fps", "total_seconds", "valid_duration"])   # Add all new elements   "num_frames" is already incuded
                info_lists.append(row)
                print("The first row is ", row)
                continue


            elif idx % store_freq == 0:    # Store and update the log
                
                # Log
                print("We have processed ", idx / 1000, "K videos")
                print("The number of valid videos we found in this iter is ", len(info_lists))
                full_time_spent = int(time.time() - start_time)
                print("Time spent is %d min %d s" %(full_time_spent//60, full_time_spent%60))

                # Store the csv
                with open(store_file_path, 'a', newline='', encoding="utf-8") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(info_lists)

                # Restart the info_lists
                info_lists = []    


            # Try to read basic information and see if it is still valid
            try:

                # Fetch the needed and important information
                video_path = row[elements["video_path"]]        

                
                # Read all information by professional ffprobe
                cap = cv2.VideoCapture(video_path)

                # Get basic setting
                height, width, fps = (
                                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                                        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                        float(cap.get(cv2.CAP_PROP_FPS)),
                                    )
                

                # Read the number of frames by ffmpeg
                resolution = str(width) + "x" + str(height)
                video_stream, err = ffmpeg.input(
                                                    video_path
                                                ).output(
                                                    "pipe:", format = "rawvideo", pix_fmt = "rgb24", s = resolution, vsync = 'passthrough',
                                                ).run(
                                                    capture_stdout = True, capture_stderr = True
                                                )      # The resize is already included
                video_np = np.frombuffer(video_stream, np.uint8).reshape(-1, height, width, 3)
                num_frames = len(video_np)
            

                # Read the Duration
                if fps != 0:
                    total_seconds = num_frames / fps
                else:
                    total_seconds = 0
                valid_duration = [0, num_frames]         # Duration for the range of frame idx we will read, very fixed
                

                # Set threshold for the invalid fps
                aspect_ratio = width / height
                if aspect_ratio < valid_aspect_ratio:
                    print("The aspect ratio is not ideal:", aspect_ratio)
                    invalid_aspect_ratio.append([video_path, "Invalid Aspect Ratio at " + str(aspect_ratio)])
                    continue


                # Filter for those whose resolution is too small
                if width < min_width_threshold or height < 0.7 * min_width_threshold:
                    print("The width is too small:", width)
                    invalid_resolution.append([video_path, "Invalid Resolution with width and height " + str(width) + ", " + str(height)])
                    continue
                
                # Check the FPS
                if fps < valid_fps_range[0] or fps > valid_fps_range[1]:
                    print("The fps is not ideal: ", fps)
                    invalid_fps.append([video_path, "Invalid FPS at " + str(fps)])
                    continue

                # Check threshold for frame num
                if num_frames <= min_num_frames_needed:
                    print("The number of frames is too small:", num_frames, " from video ", video_path)
                    invalid_num_frames_too_small.append([video_path, "Invalid Number of frame of " + str(num_frames)])
                    continue
                

                # For more frames available, we can choose to crop the video
                if num_frames >= max_num_frames_needed:
                    print("The number of frames is too many:", num_frames)

                    # Crop the video
                    if crop_long_frames:
                        
                        # Rewrite the valid duratio range
                        crop_section_num = min(num_frames // max_num_frames_needed, max_crop_iter_num)
                        
                        for crop_idx in range(crop_section_num):
                            # Find the valid duration
                            valid_duration = [crop_idx * max_num_frames_needed, (crop_idx + 1) * max_num_frames_needed]

                            # Extend to a copy of the existing information
                            existing_row = copy.deepcopy(row)
                            existing_row.extend([height, width, num_frames, fps, total_seconds, json.dumps(valid_duration)])
                            info_lists.append(existing_row)

                        # Everything Ends after this last one process
                        print("Finished Instance ", idx)
                        continue

                    else:
                        invalid_num_frames_too_many.append([video_path, "Invalid Number of frame of " + str(num_frames)])
                        continue


                # Record the valid one
                row.extend([height, width, num_frames, fps, total_seconds, json.dumps(valid_duration)])
                info_lists.append(row)
                print("Finished Instance ", idx)

            except Exception as Error:
                print("error as ", Error)
                continue


    print("invalid_num_frames_too_small, invalid_num_frames_too_many, invalid_aspect_ratio, invalid_resolution, invalid_fps, and invalid_duration is ", 
          len(invalid_num_frames_too_small), len(invalid_num_frames_too_many), len(invalid_aspect_ratio), len(invalid_resolution), len(invalid_fps), len(invalid_duration))
    # print("Valid video num is ", len(info_lists))



    # Store the csv for the remaining information
    with open(store_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(info_lists)



if __name__ == "__main__":


    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--Process_idx', type=int, default=0)
    args = parser.parse_args()
    

    # Fundamental Setting
    csv_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_raw"             # Input
    store_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_filter_basic"          # Ouput
    process_idx = args.Process_idx


    # Filter Setting
    min_num_frames_needed = 100     # ~ 49 * 2
    max_num_frames_needed = 500     # If there are too many frames, it is not very ideal then
    max_crop_iter_num = 1           # Cannot foreover crop long videos, which repeat one video too much
    valid_fps_range = [20, 31]      # Exactly for 24/30 FPS
    valid_aspect_ratio = 1.25       # Min valid aspect ratio Here we want to filter 1:1 case 
    min_width_threshold = 400       # The height is 0.7 * min_width_threshold
    crop_long_frames = True         # Whether we crop video that is too long to speed up


    # Prepare the csv file
    if not os.path.exists(store_folder_path):
        os.makedirs(store_folder_path, exist_ok = True)


    # Single Process
    single_process(csv_folder_path, store_folder_path, process_idx, min_num_frames_needed, max_num_frames_needed, valid_aspect_ratio, min_width_threshold)
