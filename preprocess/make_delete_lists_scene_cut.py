'''
    This file is to filter scene cut based on the duration requirement for each clip
'''

import os, sys, shutil
import time
from multiprocessing import Process
import multiprocessing
import csv
import collections
import matplotlib.pyplot as plt
import random
import json
import ast
csv.field_size_limit(sys.maxsize)



if __name__ == "__main__":
    

    # Basic Setting
    csv_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_scoring_SceneCut"            # Input
    store_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_scoring_SceneCut_left"     # Output
    target_name = "SceneCut_AutoShot"
    shuffle = True              # This will shuffle the datset just in case patternlized distribution like 
    minimum_frame_duration = 100        # NOTE: This should be consistent with setting in filter_basic, usually 100: 49 * 2
    


    # Prepare the folder
    if os.path.exists(store_folder_path):
        shutil.rmtree(store_folder_path)
    os.makedirs(store_folder_path)


    # Prepare return items
    manager = multiprocessing.Manager()
    all_process_video_paths = manager.dict()



    # Collect the data in a whole
    left_info = []
    total_video_num = 0
    changed_duration_num = 0        # Record how many videos have been changed
    parallel_num = len(os.listdir(csv_folder_path))
    for process_id in range(parallel_num):
        
        # Read the csv file
        csv_file_path = os.path.join(csv_folder_path, "sub" + str(process_id) + ".csv")
        print("Processing ", csv_file_path)

        # analysis_names should all be float type
        with open(csv_file_path) as file_obj: 
            # Read csv
            reader_obj = csv.reader(file_obj) 
            
            # Iterate over each row in the csv  
            for idx, row in enumerate(reader_obj): 
                
                # The first line is the title of content
                if idx == 0:    
                    print("The first row is ", row)
                    elements = dict()
                    for element_idx, key in enumerate(row):
                        elements[key] = element_idx
                    first_row_info = row
                    continue
                total_video_num += 1


                # Fetch data
                scene_cut_info = json.loads(row[elements[target_name]])
                valid_duration = json.loads(row[elements["valid_duration"]])

                
                # Replace the valid_duration with the first valid Scene Cut range
                scene_num = len(scene_cut_info)
                if scene_num != 1:      # If it just has one scene, use the original duration setting

                    # Update the new duration end position
                    find_ideal = False
                    for (start_idx, end_idx) in scene_cut_info:

                        # Check if the video might be too short for one chunk duration detected
                        if end_idx - start_idx <= minimum_frame_duration:   
                            continue

                        # Find one ideal one, we can break;
                        find_ideal = True
                        valid_duration = [start_idx, end_idx]       # Left range is closed start, right is open end [start, end)
                        break
                    

                    # If there is no ideal chunk of duration is ideal, we skip this instance
                    if not find_ideal:
                        continue

                    # Rewrite the new valid duration information
                    print("New duration is ", valid_duration)
                    row[elements["valid_duration"]] = json.dumps(valid_duration)
                    changed_duration_num += 1
                
                # Append to the row
                left_info.append(row)

    print("Total Changed video num is ", changed_duration_num)
    print("Left number video we have is ", len(left_info))
    print("Total Video Num Theoretically we have is ", total_video_num)


    
    # Shuffle the dataset if needed
    if shuffle:
        random.shuffle(left_info)


    # Write to the list
    effective_length = len(left_info)
    for division_idx in range(parallel_num):
        sub_video_lists = left_info[ int(effective_length * division_idx/parallel_num) : int(effective_length * (division_idx+1)/parallel_num)]
        sub_video_lists.insert(0, first_row_info)
        sub_csv_path = os.path.join(store_folder_path, "sub"+str(division_idx) + ".csv")

        with open(sub_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(sub_video_lists)

