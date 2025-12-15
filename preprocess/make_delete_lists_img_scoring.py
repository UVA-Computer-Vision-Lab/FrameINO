'''
    This file is to get a data distribution of the dataset, like fps
'''

import os, sys, shutil
import time
from multiprocessing import Process
import multiprocessing
import csv
import collections
import matplotlib.pyplot as plt
import random
csv.field_size_limit(sys.maxsize)



if __name__ == "__main__":
    
    # Basic Setting
    csv_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_scoring_img"                   # Input
    store_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_scoring_img_left"            # Output


    # New General Setting
    target_types = ["Text_Area", "Image_Quality_Assessment", "Aesthetic", "Image_Complexity", "First_Frame_Clarity"]        # Available Choice: "Text_Area", "Image_Quality_Assessment", "Aesthetic", "Image_Complexity", "First_Frame_Clarity"
    delete_ranges = [
                        [[0.9, 1.0]],                   # Text Area   Needs bigger to [0.85, 1.0] at least; Was used with [0.92, 1.0]
                        [[0.0, 0.1]],                   # Image Quality Assessment
                        [[0.0, 0.05]],                  # Aesthetic
                        [[0.0, 0.03], [0.99, 1.0]],     # Image Complexity
                        [[0.0, 0.075]],                 # First_Frame_Clarity
                    ]    


    # Set the parallel num
    parallel_num = len(os.listdir(csv_folder_path))        # Usually should be the same number as sub csv needed


    # Prepare the folder
    if os.path.exists(store_folder_path):
        shutil.rmtree(store_folder_path)
    os.makedirs(store_folder_path)



    # Collect the data in a whole
    all_video_info = []
    for process_id in range(parallel_num):
        # Read the csv file

        csv_file_path = os.path.join(csv_folder_path, "sub" + str(process_id) + ".csv")

        # analysis_names should all be float type
        with open(csv_file_path) as file_obj: 
        
            reader_obj = csv.reader(file_obj) 
            
            # Iterate over each row in the csv  
            for idx, row in enumerate(reader_obj): 

                if idx == 0:    # The first line is the title of content
                    print("The first row is ", row)

                    elements = dict()
                    for element_idx, key in enumerate(row):
                        elements[key] = element_idx
                    first_row_info = row
                    continue

                # Curate all the videos together.
                info = dict()
                for key in elements:
                    element_idx = elements[key]
                    info[key] = row[element_idx]     # HACK: We must use delete lists to split the value 
                all_video_info.append(info)

    print("Full number of videos we have is ", len(all_video_info))



    # Sort and Collect all not ideal range videos into a set
    delete_sets = set()
    for type_idx, target_type in enumerate(target_types):
        print("Evaluating", target_type, "...")
        # For loop to make a combo
        combo = []
        for video_idx, info in enumerate(all_video_info):
            video_combo_path = info["video_path"] + info["valid_duration"]    # NOTE: Now, we need a combo of video path and the duration
            try:
                target = float(info[target_type])
            except Exception:
                print("Exception for ", info[target_type], " for video ",  video_idx)
                continue
            combo.append([video_combo_path, target])
        combo.sort(key = lambda x: float(x[1]))
        combo_len = len(combo)
        print("The worst case from [0,1] range is ", combo[0])
        print("The best case from [0,1] range is ", combo[-1])


        # Collect delete datasets set
        for delete_range in delete_ranges[type_idx]:
            start_ratio, end_ratio = delete_range
            delete_list = combo[int(start_ratio*combo_len): int(end_ratio*combo_len)]
            # print("Delete list has len", len(delete_list))
            for (delete_video_path, score) in delete_list:
                delete_sets.add(delete_video_path)
            print("Delete Set has been increased to ", len(delete_sets))

    print("Total Delete set has length ", len(delete_sets))



    # Find the not deleted videos
    left_video_info = []
    for info in all_video_info:
        
        # Combine a combo
        video_combo_path = info["video_path"] + info["valid_duration"]

        if video_combo_path in delete_sets:
            continue

        row = []
        for key in first_row_info:
            row.append(info[key])
        left_video_info.append(row)
    print("Left video num has ", len(left_video_info))



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