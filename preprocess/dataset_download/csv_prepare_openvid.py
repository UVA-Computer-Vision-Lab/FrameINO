'''
    Currently, this is for the WebVid10M Process
'''

import os, sys, shutil
import pandas as pd
import csv
import json



if __name__ == "__main__":

    # Important Setting
    video_parent_path = "/PATH/TO/OpenVid"     # Input
    csv_file_path = "/PATH/TO/OpenVid/data/train/OpenVid-1M.csv"      # Input (Should be downloaded with video dataset folder)
    store_csv_folder = "/PATH/TO/CSV_FOLDER/general_dataset_raw"       # Output
    division_num = 1       # Sub CSV number  (We set to 1 here, but usually I set to 32 to process 32 GPU in parallel by GPU_offset in the argument of the following preprocessing codes)


    # Other Setting
    title_conent = ["ID", "video_path", "provided_text"]


    # Prepare the folder
    if os.path.exists(store_csv_folder):
        shutil.rmtree(store_csv_folder)
    os.makedirs(store_csv_folder)



    # Read based on the csv
    video_lists = []
    effective_idx = 0
    with open(csv_file_path) as file_obj: 
        reader_obj = csv.reader(file_obj) 
        for idx, row in enumerate(reader_obj): 
            if idx == 0:
                
                elements = dict()
                for element_idx, key in enumerate(row):
                    elements[key] = element_idx

                continue
            
            # Fetch
            video_name = row[elements["video"]]
            provided_text = row[elements["caption"]]

            # Check video path
            video_path = os.path.join(video_parent_path, video_name)
            if not os.path.exists(video_path):
                continue

            # Append
            video_lists.append([effective_idx, video_path, provided_text])
            effective_idx += 1

    print("The number of valid video length is ", len(video_lists))   

    


    # Divide into division_num of sections
    effective_length = len(video_lists)
    for division_idx in range(division_num):
        sub_video_lists = video_lists[ int(effective_length * division_idx/division_num) : int(effective_length * (division_idx+1)/division_num)]
        sub_video_lists.insert(0, title_conent)
        sub_csv_path = os.path.join(store_csv_folder, "sub"+str(division_idx) + ".csv")

        with open(sub_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(sub_video_lists)


    print("Finished!")