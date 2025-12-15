
'''
    This file is trying to use SpatialTrackerV2 to fetch the camera pose
'''


import os, sys, shutil
import numpy as np
import torch
import time
import glob
import random
import cv2
import argparse
import json
import ffmpeg
import csv
from copy import deepcopy
import PIL.Image
from PIL import Image
from PIL.ImageOps import exif_transpose
import torchvision.transforms as tvf
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# Import model and inference functions after adding the ckpt path.
root_path = os.path.abspath('.')
sys.path.append(root_path)
from preprocess.SpaTrackV2_code.models.vggt4track.models.vggt_moe import VGGT4Track
from preprocess.SpaTrackV2_code.models.vggt4track.utils.load_fn import preprocess_image



def camera_pose_estimation(video_np, vggt4track_model):

    # Convert
    video_tensor = torch.from_numpy(video_np).permute(0, 3, 1, 2)
    video_tensor = preprocess_image(video_tensor)[None]

    # Execute
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # Predict attributes including cameras, depth maps, and point maps.
            predictions = vggt4track_model(video_tensor.cuda()/255)
            extrinsic, intrinsic = predictions["poses_pred"], predictions["intrs"]

            # Transform
            extrs = extrinsic.squeeze().cpu().numpy()
            intrs = intrinsic.squeeze().cpu().numpy()

    # Fetch Rotation, Translation, and Focal (X,Y axis) information
    rotation = extrs[:, :3, :3]
    translation = extrs[:, :3, 3]
    fx = intrs[:, 0, 0]
    fy = intrs[:, 1, 1]


    # Return information needed
    cam_dict = {
                    "rotation": rotation.tolist(),      
                    "translation":translation.tolist(),     
                    "focal_x": fx.tolist(),
                    "focal_y": fy.tolist(),
                }


    return cam_dict



@torch.no_grad()
def single_process(csv_folder_path, store_folder_path, GPU_offset, speedup_factor, max_sec_consider):
    
    # Setting 
    store_freq = 10
    device = "cuda"


    # Read the csv file
    csv_idx = GPU_offset
    csv_file_path = os.path.join(csv_folder_path, "sub" + str(csv_idx) + ".csv")
    print("CSV file we read is ", csv_file_path)


    # Prepare the store file path
    store_file_path = os.path.join(store_folder_path, "sub" + str(csv_idx) + ".csv")
    if os.path.exists(store_file_path):
        # Remove existing csv
        os.remove(store_file_path)


    # Init the Spatial Tracker V2 model
    vggt4track_model = VGGT4Track.from_pretrained(VGGTTrack_path)
    vggt4track_model.eval()
    vggt4track_model = vggt4track_model.to("cuda")



    # Read all row in the csv file
    start_time = time.time()
    info_lists = []       # The order will be follow automatically
    with open(csv_file_path) as file_obj:

        reader_obj = csv.reader(file_obj) 
        
        # Iterate over each row in the csv  
        for row_idx, row in enumerate(reader_obj): 

            # For the first row case (With all title content)
            if row_idx == 0:    # The first line is the title of content
                elements = dict()
                for element_idx, key in enumerate(row):
                    elements[key] = element_idx

                print("The first row is ", row + ["Camera_Pose"])

                # Store the csv
                with open(store_file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows([row + ["Camera_Pose"]])
                continue
            

            try:

                # Read important information
                video_path = row[elements["video_path"]]
                fps = float(row[elements["fps"]])
                height = int(row[elements["height"]])
                width = int(row[elements["width"]])
                valid_duration = json.loads(row[elements["valid_duration"]])


                # Read the video as the Original Resolution (Resize convert will do later)
                resolution = str(width) + "x" + str(height)
                video_stream, err = ffmpeg.input(
                                                    video_path
                                                ).output(
                                                    "pipe:", format = "rawvideo", pix_fmt = "rgb24", s = resolution, vsync = 'passthrough',
                                                ).run(
                                                    capture_stdout = True, capture_stderr = True    # If there is bug, command capture_stderr
                                                )    # The resize is already included
                video_full_np = np.frombuffer(video_stream, np.uint8).reshape(-1, height, width, 3)

                # Fetch the valid duration
                video_np = video_full_np[valid_duration[0] : valid_duration[1]]
                num_frames = len(video_np)


                # Crop the frames to the maximum of frames needed
                if max_sec_consider != -1:
                    video_np = video_np[:int(max_sec_consider * fps)]


                # Speed Up the sample
                video_np = video_np[::speedup_factor]


                # Camera Pose Estimation
                cam_dict = camera_pose_estimation(video_np, vggt4track_model)     
                full_info = [max_sec_consider, speedup_factor, cam_dict]


                # Update the text prompt
                info_lists.append(row + [full_info])

            except Exception as error:
                print("We found exception for instance ", row_idx)
                continue


            print("We finished instance", row_idx, "with video length of", len(video_np), "frames processed")


            # Log update 
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

    # Don't forget "conda activate spatracker2" !!!

    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_offset', type=int, default=0)
    args = parser.parse_args()


    # Fundamental Setting
    csv_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_panoptic"                    # Input
    store_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_camera_estimation"         # Output
    VGGTTrack_path = "../pretrained/SpatialTrackerV2_Front"         # Check their codebase for the pretrained weight download
    GPU_offset = args.GPU_offset
    speedup_factor = 3           # We will not choose all frames, but have some interpolation; 3 is the level used in the official codebase
    max_sec_consider = 6        # -1 means the longest time avilable
    


    # Prepare the folder
    if not os.path.exists(store_folder_path):
        os.makedirs(store_folder_path)


    # Process
    single_process(csv_folder_path, store_folder_path, GPU_offset, speedup_factor, max_sec_consider)


    print("Finished!")