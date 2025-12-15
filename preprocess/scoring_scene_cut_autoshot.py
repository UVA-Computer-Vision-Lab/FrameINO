'''
    Scene Cut using the model of AutoShot
'''

import os, sys, shutil
import pandas as pd
import time
import csv
import collections
from multiprocessing import Process
import multiprocessing
import cv2
import numpy as np
from torchvision import transforms
import random
from PIL import Image
import argparse
import torch
import ffmpeg
import matplotlib.pyplot as plt
import json
csv.field_size_limit(sys.maxsize)


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from preprocess.auxiliary.AutoShot import TransNetV2Supernet




def get_batches(frames):
    reminder = 50 - len(frames) % 50
    if reminder == 50:
        reminder = 0
    frames = np.concatenate([frames[:1]] * 25 + [frames] + [frames[-1:]] * (reminder + 25), 0)

    def func():
        for i in range(0, len(frames) - 50, 50):
            yield frames[i:i + 100]

    return func()


def predict_probs_for_video(model, device, frames: np.ndarray) -> np.ndarray:
    """
        Run the model on one video and return per-frame probabilities in [0,1], length T.
        Matches the repo's behavior by taking center 50 frames from each 100-frame batch: [25:75].
    """

    probs = []
    with torch.no_grad():
        for batch in get_batches(frames):  # expects np.ndarray [100,H,W,3] or similar
            # to torch: [B=1, C=3, T, H, W]
            t = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...]).float().to(device)
            one_hot = model(t)
            if isinstance(one_hot, tuple):
                one_hot = one_hot[0]
            out = torch.sigmoid(one_hot[0])  # shape [T_batch]
            # follow original script: keep the center slice
            out = out.detach().cpu().numpy()
            probs.append(out[25:75])

    if len(probs) == 0:
        print("  WARN: no batches produced any output.")
        return np.zeros((len(frames),), dtype=np.float32)

    probs = np.concatenate(probs, axis=0)
    # trim/pad to match video length
    probs = probs[:len(frames)]

    if len(probs) < len(frames):
        pad = np.zeros((len(frames) - len(probs),), dtype=probs.dtype)
        probs = np.concatenate([probs, pad], axis=0)

    return probs




@torch.no_grad
def single_process(csv_folder_path, store_folder_path, GPU_offset):

    # Setting
    task_name = "SceneCut_AutoShot"
    store_freq = 50


    # Read the csv file
    csv_idx = GPU_offset
    csv_file_path = os.path.join(csv_folder_path, "sub" + str(csv_idx) + ".csv")
    print("CSV file we read is ", csv_file_path)


    # Prepare the store csv path
    store_file_path = os.path.join(store_folder_path, "sub" + str(csv_idx) + ".csv")
    if os.path.exists(store_file_path):
        # Remove existing csv
        os.remove(store_file_path)



    # Init the model
    device = "cuda"
    model = TransNetV2Supernet().eval()
    model = model.to(device)


    # Load checkpoint and filter keys to match current model state_dict
    sd = model.state_dict()
    ckpt = torch.load(pretrained_weight_path, map_location=device)
    if "net" in ckpt:
        ckpt = ckpt["net"]
    filtered = {k: v for k, v in ckpt.items() if k in sd}
    sd.update(filtered)
    missing = [k for k in sd.keys() if k not in filtered]
    if len(filtered) == 0:
        print("[WARN] None of the checkpoint keys matched the model. "
              "Double-check you're using the right supernet file.", file=sys.stderr)
    model.load_state_dict(sd)




    # Read all row in the csv file
    start_time = time.time()
    info_lists = []       # The order will be follow automatically
    with open(csv_file_path) as file_obj: 
    
        reader_obj = csv.reader(file_obj) 
        
        # Iterate over each row in the csv  
        for idx, row in enumerate(reader_obj): 
            exception_case = 0


            if idx == 0:    # The first line is the title of content

                elements = dict()
                for element_idx, key in enumerate(row):
                    elements[key] = element_idx

                print("The first row is ", row + [task_name])

                # Store the csv
                with open(store_file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows([row + [task_name]])

                continue

            # Read the important information
            video_path = row[elements["video_path"]]
            valid_duration = json.loads(row[elements["valid_duration"]])
            


            try:

                # Read the video
                video_stream, err = ffmpeg.input(
                                                    video_path
                                                ).output(
                                                    "pipe:", format = "rawvideo", pix_fmt = "rgb24", s = "48x27", vsync = 'passthrough',
                                                ).run(
                                                    capture_stdout=True, capture_stderr=True
                                                )       # Different from other curaiton, we set a low resolution Setting here.
                video_np = np.frombuffer(video_stream, np.uint8).reshape([-1, 27, 48, 3])

                # Fetch in the valid duration range
                video_np = video_np[valid_duration[0] : valid_duration[1]]


                # Predict the Number of Scene Cut
                probs = predict_probs_for_video(model, device, video_np)
                prediction_labels = (probs > threshold).astype(np.uint8)


                # Convert to range in scenes
                scenes = []
                cur_begin_idx = 0
                for frame_idx, label in enumerate(prediction_labels):
                    
                    # The value 1 in the prediction outputs refers to the scene cut signal.
                    if label[0] == 1 or frame_idx == len(prediction_labels) - 1:        # Either we have value 1 or the last one the label list
                        scenes.append([cur_begin_idx, frame_idx+1])       # Closed left and open right range
                        cur_begin_idx = frame_idx + 1
                        
                print("We find scenes of range", scenes, "for video", video_path, "of duration", valid_duration)


                # Append to the list with other information
                info = row + [scenes]
                info_lists.append(info)


                # Log update
                if idx % store_freq == 0:

                    print("We have processed", float(idx/1000), "K video")
                    print("The number of valid videos we found in this iter is ", len(info_lists))
                    full_time_spent = int(time.time() - start_time)
                    print("Time spent for now is %d min %d s" %(full_time_spent//60, full_time_spent%60))

                    # Store the csv
                    with open(store_file_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows(info_lists)

                    # Restart the info_lists
                    info_lists = []   


            except Exception:
                print("There is exception cases")
                continue    # For any error occurs, we just skip
            


        # Final Log update
        with open(store_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(info_lists)





if __name__ == "__main__":

    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_offset', type=int, default=0)
    args = parser.parse_args()
    GPU_offset = args.GPU_offset


    # Fundamental Setting
    csv_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_filter_basic"            # Input
    store_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_scoring_SceneCut"      # Output
    pretrained_weight_path = "../pretrained/ckpt_0_200_0.pth"           # Weight Path (needs to download from their original website)
    threshold = 0.296       # Empricial Setting for the threshold



    # Prepare the csv file
    if not os.path.exists(store_folder_path):
        os.makedirs(store_folder_path)



    # Single Process
    single_process(csv_folder_path, store_folder_path, GPU_offset)


    print("Finished!")



