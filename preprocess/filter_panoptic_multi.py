
'''
    This file is used to segment multiple frames in the video by panoptic segmentation. We still store sampled points at the end
    "conda activate oneformer" with oneformer environment is needed
'''

import argparse
import numpy as np
from PIL import Image
import cv2
import random
import imutils
import time
import ffmpeg
import json
import collections
import csv
import subprocess
import os, sys, shutil
from sklearn.cluster import KMeans


import torch
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from huggingface_hub import hf_hub_download


# Import files from the local path
root_path = os.path.abspath('.')
sys.path.append(root_path)
from preprocess.oneformer_code.oneformer import (
                                                    add_oneformer_config,
                                                    add_common_config,
                                                    add_swin_config,
                                                    add_dinat_config,
                                                )
from preprocess.oneformer_code.demo.defaults import DefaultPredictor
from preprocess.oneformer_code.demo.visualizer import Visualizer, ColorMode



KEY_DICT = {
                "COCO (133 classes)": "coco",
            }

SWIN_CFG_DICT = {
                    "coco": "preprocess/oneformer_code/configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml",
                }

SWIN_MODEL_DICT = {
                    "coco": hf_hub_download(repo_id="shi-labs/oneformer_coco_swin_large", 
                                                    filename="150_16_swin_l_oneformer_coco_100ep.pth"),
                }

DINAT_CFG_DICT = {
                    "coco": "oneformer_code/configs/coco/oneformer_dinat_large_bs16_100ep.yaml",
                }

DINAT_MODEL_DICT = {
                    "coco": hf_hub_download(repo_id="shi-labs/oneformer_coco_dinat_large", 
                                                    filename="150_16_dinat_l_oneformer_coco_100ep.pth"),
                    }

MODEL_DICT = {"DiNAT-L": DINAT_MODEL_DICT,
                "Swin-L": SWIN_MODEL_DICT }

CFG_DICT = {"DiNAT-L": DINAT_CFG_DICT,
            "Swin-L": SWIN_CFG_DICT }

WIDTH_DICT = {
                "coco": 512,
            }

cpu_device = torch.device("cpu")

PREDICTORS = {
    "DiNAT-L": {
                    "COCO (133 classes)": None,
                },
    "Swin-L": {
                "COCO (133 classes)": None,
            }
}

METADATA = {
    "DiNAT-L": {
                    "COCO (133 classes)": None,
                },
    "Swin-L": {
                    "COCO (133 classes)": None,
                }
}


# HACK:  MOTIONABLE_OBJECT is a subset of the REFERENCE_OBJECT_CLASS
# HACK:  REFERENCE_OBJECT_CLASS + NON_OBJECT_CLASS = ALL Unit
MOTIONABLE_OBJECT = [
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
                        'sports ball', 'kite', 'flower', 
                        # We delete:
                        # Belows are newly added cases
                        'snowboard', 'surfboard', 'skateboard',
                    ]

# NOTE: We want to make it simpler for the object motion CTRL case, so neglect some that may be useful in the Camera CTRL
REFERENCE_OBJECT_CLASS = [
                            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
                            'bird', 'cat', 'dog', 
                            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 
                            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
                            'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 
                            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 
                            'pizza', 'donut', 'cake', 'chair', 'dining table', 'laptop', 'mouse', 'remote', 
                            'keyboard', 'cell phone', 'book', 'clock', 
                            'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'blanket', 'cardboard', 'counter',
                            'flower', 'fruit', 'pillow', 'towel', 'food-other-merged', 'door-stuff',
                        ]

NON_OBJECT_CLASS = [
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'tv', 'potted plant', 'couch', 'parking meter', 'fire hydrant', 'stop sign',
                        'toilet', 'banner', 'net', 'platform', 'road', 'snow', 'sea', 'railroad', 'roof', 'traffic light', 'bench', 
                        'floor-wood', 'gravel', 'light', 'playingfield', 'mountain-merged', 'water-other', 'wall-brick', 'wall-stone', 
                        'wall-tile', 'rock-merged', 'mirror-stuff', 'sand', 'bed', 'bridge', 'stairs', 'house', 'vase', 'curtain',
                        'grass-merged', 'dirt-merged', 'paper-merged', 'window-blind', 'building-other-merged',  'shelf', 'tent',
                        'wall-other-merged', 'rug-merged', 'river', 'window-other', 'fence-merged', 'ceiling-merged', 'tree-merged', 
                        'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'wall-wood', 
                    ]


def setup_modules():

    # Only load one kinds of model
    dataset = "COCO (133 classes)"
    backbone = "Swin-L"


    cfg = setup_cfg(dataset, backbone)

    metadata = MetadataCatalog.get(
                                    cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
                                )

    PREDICTORS[backbone][dataset] = DefaultPredictor(cfg)
    METADATA[backbone][dataset] = metadata

    return metadata


def setup_cfg(dataset, backbone):

    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_oneformer_config(cfg)
    add_dinat_config(cfg)
    dataset = KEY_DICT[dataset]
    cfg_path = CFG_DICT[backbone][dataset]
    cfg.merge_from_file(cfg_path)
    if torch.cuda.is_available():
        cfg.MODEL.DEVICE = 'cuda'
    else:
        raise Exception('You must have the CUDA GPU!')
    cfg.MODEL.WEIGHTS = MODEL_DICT[backbone][dataset]
    cfg.freeze()
    
    return cfg


def panoptic_run(img, predictor, debug = False):

    # Inference Execute
    predictions = predictor(img, "panoptic")
    panoptic_seg, segments_info = predictions["panoptic_seg"]
    
    if debug:

        visualizer = Visualizer(img[:, :, ::-1], metadata = metadata, instance_mode=ColorMode.IMAGE)
        out = visualizer.draw_panoptic_seg_predictions(
                                                        panoptic_seg.to(cpu_device), segments_info, alpha=0.5
                                                    )
        out = Image.fromarray(out.get_image())

        visualizer_map = Visualizer(img[:, :, ::-1], is_img=False, metadata = metadata, instance_mode=ColorMode.IMAGE)
        out_map = visualizer_map.draw_panoptic_seg_predictions(
                                                                panoptic_seg.to(cpu_device), segments_info, alpha=1, is_text=False
                                                            )
        out_map = Image.fromarray(out_map.get_image())

        save_name = "segmentation" + str(cur_idx) + "_" + str(cur_frame_idx) + ".png"
        out.save(save_name)
        print("Save ", save_name)
        
    return panoptic_seg, segments_info




@torch.no_grad()
def segment(img, dataset, backbone, debug):

    # Set the model
    predictor = PREDICTORS[backbone][dataset]
    metadata = METADATA[backbone][dataset]

    # Read the image
    width = WIDTH_DICT[KEY_DICT[dataset]]
    img = imutils.resize(img, width = width)

    # Inference
    panoptic_seg, segments_info = panoptic_run(img, predictor, debug)


    return panoptic_seg, segments_info, metadata




def get_frame_types(video_path):

    # ffmpeg-based fetch
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_path]).decode()
    frame_types = out.replace('pict_type=','').split()

    # return
    return frame_types




def get_closest_IFrame(frame_types, cur_ids, max_shift_allow):
    # NOTE: max_shift_allow is set to be slightly less than 1 second

    distance = [10000] * len(cur_ids)      # This hold the distance to the closest I-Frame
    new_ids = [0] * len(cur_ids)           # This updates the new idx to be rewritten

    # Iterate to find the closest I-Frame index one
    for frame_idx, frame_type in enumerate(frame_types):

        if frame_type == "I":
            
            # Iterate all target index frame
            for idx in range(len(cur_ids)):

                # If we find a closer I-Frame position
                if distance[idx] > abs(frame_idx - cur_ids[idx]):
                    new_ids[idx] = frame_idx

                    # Update the current closest distance to any I-Frame
                    distance[idx] = abs(frame_idx - cur_ids[idx])


    # If over max_shift_allow, use the original one
    for idx in range(len(distance)):
        if abs(new_ids[idx] - cur_ids[idx]) > max_shift_allow:
            # print("The change of frame num is too much, so we still use the original frame index")
            new_ids[idx] = cur_ids[idx] 

    # breakpoint()    # cur_ids, new_ids
    return new_ids



def draw_points(frame, cluster_points, height, width, resize_dot_radius, color_code):

    frame_draw = np.copy(frame)

    # Iterate all points
    for (vertical, horizontal) in cluster_points:

        # Draw square around the target position
        vertical_start = int(min(height, max(0, vertical - resize_dot_radius)))
        vertical_end = int(min(height, max(0, vertical + resize_dot_radius)))       # Diameter, used to be 10, but want smaller if there are too many points now
        horizontal_start = int(min(width, max(0, horizontal - resize_dot_radius)))
        horizontal_end = int(min(width, max(0, horizontal + resize_dot_radius)))

        # Paint
        frame_draw[vertical_start:vertical_end, horizontal_start:horizontal_end, :] = color_code   

    return frame_draw



@torch.no_grad()
def single_process(csv_folder_path, store_folder_path, GPU_offset, sample_duration_ratio, use_MOTIONABLE_OBJECT,
                   max_IFrame_shift_ratio_allowed, area_range, sample_point_num_range, debug):

    # Setting
    store_freq = 50
    task = "panoptic"
    backbone = "Swin-L"
    dataset = "COCO (133 classes)"


    # Read the csv file
    csv_idx = GPU_offset
    csv_file_path = os.path.join(csv_folder_path, "sub" + str(csv_idx) + ".csv")
    print("CSV file we read is ", csv_file_path)


    # Prepare the store file path
    store_file_path = os.path.join(store_folder_path, "sub" + str(csv_idx) + ".csv")
    if os.path.exists(store_file_path):
        # Remove existing csv
        os.remove(store_file_path)


    # Read all row in the csv file
    start_time = time.time()
    info_lists = []
    not_match_time = 0
    with open(csv_file_path) as file_obj: 
        reader_obj = csv.reader(file_obj) 
        
        # Iterate over each row in the csv  
        for row_idx, row in enumerate(reader_obj): 
            global cur_idx
            cur_idx = row_idx

            # For the first row case (With all title content)
            if row_idx == 0:    # The first line is the title of content
                elements = dict()
                for element_idx, key in enumerate(row):
                    elements[key] = element_idx

                print("The first row is ", row + ["Panoptic_Segmentation"])

                # Store the csv
                with open(store_file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows([row + ["Panoptic_Segmentation"]])
                continue


            # Read important information
            video_path = row[elements["video_path"]]
            fps = float(row[elements["fps"]])
            height = int(row[elements["height"]])
            width = int(row[elements["width"]])
            valid_duration = json.loads(row[elements["valid_duration"]])
            


            # Read the video
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



            ####################################### Define the index based on closest I-Frame located ######################################

            # Find the initial position
            initial_position = None
            raw_ids = []
            fps_scale = preset_decode_fps / fps
            downsample_num_frames = int(num_frames * fps_scale)
            for ratio in sample_duration_ratio:

                # Fetch the frame idx
                frame_idx = int(ratio * num_frames)

                # Check if two choosen frame is too close initially
                if len(raw_ids) >= 1:
                    if frame_idx - raw_ids[-1] <= min_num_frame_gap:
                        print("Sample Gap is too short between two selected frame for frame idx", frame_idx)
                        continue
                
                # Make sure there is enough number of frames 
                downsample_start_frame_idx = max(0, int(frame_idx * fps_scale))
                max_step_num = (downsample_num_frames - downsample_start_frame_idx) // train_frame_num
                if max_step_num == 0:      
                    print("The remaining frame is not enough for frame idx", frame_idx, " which has ", downsample_num_frames - downsample_start_frame_idx, "frame to train left")
                    continue
                
                # Append to the raw idx list
                raw_ids.append(frame_idx)
            
            # Should have the I-Frame works good
            if len(raw_ids) == 0:   
                print("We cannot find ideal one!")
                continue
            print("Old idx is ", raw_ids)


            # Get each Frame Type and Crop based on the valid duration
            frame_types = get_frame_types(video_path)       # TODO: I want to write assert(len(frame_types) == num_frames)
            
            # Sanity Check if Frame Type length and the full video length is matched. If it is not matched, 1 frame error is a catastrophe for the I-Frame-based methods
            if len(frame_types) != len(video_full_np):
                not_match_time += 1
                print("The ffprobe and ffmpeg read frames cannot match, so we don't use this I-Frame methods for", not_match_time, "times, which are", len(frame_types), len(video_full_np))
                ideal_frame_start_idxs = raw_ids

            else:

                # Crop based on the the valid duration
                frame_types = frame_types[valid_duration[0] : valid_duration[1]]
                
                # Find the Closest I-Frame
                ideal_frame_start_idxs = get_closest_IFrame(frame_types, raw_ids, max_shift_allow = num_frames * max_IFrame_shift_ratio_allowed)
            
            print("New idx is ", ideal_frame_start_idxs)

            ################################################################################################################################



            # While loop until the last feasible frame
            effective_info = []
            for frame_idx in ideal_frame_start_idxs:
                
                # Read the frame
                global cur_frame_idx
                cur_frame_idx = frame_idx
                frame = video_np[frame_idx]     # NOTE: this is already cropped video by valid duration
                

                # Process the Panoptic Segmentation
                # try:
                panoptic_seg, segments_info, metadata = segment(frame, dataset, backbone, debug)        # TODO: 为神马这个metadata是不停的在覆盖中？

                # except Exception:
                #     print("Segment Raise error")
                #     continue
                

                # Iterate all the segment info that match the criteria
                current_frame_info = []
                class_num_dict = collections.defaultdict(int)
                for segment_idx, segment_info in enumerate(segments_info):
                    
                    # Read the value
                    id_num = segment_info['id']     # This is the unique id number
                    category_id = segment_info['category_id']
                    text_name = metadata.stuff_classes[category_id]
                    height_panoptic, width_panoptic = panoptic_seg.shape
                    

                    # Must have the class type in pre-defined case
                    if use_MOTIONABLE_OBJECT:
                        if not text_name in MOTIONABLE_OBJECT:
                            print("This object type is not what we want", text_name)
                            continue
                    else:
                        if not text_name in REFERENCE_OBJECT_CLASS:
                            print("This object type is not what we want", text_name)
                            continue
                    

                    # Calculate the relative area and  Check if the area is too small or too big
                    map_area = (panoptic_seg == id_num).cpu().numpy()
                    relative_area = int(np.sum(map_area)) / (height_panoptic * width_panoptic)
                    if relative_area < area_range[0] or relative_area > area_range[1]:
                        print("The relative area of ", text_name, "is not ideal for", relative_area, ", which is from segment idx of", segment_idx)
                        continue
                        # TODO: Write a failure log

                    # Visualize
                    # seg_img = np.repeat(map_area[:, :, np.newaxis], 3, axis=2) * cv2.resize(frame, (width_panoptic, height_panoptic))
                    # cv2.imwrite("img_case"+str(row_idx)+"_frame"+str(frame_idx)+"_segment"+str(segment_idx)+text_name+".png", cv2.cvtColor(seg_img, cv2.COLOR_BGR2RGB))

 

                    ############################## After this line, all are valid cases ##################################


                    # Re-Sample points from the complete segmentation mask
                    all_segmentation_points = []
                    true_values = np.where(map_area)
                    for idx in range(len(true_values[0])):
                        if random.random() < point_selection_ratio:
                            vertical, horizontal  = true_values[0][idx], true_values[1][idx]
                            all_segmentation_points.append([vertical, horizontal])      # NOTE: 这里又变回了跟原本order一致的points


                    # Do K-means Sampling where the cluster number is choosen in the linear range
                    n_clusters = sample_point_num_range[0] + int((sample_point_num_range[1] - sample_point_num_range[0]) * (relative_area - area_range[0]) / (area_range[1] - area_range[0]))
                    kmean_points = KMeans(n_clusters = n_clusters, random_state = 0, n_init = "auto").fit(all_segmentation_points)
                    cluster_centers_raw = np.int64(kmean_points.cluster_centers_)


                    # Sanity Check the area of the mask and then 
                    cluster_centers_filterred = []
                    for cluster_center in cluster_centers_raw:
                        (cord_y, cord_x) = cluster_center

                        # Make sure that the point is inside the mask
                        if map_area[cord_y][cord_x] == True:
                            
                            # Check if this is watermark region     NOTE: Should be very needed in current curation pipeline, because we discard slow motion poiints (which track on the watermark), and no more watermark dataset for OpenVID and VidGen
                            # if watermark_avoid_region is not None:
                            #     start_ratio, end_ratio = watermark_avoid_region
                            #     start_width, start_height = start_ratio[0] * width, start_ratio[1] * height
                            #     end_width, end_height = end_ratio[0] * width, end_ratio[1] * height

                            #     if (cord_y < end_height) and (cord_y > start_height) and (cord_x < end_width) and (cord_x > start_width):
                            #         continue
                            
                            # Resize the cordinate to the original height and width
                            cord_y = int(cord_y * height / height_panoptic)
                            cord_x = int(cord_x * width / width_panoptic)

                            # Append to the valid lists
                            cluster_centers_filterred.append((cord_y, cord_x))
                    cluster_centers_filterred = np.array(cluster_centers_filterred).tolist()


                    # Must have something at least one valid point after the filter
                    if len(cluster_centers_filterred) == 0:
                        continue

                    

                    # Visualize the figure with kmeans
                    # frame_draw = draw_points(frame, cluster_centers_filterred, height, width, resize_dot_radius=7, color_code=(255, 0, 0))
                    # cv2.imwrite("VisImg_case"+str(row_idx)+"_frame"+str(frame_idx)+"_segment"+str(segment_idx)+text_name+".png", cv2.cvtColor(frame_draw, cv2.COLOR_RGB2BGR))


                    # Append one instance information to the list
                    current_frame_info.append([id_num, text_name, cluster_centers_filterred])   

                    # Update the number of class
                    class_num_dict[text_name] += 1


                # Check if we have too many same class label (which make multi-tracking too hard)
                # NOTE: 这个放弃了，多一点同case的video也没啥吧
                # too_many_same_case = False
                # for text_name in class_num_dict:
                #     if class_num_dict[text_name] > max_same_class_num:
                #         print("There are too many label for ", text_name, " for video ", video_path)
                #         too_many_same_case = True
                #         break
                # if too_many_same_case:
                #     continue


                ############################# Upon this point, all are ideal cases ##############################
                # Collect current frame info
                if len(current_frame_info) != 0:
                    effective_info.append((frame_idx, current_frame_info))
                else:
                    print("No segmentation detected for frame idx", frame_idx)



            # Append new row information for valid video case
            if len(effective_info) != 0:    # If we didn't find any effective info, we don't write this to the csv
                # frame_idx is based on the original FPS setting
                info_lists.append(row + [json.dumps(effective_info)])  
                print("We accept video ", video_path)   
            else:
                print("Cannot find any ideal frame for ", video_path)

            # Log update (The update will be quite random for this file, because we may skip earlier on)
            if row_idx % store_freq == 0:
                
                print("We have processed ", float(row_idx/1000), "K video")
                print("The number of valid videos we found in this iter is ", len(info_lists), " over", store_freq, "samples")
                full_time_spent = int(time.time() - start_time)
                print("Time spent is %d min %d s" %(full_time_spent//60, full_time_spent%60))

                # Store the csv
                with open(store_file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(info_lists)

                # Restart the info lists
                info_lists = []       


        # Final Log update
        with open(store_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(info_lists)




if __name__ == "__main__": 

    # NOTE: Please  "conda activate oneformer"

    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_offset', type=int, default=0)
    parser.add_argument('--debug', type=bool, default = False)
    args = parser.parse_args()


    # Fundamental Setting
    csv_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_scoring_img_left"        # Input 
    store_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_panoptic"      # Output
    GPU_offset = args.GPU_offset
    

    # Sample Setting
    sample_duration_ratio = [0.0, 0.33, 0.66]    # Relative to the whole duration [0.01, 0.35, 0.7]  # 0.01 (less than max_IFrame_shift_ratio_allowed) is to prevent the full black case    
    max_IFrame_shift_ratio_allowed = 0.05        # Maximum I-Frame adjustment shift alllowed; this will be based on total number of frames available
    min_num_frame_gap = 50                       # Between two sample, it must be over this gap; there is no need for dense sampling
    preset_decode_fps = 12
    train_frame_num = 49


    # Tracking Points Setting
    area_range = [0.033, 0.4]           # Only area in this relative range is effective  NOTE: Now the max area becomes 0.4, smaller than before
    point_selection_ratio = 0.15        # How much percent of points is left from segmentation mask
    sample_point_num_range = [8, 26]    # This is relative to the area_range setting; 
    use_MOTIONABLE_OBJECT = True        # Set to True for now
    # max_same_class_num = 100              # We don't want too many valid motion with the same class



    # First, let us setup the modules
    setup_modules()


    # Prepare the folder
    if not os.path.exists(store_folder_path):
        os.makedirs(store_folder_path)


    # Process
    single_process(csv_folder_path, store_folder_path, GPU_offset, sample_duration_ratio, use_MOTIONABLE_OBJECT,
                    max_IFrame_shift_ratio_allowed, area_range, sample_point_num_range, debug = args.debug)


    print("Finished!")
