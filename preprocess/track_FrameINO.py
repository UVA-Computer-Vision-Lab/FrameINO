
'''
    Track Frame In and Frame Out cases at the same time.
    We require ID to be outside the region box, but don't intentionally find FrameOut cases, because learning arbitrary motion cases with a small region box must include Frame Out cases.
'''

import os, sys, shutil
import argparse
import numpy as np
from PIL import Image, ImageDraw
import cv2
import random
import ffmpeg
import time
import json
import csv
import imageio.v3 as iio
import torch
from torchvision import transforms
from sam2.sam2_video_predictor import SAM2VideoPredictor 
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

# Data Transforms
train_transforms = transforms.Compose(
                                        [
                                            transforms.Lambda(lambda x: x / 255.0 * 2.0 - 1.0),
                                        ]
                                    )




def prepare_traj_visual(full_pred_tracks, full_visibility_tracks, original_height, original_width, selected_frames, 
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
        base_img = cv2.filter2D(base_img, -1, blur_kernel).astype(np.uint8)


        # Append selected_frames and the color together for visualization
        merge_frame = selected_frames[temporal_idx].copy()
        merge_frame[base_img < 250] = base_img[base_img < 250]
        cv2.imwrite("Video"+name + "_traj" + str(temporal_idx).zfill(2) + ".png", cv2.cvtColor(merge_frame, cv2.COLOR_RGB2BGR)) # The base_img is designed in RGB form


        # Append to the temporal index
        traj_img_lists.append(base_img)
    
        
    # Write to video (For Debug Purpose)
    os.system("ffmpeg -loglevel quiet -r 12 -f image2 -i Video"+name+"_traj%02d.png -vcodec libx264 -crf 30 -pix_fmt yuv420p gen_video_" + name + ".mp4")

    # Clean frames
    os.system("rm Video" + name + "_traj*.png")



def visualize_motion(Track_Traj, Obj_info, Track_Visibility, height, width, video_np, dot_radius, max_step_num, region_box=None, name=""):

    
    # Convert Track_Traj to the data form of (temporal, object, points, xy)
    train_frame_num = 49
    if max_step_num > 1:
        iter_gap = 1
    else:
        iter_gap = 1

    

    # Draw the Region Box Area
    if region_box is not None:
        _, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) = region_box
        new_video_np = []
        for frame_idx, frame in enumerate(video_np):
            new_video_np.append(cv2.rectangle(frame.copy(), (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (255, 0, 0), 5))
        video_np = np.array(new_video_np)


    # Read the basic info
    text_name, original_start_frame_idx, fps_scale = Obj_info      # This is expected to be all the same in the video
    downsample_panoptic_frame_idx = int(fps_scale * original_start_frame_idx)

    # Prepare the tracjectory
    pred_tracks = Track_Traj
    visibility_tracks = Track_Visibility

    # Fetch the ideal range needed and Convert to the full_pred_tracks data structure
    selected_frames = video_np[downsample_panoptic_frame_idx : downsample_panoptic_frame_idx + iter_gap * train_frame_num : iter_gap]
    pred_tracks = pred_tracks[downsample_panoptic_frame_idx : downsample_panoptic_frame_idx + iter_gap * train_frame_num : iter_gap]   
    visibility_tracks = visibility_tracks[downsample_panoptic_frame_idx : downsample_panoptic_frame_idx + iter_gap * train_frame_num : iter_gap]
    if len(pred_tracks) != train_frame_num:
        raise Exception("The len of pre_track does not match")

    # Change the dim form to be usable in the following traj cases
    full_pred_tracks = [[] for _ in range(train_frame_num)]     # The dim should be: (temporal, object, points, xy) The fps should be fixed to 12 fps, which is the same as training decode fps
    full_visibility_tracks = [[] for _ in range(train_frame_num)]
    for temporal_idx, pred_track in enumerate(pred_tracks):
        full_pred_tracks[temporal_idx].append(pred_track)       # pred_tracks will be 49 frames, and each one represent all tracking points for single objects; only one object here
        full_visibility_tracks[temporal_idx].append(visibility_tracks[temporal_idx])


    # Call the Drawing function
    prepare_traj_visual(full_pred_tracks, full_visibility_tracks, height, width, selected_frames, dot_radius, width, height, name = name)



def draw_circle(rgb, coord, radius, color=(255, 0, 0), visible=True, color_alpha=None):
    
    # Create a draw object
    draw = ImageDraw.Draw(rgb)
    # Calculate the bounding box of the circle
    left_up_point = (coord[0] - radius, coord[1] - radius)
    right_down_point = (coord[0] + radius, coord[1] + radius)
    # Draw the circle
    color = tuple(list(color) + [color_alpha if color_alpha is not None else 255])

    draw.ellipse(
        [left_up_point, right_down_point],
        fill=tuple(color) if visible else None,
        outline=tuple(color),
    )
    return rgb


def mask_to_bbox(mask):
    # Mask shape is (1, height, width)
    mask = mask[0]

    # Find rows and columns where the mask is not zero
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        # If there are no nonzero values in the mask, return None
        return None

    # Find the bounding box's edges
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    # Return the bounding box coordinates
    return int(xmin), int(ymin), int(xmax), int(ymax)



def SAM2_Refine(sam2_predictor, all_frames, GPU_offset, visual_store_folder, obj_Track_Traj, obj_Track_Visibility, 
                    original_start_frame_idx, potential_useful_region_box, min_area_required, debug):


    # Prepare the folder needed
    tmp_store_folder = "tmp_SAM2/process" + str(GPU_offset)
    if os.path.exists(tmp_store_folder):
        shutil.rmtree(tmp_store_folder)
    os.makedirs(tmp_store_folder)
    if os.path.exists(visual_store_folder):
        shutil.rmtree(visual_store_folder)
    os.makedirs(visual_store_folder)


    # Init the tracking
    panoptic_track_points_start = obj_Track_Traj[original_start_frame_idx]
    point_valid_status = [True for _ in range(len(panoptic_track_points_start))]


    # Write to a temp storage available
    sam_frames = all_frames[original_start_frame_idx : ]
    for frame_idx, frame in enumerate(sam_frames):
        store_img_path = os.path.join(tmp_store_folder, str(frame_idx).zfill(4) + ".jpg")
        cv2.imwrite(store_img_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


    # SAM Process
    info, bbox_info = [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):

        # Init the state
        state = sam2_predictor.init_state(tmp_store_folder)
        sam2_predictor.reset_state(state)

        # Add new prompts and instantly get the output on the same frame
        labels = np.array([1] * len(panoptic_track_points_start), np.int32)  # All are the same label
        frame_idx, object_ids, masks = sam2_predictor.add_new_points_or_box(
                                                                                state, 
                                                                                frame_idx = 0,
                                                                                obj_id = 1,     # Only consider single isntance now
                                                                                points = panoptic_track_points_start,  # Use points in the first frame
                                                                                labels = labels,
                                                                            )

        # Iterate all frames and Recognize multiple masks
        for frame_idx, object_ids, masks in sam2_predictor.propagate_in_video(state, start_frame_idx=0):
            for obj_idx, out_obj_id in enumerate(object_ids):

                # Convert to boolean mask and 3 channels
                segmentation_mask_raw = (masks[obj_idx] > 0.0).cpu().numpy().astype(np.uint8)
                _, height, width = segmentation_mask_raw.shape
                segmentation_mask_cat = np.stack([segmentation_mask_raw, segmentation_mask_raw, segmentation_mask_raw], axis=-1)[0]     

                # Prepare masked segmentation images
                segmentation_img = segmentation_mask_cat * cv2.cvtColor(sam_frames[frame_idx], cv2.COLOR_BGR2RGB)
                segmentation_mask = segmentation_mask_raw * 255


                # Re-Sample points from the accurate SAM mask
                for point_idx, (cord_x, cord_y) in enumerate(obj_Track_Traj[original_start_frame_idx + frame_idx]):
                    if cord_y >= 0 and cord_y < height and cord_x >= 0 and cord_x < width:
                        if segmentation_mask_raw[0][cord_y][cord_x] == 0:   # If there is any point that we cannot find, it will be considerred as invalid case.
                            point_valid_status[point_idx] = False

                # Find the Bounding Box Area
                box_info = mask_to_bbox(segmentation_mask)


                # In the first frame, there must be several criteria satisfied; else, we skip
                if frame_idx == 0:
                    
                    #  Save the first segmentation mask (in the first frame)
                    first_segmentation_mask = segmentation_mask_raw[0]
                    first_processed_segmentation_mask = segmentation_mask_cat * 255

                    # We must guarantee the first frame can find a valid BBox, such that it is a valid case
                    if box_info is None:
                        break

                    # The first frame Region cannot be too small
                    xmin, ymin, xmax, ymax = box_info
                    area = (xmax - xmin) * (ymax - ymin)
                    if area < min_area_required:
                        break

                    # The Region must not be Corrupted or Confusing case: Only has hand
                

                # Append those large enough BBox
                if box_info is not None:
                    xmin, ymin, xmax, ymax = box_info
                    area = (xmax - xmin) * (ymax - ymin)

                    # They cannot be too small for training
                    if area < min_area_required:
                        continue

                    # Ideal case, we append it to the list
                    segmentation_img_with_bbox = cv2.rectangle(segmentation_img.copy(), (xmin, ymin), (xmax, ymax), (255,0,0), 3)
                    info.append((segmentation_img, segmentation_img_with_bbox, box_info, area))
                    bbox_info.append(box_info)      # Either None of not empty, append it

                break   # Only consider one SAM object, and we only takes points related to one object, so should ONLY has one object
        
        
        # Check the point skip ratio
        skip_ratio = 1 - (sum(point_valid_status) / len(point_valid_status))
        if skip_ratio > max_skip_ratio_allowed:
            print("Point skip ratio is too high!")
            shutil.rmtree(visual_store_folder)  # Save storage
            return [], [], []




        # Choose the mask from all temporally
        segmentation_info = []
        if len(info) != 0:

            ########################## Prepare the first frame ID information for return ##########################

            # Prepare the information needed
            xmin, ymin, xmax, ymax = bbox_info[0]
            (segmentation_img, segmentation_img_with_bbox, box_info, area) = info[0]
            first_frame_reference_path = os.path.join(visual_store_folder, "Main_Reference.png")
            cv2.imwrite(first_frame_reference_path, segmentation_img[ymin:ymax, xmin:xmax])

            # Append the full mask, just in case BBox crop is wrong and also for better visual
            # full_reference_path = os.path.join(visual_store_folder, "Full_Main_Reference.png")
            # cv2.imwrite(full_reference_path, segmentation_img)
            
            ########################################################################################################

            
            # Sort all masks based on the area of the mask (Larger should be better)
            info.sort(key=lambda x: x[-1])  
            choosen_start_idx = int(len(info) * 0.25)       # Discard smallest those
            selected_info = random.sample(info[choosen_start_idx:], min(SAM_min_reference_store_num, len(info) - choosen_start_idx))

            store_img_path_lists = []
            for reference_idx, (segmentation_img, segmentation_img_with_bbox, box_info, area) in enumerate(selected_info):
                
                # Crop the frame
                xmin, ymin, xmax, ymax = box_info
                cropped_img = segmentation_img[ymin:ymax, xmin:xmax]

                # Save img
                store_img_path = os.path.join(visual_store_folder, "Other_Reference"+str(reference_idx)+".png").zfill(3)
                store_img_path_lists.append(store_img_path)
                cv2.imwrite(store_img_path, cropped_img)


            # Write segmentation info 
            segmentation_info = [bbox_info[0], first_frame_reference_path, store_img_path_lists]      # This should contains 1st frame BBox Coordinate, and 1st frame 


    # If there is no valid case, we return earlier
    if segmentation_info == []:         # Pass this branch if the first frame doesn't has large enough ID Reference
        print("SAM2 cannot find ideal case we want and we will remove the folder to save storage")
        shutil.rmtree(visual_store_folder)  # Save storage
        return [], [], []



    # Iterate all potential useful region box we generate before
    useful_region_box = []       # potential_useful_region_box
    for region_box in potential_useful_region_box:
        area, (x_min, y_min), (x_max, y_max) = region_box

        # Find the overlap between ID and the potential region box
        no_overlap = np.all(first_segmentation_mask[y_min:y_max, x_min:x_max] == 0)
        if no_overlap:
            useful_region_box.append(region_box)


    # If there is not ideal Region Box found, we return empty lists information
    if len(useful_region_box) == 0:
        print("SAM2 filter all potential useful region box")
        shutil.rmtree(visual_store_folder)  # Save storage
        return [], [], []
    
    
    # If everything is good, return the information needed
    return segmentation_info, useful_region_box, point_valid_status





@torch.no_grad()
def single_process(csv_folder_path, store_folder_path, GPU_offset, debug):
    
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
            


    # Init the SAM2 model
    sam2_predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2.1-hiera-large")     # Large model has 224.4M Param at 39.5 FPS



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

                print("The first row is ", row + ["ID_info"])       # TODO: we need ID_path (Especially 1st frame) + Region BBox position (All corresponds to Motion Traj)

                # Store the csv
                if not resume:
                    with open(store_file_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows([row + ["ID_info"]])
                continue 


            # Fetch important information
            video_path = row[elements["video_path"]]
            # num_frames = int(row[elements["num_frames"]])
            fps = float(row[elements["fps"]])
            height = int(row[elements["height"]])
            width = int(row[elements["width"]])
            valid_duration = json.loads(row[elements["valid_duration"]])
            Obj_Info_raw = json.loads(row[elements["Obj_Info"]])            # Data Struct is: panoptic -> object -> 2 [id_type, frame_idx]
            Track_Traj_raw = json.loads(row[elements["Track_Traj"]])
            Track_Visibility_raw = json.loads(row[elements["Track_Visibility"]])
            print("This is instance", row_idx, "and we are processing", video_path)


            # Resume mode will execute until we have the last store row matched
            if resume:  # NOTE: 这里默认只比较video path，没有valid duration的参考（也就是默认scene cut没有crop多个instance）
                if video_path == last_store_row[elements["video_path"]]:     # Check ID and video Path (Should all)
                    find_resume = True      # We find the exact row we want
                    continue

            if not find_resume:
                continue
            

            # Log update NOTE: Put to the begging due to the complext sturcture of this file
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


            # Init variables
            ID_info_all = []  # Data structure should be (panoptic frame, object, 2 (segmentation_info, useful_region_box),  Specific info list)
            Obj_info_all, Track_Traj_all, Track_Visibility_all = [], [], []     # REWRITE previous info for important use


            # Iterate all Panoptic Frames
            for panoptic_idx, panoptic_Obj_info in enumerate(Obj_Info_raw):

                if len(panoptic_Obj_info) == 0:     # Sometimes there is no info inside (But should have such phenomenon for now)
                    print("No panoptic_Obj_info found!")
                    continue

                # Init and Fetch Variables
                panoptic_ID_info = []
                original_start_frame_idx = panoptic_Obj_info[0][1]       # NOTE: If there is multiple objects Obj_info[X][1] should be the same
                # 之前有check长度是否足够，现在被我放弃了; 还有以前直接去做downsample，现在不整了


                # Iterate all objects inside each panoptic frame
                effective_obj_num = 0
                for obj_idx, single_obj_info in enumerate(panoptic_Obj_info):

                    # Fetch and Process information
                    obj_Track_Traj = Track_Traj_raw[panoptic_idx][obj_idx]              # NOTE: should be corresponding to all frames in video, no jump
                    obj_Track_Visibility = Track_Visibility_raw[panoptic_idx][obj_idx]
                    panoptic_ID_info.append([])
                

                    ##################################### Randomly generate BBox with arbitrary aspect ratio #####################################################

                    # Fetch Object Motion information
                    first_frame_points = obj_Track_Traj[original_start_frame_idx]         # The first frame in the training


                    # Randomly generate Region Box and check if any of them is ideal and Finish STEP4. This is a more heruristic and regressive methods.
                    potential_useful_region_box = []
                    for _ in range(max_box_find_times):  # Iterate 1000 times to potentially find an ideal one

                        # NOTE: The Region is respect to the original resolution
                        random_idx = random.choices(range(len(fixed_aspect_ratios)), weights=fixed_aspect_ratio_prob)[0]     # Randomly choose one, to make it more robus 
                        fixed_aspect_ratio = fixed_aspect_ratios[random_idx]
                        new_unbounded_mask_box_scale_min = unbounded_mask_box_scale_min[random_idx]

                        # Random init a top left position
                        top_left_x = random.randint(0, int(unbounded_mask_box_top_left_position[0] * width))
                        top_left_y = random.randint(0, int(unbounded_mask_box_top_left_position[1] * height))

                        # Check if either reminaing height or width is less that the threshold available
                        width_left = width - top_left_x
                        height_left = height - top_left_y
                        if width_left < new_unbounded_mask_box_scale_min * width * fixed_aspect_ratio or height_left < new_unbounded_mask_box_scale_min * height:
                            continue
                
                        # Map to the bottom right position
                        region_height = random.randint(int(new_unbounded_mask_box_scale_min * height), height_left)
                        region_width = int(region_height * fixed_aspect_ratio)       # We require a 4:3 aspect ratio, so the width should be fixed also.
                        bottom_right_x = min(top_left_x + region_width, width)
                        bottom_right_y = min(top_left_y + region_height, height)
                        


                        # If the cropped frame is smaller than XXX region threshold, discard it. The BBox must not cover any initial position. If none of ideal BBox is found, drop.
                        outside_region = []
                        is_first_frame_outside_region = False


                        ###################################  Check If the First Frame is inside the Region Box  ###################################################
                        
                        for (point_x, point_y) in first_frame_points:
                            if point_x+outside_offset < top_left_x or \
                                point_x-outside_offset > bottom_right_x or \
                                    point_y+outside_offset < top_left_y or \
                                        point_y-outside_offset > bottom_right_y:
                                outside_region.append(True)
                            else:
                                outside_region.append(False)
                        
                        # We require all points (with offset) to be outside of the Region Box for the first frame
                        if sum(outside_region) == len(outside_region):
                            # See if All points are outside the region
                            is_first_frame_outside_region = True
                        else:
                            continue        # If the first frame is not ideal, continue; don't waste time on the following
                        
                        #############################################################################################################################################



                        # Check if the ID Reference is out of the BBox
                        if is_first_frame_outside_region:           # NOTE: the ID has no need to must come inside the region box (designed because of data scarcity)   
                            
                            # Append each possible random useful region for training
                            area = (bottom_right_x - top_left_x) * (bottom_right_y - top_left_y)
                            potential_useful_region_box.append([area, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y)])


                    # If none of the region box is ideal, we jump to the next ID object   
                    if len(potential_useful_region_box) == 0:

                        # Visualize for the failure case (Can comment Out later)
                        # video_stream, err = ffmpeg.input(
                        #                                 video_path
                        #                             ).filter(
                        #                                 'fps', fps = preset_decode_fps, round = 'up'
                        #                             ).output(
                        #                                 "pipe:", format = "rawvideo", pix_fmt = "rgb24",
                        #                             ).run(
                        #                                 capture_stdout = True, capture_stderr = True
                        #                             )      # The resize is already includede
                        # video_np = np.frombuffer(video_stream, np.uint8).reshape(-1, height, width, 3)
                
                        # visualize_motion(obj_Track_Traj, single_obj_info, obj_Track_Visibility, height, width, video_np, 6, 
                        #                     max_step_num, name="failure_case_row"+str(row_idx)+"_panoptic"+str(panoptic_idx)+"_obj"+str(obj_idx))
                        print("Cannot find any potential_useful_region_box")
                        continue

                    ##################################################################################################################################################################                            

            


                    ################################################## Read the video and Do Cropping   ##########################################################
                    
                    # Read the video by ffmpeg, not decod
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
                    # video_tensor = torch.tensor(video_np).to(device)      # Not Used
                    # num_frames = len(video_np)            # Not Used
                    
                    ################################################################################################################################################




                    ###############################  If the result is ideal, use SAM2 to fetch the accurate Segmentation mask and then store as images.  #########################################

                    # Prepare the Mask visual store position
                    visual_store_folder = os.path.join(visual_store_parent_folder, str(row_idx) + "_" + video_path.split('/')[-1].split('.')[0] + "_clip" + str(valid_duration[0]) + "_" + str(panoptic_idx) + "_" + str(obj_idx))
                    visual_store_folder = os.path.abspath(visual_store_folder)
                    min_area_required = SAM_min_area_ratio * height * width


                    # Execute SAM2 Refine, Should obtain 1st frame ID_path + 1st frame Region BBox position
                    segmentation_info, useful_region_box, point_valid_status = SAM2_Refine(
                                                                                                sam2_predictor, video_np, GPU_offset, visual_store_folder, obj_Track_Traj, 
                                                                                                obj_Track_Visibility, original_start_frame_idx, 
                                                                                                potential_useful_region_box, min_area_required, debug
                                                                                            )   
                    

                    # Check if SAM fails
                    if segmentation_info == []:     
                        # Visualize for Failure Case due to SAM
                        # sample_region_box = random.choice(potential_useful_region_box)
                        # visualize_motion(obj_Track_Traj, single_obj_info, obj_Track_Visibility, height, width, video_np, 6, 
                        #                     max_step_num, sample_region_box, name="FrameIn_SAM_Fail_row"+str(row_idx)+"panoptic"+str(panoptic_idx)+"_idx"+str(obj_idx))

                        print("SAM2 didn't pass")
                        continue


                    # Select the region box (useful_region_box are those guarantee has no overlap between ID and the first frame)
                    useful_region_box.sort(key = lambda x: x[0])
                    useful_region_box = useful_region_box[ -1 * store_useful_region_box_num : ]      # Choose the largest area
                    # useful_region_box = random.sample(useful_region_box, min(store_useful_region_box_num, len(useful_region_box)))        # Random Sample



                    # Prune tracking points: Store those with valid point status (aligned with SAM2 mask temporally for all points) for obj_Track_Traj and obj_Track_Visibility
                    aligned_obj_Track_Traj = [[] for _ in range(len(obj_Track_Traj))]
                    aligned_obj_Track_Visibility = [[] for _ in range(len(obj_Track_Visibility))]
                    # Iterate all points
                    for point_idx, point_status in enumerate(point_valid_status):       
                        if point_status:     # True means ideal; False case will be skipped
                            # Add all points temporally uniformly
                            for temporal_idx in range(len(obj_Track_Traj)):
                                aligned_obj_Track_Traj[temporal_idx].append(obj_Track_Traj[temporal_idx][point_idx])
                                aligned_obj_Track_Visibility[temporal_idx].append(obj_Track_Visibility[temporal_idx][point_idx])


                    # Visualize for Success Case (Should Comment Out)
                    # sample_region_box = random.choice(useful_region_box)
                    # visualize_motion(aligned_obj_Track_Traj, single_obj_info, aligned_obj_Track_Visibility, height, width, video_np, 6, 
                    #                     max_step_num, sample_region_box, name="FrameIn_Success_Aligned_row"+str(row_idx)+"panoptic"+str(panoptic_idx)+"_idx"+str(obj_idx))


                    ##########################################################################################################################################################################


                    # Append information PER Object
                    panoptic_ID_info[-1] = [segmentation_info, useful_region_box]       # NOTE: we already append earlier
                    Track_Traj_raw[panoptic_idx][obj_idx] = aligned_obj_Track_Traj              # Update with the SAM2 aligned results
                    Track_Visibility_raw[panoptic_idx][obj_idx] = aligned_obj_Track_Visibility

                    # Update
                    effective_obj_num += 1


                # Check if there is any effective panoptic frame
                if effective_obj_num != 0:  
                    ID_info_all.append(panoptic_ID_info)
                    Obj_info_all.append(panoptic_Obj_info)
                    Track_Traj_all.append(Track_Traj_raw[panoptic_idx])
                    Track_Visibility_all.append(Track_Visibility_raw[panoptic_idx])


            # Store all the region box that is ideal
            if len(ID_info_all) != 0:       # At least one panoptic frame should have the information

                # Dump the old information with the new information
                ID_info_all = json.dumps(ID_info_all)           # New Column
                row[elements["Obj_Info"]] = json.dumps(Obj_info_all)        # Old column
                row[elements["Track_Traj"]] = json.dumps(Track_Traj_all)            # Old column
                row[elements["Track_Visibility"]] = json.dumps(Track_Visibility_all)        # Old column 


                # Append new cases of the ID_info_all
                info_lists.append(row + [ID_info_all])  


        # Final Log update
        with open(store_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(info_lists)



            
if __name__ == "__main__": 

    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_offset', type=int, default=0)
    parser.add_argument('--debug', type=bool, default = False)
    args = parser.parse_args()


    # Fundamental Setting
    csv_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_TrackMotion_left"        # Input
    store_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_FrameINO"              # Output
    visual_store_parent_folder = "../General_ID_INO"                                # Output for ID images
    GPU_offset = args.GPU_offset
    resume = False


    # Region Box Setting
    unbounded_mask_box_top_left_position = (0.55, 0.55)                                     # Max Top Left Coord position [horizontal, vertical]
    fixed_aspect_ratios = [16.0/9.0, 3.0/2.0, 4.0/3.0, 5.0/4.0, 1.0/1.0, 4.0/5.0]           # Width:Height Aspect Ratio
    fixed_aspect_ratio_prob = [0.35, 0.3, 0.2, 0.13, 0.01, 0.01]                            # Corresponding to previous row   (发现不能让后面的prob太高，因为越后面success rate越高)
    unbounded_mask_box_scale_min = [0.6, 0.6, 0.65, 0.65, 0.75, 0.85]                       # This setting set the minimum height value
    max_box_find_times = 2000                                                               # Number in the initial search
    store_useful_region_box_num = 15                                                        # Maximum number of Region Box to keep
    outside_offset = 15                                                                     # We want the Region box selected to some distance to the tracking points (一种dilation机制)


    # SAM Setting
    # SAM_video_max_sec = 8               # -1 refer to all is used; Unit: seconds; Already start from panoptic frame IDX, should be long enough for 10s
    SAM_min_reference_store_num = 3         # This refers to other auxiliary SAM information
    SAM_min_area_ratio = 0.1            # The ID Reference in the first frame must be large enough, avoid somethere that is too small
    max_skip_ratio_allowed = 0.33       # This is unmatch point ratio of between SAM2 results and the Tracking Results           



    # Prepare the folder
    if not os.path.exists(store_folder_path):
        os.makedirs(store_folder_path)
    if not os.path.exists(visual_store_parent_folder):      # Don't directly remove all
        os.makedirs(visual_store_parent_folder)
    


    # Single Process based on the GPU offset
    single_process(csv_folder_path, store_folder_path, GPU_offset, debug=args.debug)


    print("Finished!")