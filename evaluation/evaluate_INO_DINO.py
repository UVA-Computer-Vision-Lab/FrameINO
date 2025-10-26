'''
    Evaluate the metrics
'''
import os, sys, shutil
import random
import cv2
import numpy as np
from PIL import Image
import gc
import pickle
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
import torch.nn.functional as F
import torchvision.transforms as transforms
# from sam2.sam2_video_predictor import SAM2VideoPredictor 



def dino_transform_Image(n_px):
    return Compose([
        Resize(size=n_px, antialias=False),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])



def INO_DINO_evaluation(data_parent_path, target_height, target_width, test_num_frames=None):
    '''
        Args:
            num_frames (int): If this is none, it means that we need find gen frames num by ourselves
            test_num_frames (int): how many frames to be selected in the generated results 
    '''

    # Init the Dino model
    device = "cuda"
    dinov2_dict = {
                    'repo_or_dir': f'facebookresearch/dinov2',
                    'model': 'dinov2_vitb14',
                }
    dinov2_model = torch.hub.load(**dinov2_dict).to(device)



    # Image transforms
    image_transform = dino_transform_Image(224)


    # Calculate the MAX nubmer of frames in video folder
    total_gen_num_frames_one_video, total_gt_num_frames_one_video = 0, 0
    for file_name in sorted(os.listdir(os.path.join(data_parent_path, "instance0"))):
        if file_name.find("gen_frame") != - 1:
            total_gen_num_frames_one_video += 1
    
        if file_name.find("gt_frame") != - 1:
            total_gt_num_frames_one_video += 1
    print("We have total gen and gt number in one video of frames of ", total_gen_num_frames_one_video, total_gt_num_frames_one_video)




    # Iterate each sub folder
    all_video_score = []
    for instance_idx in range(len(os.listdir(data_parent_path))):
        sub_folder_path = os.path.join(data_parent_path, "instance"+str(instance_idx))

        # Read the main reference img
        reference_img_path = os.path.join(sub_folder_path, "Main_Reference.png")
        assert(os.path.exists(reference_img_path))

        # Resize & Transform
        reference_img = Image.open(reference_img_path)
        reference_img = reference_img.resize((target_width, target_height))
        reference_img = image_transform(reference_img)


        # Convert
        reference_img = reference_img.unsqueeze(0)
        reference_img = reference_img.to(device)

        # Input DINO V2
        reference_image_features = dinov2_model(reference_img)
        reference_image_features = F.normalize(reference_image_features, dim=-1, p=2)


        # Unifromly sample the frame 
        gt_index_order = np.linspace(0, total_gt_num_frames_one_video - 1, test_num_frames, dtype=int)
        gen_index_order = np.linspace(0, total_gen_num_frames_one_video - 1, test_num_frames, dtype=int)

        print("Selected Index order is ", gt_index_order, gen_index_order)

        # # Choose the index based on the tracking results (whether the object is inside the Region Frame)
        # processed_meta_data_store_path = os.path.join(sub_folder_path, "processed_meta_data.pkl")
        # assert(os.path.exists(processed_meta_data_store_path))
        # with open(processed_meta_data_store_path, 'rb') as file:
        #     processed_meta_data = pickle.load(file)

        # # Fetch
        # resized_mask_region_box = processed_meta_data["resized_mask_region_box"]
        # full_pred_tracks = processed_meta_data["full_pred_tracks"]
        # (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) = resized_mask_region_box        # already resized

        # # Check which frame index has object inside the BBox, and then do the 
        # index_order = []
        # for frame_idx, per_frame_track in enumerate(full_pred_tracks):
        #     breakpoint()



        ########################################### Generate Videos Process #######################################################

        gen_similarity_lists = []
        for frame_idx in gen_index_order:

            # Read the path
            gen_frame_path = os.path.join(sub_folder_path, "gen_frame"+str(frame_idx)+".png")
            assert(os.path.exists(gen_frame_path))


            # Read and Transforms
            gen_frame = Image.open(gen_frame_path)
            gen_frame = gen_frame.resize((target_width, target_height))
            gen_frame = image_transform(gen_frame)


            # Evaluate
            with torch.no_grad():

                # Get the Dino feature
                gen_image = gen_frame.unsqueeze(0)
                gen_image = gen_image.to(device)
                gen_image_features = dinov2_model(gen_image)
                gen_image_features = F.normalize(gen_image_features, dim=-1, p=2)

                
                # Calcualte the Cosine similarity
                gen_similarity_score = max(0.0, F.cosine_similarity(reference_image_features, gen_image_features).item())


                # Update
                gen_similarity_lists.append(gen_similarity_score)

        ########################################################################################################################



        ########################################### Ground Truth Process #######################################################

        gt_similarity_lists = []
        for frame_idx in gt_index_order:
            
            # Read file
            gt_frame_path = os.path.join(sub_folder_path, "gt_frame"+str(frame_idx)+".png")
            assert(os.path.exists(gt_frame_path))


            # Read and Transforms
            gt_frame = Image.open(gt_frame_path)
            gt_frame = gt_frame.resize((target_width, target_height))
            gt_frame = image_transform(gt_frame)


            # Evaluate
            with torch.no_grad():

                # Get the Dino feature
                gt_image = gt_frame.unsqueeze(0)
                gt_image = gt_image.to(device)
                gt_image_features = dinov2_model(gt_image)
                gt_image_features = F.normalize(gt_image_features, dim=-1, p=2)

                # Calcualte the Cosine similarity
                gt_similarity_score = max(0.0, F.cosine_similarity(reference_image_features, gt_image_features).item())

                # Update
                gt_similarity_lists.append(gt_similarity_score)
        
        ########################################################################################################################

        # Sort
        # similarity_lists = sorted(similarity_lists)[-1 * selected_num:]


        # Per Video Updates
        assert(len(gen_similarity_lists) == len(gt_similarity_lists))
        gen_per_video_similarity = sum(gen_similarity_lists) / len(gen_similarity_lists)
        gt_per_video_similarity = sum(gt_similarity_lists) / len(gt_similarity_lists)
        
        if gt_per_video_similarity != 0:
            relative_distance = abs(gen_per_video_similarity - gt_per_video_similarity) / gt_per_video_similarity
        else:
            print("This video has no GT similarity")
            continue


        print("per video DINO similarity is ", relative_distance)
        all_video_score.append(relative_distance)

    average_score = sum(all_video_score) / len(all_video_score)
    return average_score


