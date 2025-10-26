'''
    This file is designed for Frame Out mass evaluation dataset running. This will use the older dataloader, which is slightly different from the first one.
'''
import os, sys, shutil
import csv
import numpy as np
import ffmpeg
import cv2
import time
import collections
import json
import pickle
import random
import imageio
import PIL

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import torch
from torch.utils.data import DataLoader, Dataset
from diffusers import AutoencoderKLCogVideoX
from transformers import T5EncoderModel
from omegaconf import OmegaConf
from diffusers.utils import export_to_video, load_image


# Import files from the local fodler
root_path = os.path.abspath('.')
sys.path.append(root_path)
from pipelines.pipeline_cogvideox_i2v_motion_FrameINO import CogVideoXImageToVideoPipeline
from architecture.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from data_loader.video_dataset_motion_FrameINO_old import VideoDataset_Motion_FrameINO
# from evaluation.mass_evaluation import mass_evaluation



if __name__ == "__main__":

    # Frequently Changed Setting
    base_model_id = "../pretrained/CogVideoX_5B_I2V"                                    # Set the pretrained CogVideoX
    transformer_ckpt_path = "uva-cv-lab/FrameINO_CogVideoX_Stage2_MotionINO_v1.0"       # Path of the model weights
    download_folder_path = "../FrameINO_data"                                           # Set the downloaded folder path, all the other csv will be read automatically


    # Output Dir  
    store_parent_folder_name = "results_FrameOut"       # Results Store path


    # Basic Inference Setting
    num_inference_steps = 50        # Default setting was 50
    preset_height = 448             # Canvas Resolution
    preset_width = 640              
    preset_num_frames = 49          # Must be 49 for CogVideoX
    num_test = 200                  # Max setting 200 (actually the evaluation dataset is less than this number)
    one_point_one_obj = False       # Whether we only use one point as traj for each obj; False means using all points (based on point_keep_ratio_regular Setting)

    

    # Prepare
    test_csv_folder_path = os.path.join(download_folder_path, "dataset_csv_files/evaluation_formal_perfect_openvid_FrameOut")
    assert(os.path.exists(test_csv_folder_path))
    dataset_folder_path = os.path.join(download_folder_path, "video_dataset/evaluation_FrameOut")
    assert(os.path.exists(dataset_folder_path))


    # Replace information (Especially the height and width information, if needed)
    config = {}     # NOTE: It was copy config available, but for this older one, we directly write a dict and then check 
    config['dataset_folder_path'] = dataset_folder_path
    config['height'] = preset_height
    config['width'] = preset_width
    config['train_frame_num'] = preset_num_frames
    config["preset_decode_fps"] = 12
    config["empty_text_prompt"] = False
    config["start_skip"] = 0
    config["end_skip"] = 0
    config["dot_radius"] = 6
    config["drop_FrameIn_prob"] = 1.0               # FrameOut don't use this, so this is just a placeholder
    config["text_mask_ratio"] = 0.0                 # Must have a text prompt
    config["faster_motion_prob"] = 0.0              # No fast running
    config["point_keep_ratio_ID"] = 1.0             # FrameOut has no valid ID reference, so this one should is just a placeholder
    config["point_keep_ratio_regular"] = 0.5        # Will be ineffective when one_point_one_obj is True; Recommend around 0.5, the training is also around 0.5
    

    # Prepare the Folder
    if os.path.exists(store_parent_folder_name):
        shutil.rmtree(store_parent_folder_name)
    os.makedirs(store_parent_folder_name)


    ###################################################### Load the Model Unit ###########################################################

    print("Start Loading the Model!")
    transformer = CogVideoXTransformer3DModel.from_pretrained(transformer_ckpt_path, torch_dtype=torch.float16)
    text_encoder = T5EncoderModel.from_pretrained(base_model_id, subfolder="text_encoder", torch_dtype=torch.float16)
    vae = AutoencoderKLCogVideoX.from_pretrained(base_model_id, subfolder="vae", torch_dtype=torch.float16)
    vae.enable_slicing()
    vae.enable_tiling()


    # Create pipeline and run inference
    print("Start Loading the Pipeline!")
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                base_model_id,
                text_encoder = text_encoder,
                transformer = transformer,
                vae = vae,
                torch_dtype = torch.float16,
            )
    pipe.enable_model_cpu_offload()

    ######################################################################################################################################


    # Prepare the Test Dataloader
    test_dataset = VideoDataset_Motion_FrameINO(
                                                    config,
                                                    test_csv_folder_path,
                                                    FrameOut_only = True,
                                                    one_point_one_obj = one_point_one_obj,           # This Refers that only one point is used
                                                    strict_validation_match = True,
                                                )
    test_dataloader = DataLoader(
                                    test_dataset,
                                    batch_size = 1,
                                    shuffle = False,
                                    num_workers = 1,
                                )   # No need for the collate


    # Iterate the test dataloader
    start_time = time.time()
    print("Start Evaluation!\n")
    for instance_idx, batch in enumerate(test_dataloader):

        if instance_idx >= num_test:
            break
        print("This is instance ", str(instance_idx))
        

        # Prepare the folder
        store_folder_path = os.path.join(store_parent_folder_name, "instance" + str(instance_idx))
        if os.path.exists(store_folder_path):
            shutil.rmtree(store_folder_path)
        os.makedirs(store_folder_path)


        # Fetch and Only Read the first batch size
        first_frame_np = np.array(batch["first_frame_np"][0], dtype=np.uint8).copy()
        text_prompt = batch["text_prompt"][0]
        traj_tensor = batch["traj_tensor"][0]   
        traj_imgs_np = np.array(batch["traj_imgs_np"][0], dtype=np.uint8).copy() 
        main_reference_tensor = batch["main_reference_tensor"][0]
        main_reference_np = np.asarray(batch["main_reference_np"][0])
        video_gt_np = np.asarray(batch["video_gt_np"][0])
        merge_frames = batch["merge_frames"][0]
        processed_meta_data = batch["processed_meta_data"]



        # Prepare
        (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) = processed_meta_data["resized_mask_region_box"]


        # Save the GT 
        assert(video_gt_np.shape[0] == preset_num_frames)
        imageio.mimsave(os.path.join(store_folder_path, "gt_video.mp4"), video_gt_np, fps=12)
        for frame_idx, gt_img_frame in enumerate(video_gt_np):

            # Save the Padded frame
            save_path = os.path.join(store_folder_path, "gt_padded_frame"+str(frame_idx)+".png")
            cv2.imwrite(save_path, cv2.cvtColor(gt_img_frame, cv2.COLOR_RGB2BGR))

            # Extract Unpadded version
            cropped_region_frame = gt_img_frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            # Save the region frame (crop the padding)
            save_path = os.path.join(store_folder_path, "gt_frame"+str(frame_idx)+".png")
            cv2.imwrite(save_path, cv2.cvtColor(cropped_region_frame, cv2.COLOR_RGB2BGR))
            
        print("Region Box size is ", cropped_region_frame.shape)

        
        # Convert first frame np to image form
        first_frame_path = os.path.join(store_folder_path, "first_frame.png")
        cv2.imwrite(first_frame_path, cv2.cvtColor(first_frame_np, cv2.COLOR_RGB2BGR))


        # Save the traj images
        # for traj_idx, traj_img_np in enumerate(traj_imgs_np):
        #     traj_img_store_path = os.path.join(store_folder_path, "traj_cond" + str(traj_idx) + ".png")
        #     cv2.imwrite(traj_img_store_path, cv2.cvtColor(traj_img_np, cv2.COLOR_RGB2BGR))
        # Save traj Video
        imageio.mimsave(os.path.join(store_folder_path, "traj_video.mp4"), traj_imgs_np, fps=12)


        # Save the ID Reference Img
        reference_img_path = os.path.join(store_folder_path, "Main_Reference.png")
        cv2.imwrite(reference_img_path, cv2.cvtColor(main_reference_np, cv2.COLOR_RGB2BGR))

        # Save the merge video
        imageio.mimsave(os.path.join(store_folder_path, "merge_cond.mp4"), merge_frames, fps=12)

        # Save the text prompt
        with open(os.path.join(store_folder_path, 'text_prompt.txt'), 'w') as file:
            file.write(text_prompt)

        # Save the Important Meta Data (like GT tracks)
        meta_data_store_path = os.path.join(store_folder_path, "processed_meta_data.pkl")
        with open(meta_data_store_path, 'wb') as f:
            pickle.dump(processed_meta_data, f)


        # Call the Inference Pipeline
        image = load_image(first_frame_path)
        video = pipe(
                        image = image, 
                        prompt = text_prompt, 
                        traj_tensor = traj_tensor, 
                        ID_tensor = main_reference_tensor,
                        height = preset_height, width = preset_width,
                        guidance_scale = 6, use_dynamic_cfg = True, num_inference_steps = 50        # Make this general setting fixed
                    ).frames[0]     


        # Store the reuslt
        export_to_video(video, os.path.join(store_folder_path, "generated_video_padded.mp4"), fps=8)


        # Write as Frames for Generated Video (Padded version)
        gen_frame_cropped = []
        for frame_idx, frame_pil in enumerate(video):
            save_path = os.path.join(store_folder_path, "gen_padded_frame"+str(frame_idx)+".png")
            frame_pil.save(save_path)

            # Extract Unpadded version
            frame_np = np.asarray(frame_pil)        # Should be RGB?
            cropped_region_frame = frame_np[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            # Save the region frame (crop the padding)
            save_path = os.path.join(store_folder_path, "gen_frame"+str(frame_idx)+".png")
            cv2.imwrite(save_path, cv2.cvtColor(cropped_region_frame, cv2.COLOR_RGB2BGR))
            gen_frame_cropped.append(cropped_region_frame)

        # Write the generated video (No Outside Region)
        imageio.mimsave(os.path.join(store_folder_path, "generated_video.mp4"), gen_frame_cropped, fps=12)


        # End Log
        print("Finished Processing instance", instance_idx, "!\n")



    full_time_spent = int(time.time() - start_time)
    print("Time spent for Inference is %d min %d s" %(full_time_spent//60, full_time_spent%60))



