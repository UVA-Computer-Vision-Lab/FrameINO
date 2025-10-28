'''
    Evaluate the metrics
'''
import os, sys, shutil
import random
import cv2
import pickle
import math
import imageio
import time
import numpy as np
from PIL import Image, ImageDraw
import gc
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
import torchvision.transforms as transforms
# from sam2.sam2_video_predictor import SAM2VideoPredictor 





def INO_VLM_evaluation(data_parent_path, region_target_height, region_target_width, is_frame_in=True):
    '''
        Args:
            region_target_height (int):     Height without the padding, the padding will be scaled together
            region_target_width (int):      Width without the padding, the padding will be scaled together
            test_num_frames (int):          Sampled number of frames from the GT and GEN generated results
            is_frame_in (bool):             Whether frame_in or frame out
    '''


    # Prepare the pretrained weight
    vlm_model_path = "Qwen/Qwen2.5-VL-32B-Instruct"


    # Prepare the VLM Setting
    device = "cuda"
    llm_fps = 1 
    test_num_frames = 14            # Override it, Qwen cannot take in too many frames; 14 is a moderate value


    # Define the Instruction Prompt
    if not is_frame_in:     # Frame Out Instruction Prompt
        instruction_prompt = "Please check if the object leave the frame. Return a Yes/No as the only response."
    else:   # Frame In
        instruction_prompt = "Please check if the object enter the frame. Return a Yes/No as the only response."


    # Messages containing a local video path and a text query
    messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                # "video": video_path,
                                "max_pixels": 360 * 420,            # The video information here should be deprecated
                                "fps": llm_fps,
                            },
                            {
                                "type": "text", 
                                "text": instruction_prompt,
                            },
                        ],
                    }
                ]



    # Init the QWen VLM Model
    processor = AutoProcessor.from_pretrained(vlm_model_path)
    bnb_config = BitsAndBytesConfig(
                                        load_in_4bit=True,
                                        bnb_4bit_compute_dtype=torch.float16,  # Use float16 for computations
                                        bnb_4bit_use_double_quant=True,
                                        bnb_4bit_quant_type='nf4',  # NormalFloat4 quantization
                                    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                                                                vlm_model_path, 
                                                                torch_dtype="auto", device_map="auto",
                                                                quantization_config=bnb_config,
                                                            )

    # In Qwen 2.5 VL, frame rate information is also input into the model to align with absolute time.
    llm_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)



    # Calculate the MAX nubmer of frames in video folder
    total_gen_num_frames_one_video, total_gt_num_frames_one_video = 0, 0
    for file_name in sorted(os.listdir(os.path.join(data_parent_path, "instance0"))):
        if file_name.find("gen_frame") != - 1:
            total_gen_num_frames_one_video += 1
    
        if file_name.find("gt_frame") != - 1:
            total_gt_num_frames_one_video += 1
    print("We have total gen and gt number in one video of frames of ", total_gen_num_frames_one_video, total_gt_num_frames_one_video)

    
    # Get the index
    gen_indices = np.linspace(0, total_gen_num_frames_one_video - 1, test_num_frames, dtype=int)     
    gt_indices = np.linspace(0, total_gt_num_frames_one_video - 1, test_num_frames, dtype=int)
    assert(len(gen_indices) == test_num_frames)
    assert(len(gt_indices) == test_num_frames)
    



    # Iterate each sub folder
    all_video_score = []
    start_time = time.time()
    for instance_idx in range(len(os.listdir(data_parent_path))):
        sub_folder_path = os.path.join(data_parent_path, "instance"+str(instance_idx))


        # Read the Important information from processed_meta_data store inside the folder
        processed_meta_data_store_path = os.path.join(sub_folder_path, "processed_meta_data.pkl")
        assert(os.path.exists(processed_meta_data_store_path))
        with open(processed_meta_data_store_path, 'rb') as file:
            processed_meta_data = pickle.load(file)

        # Fetch information
        # GT_track_traj = processed_meta_data["full_pred_tracks"]
        original_height = int(processed_meta_data["original_height"])
        original_width = int(processed_meta_data["original_width"])
        resized_mask_region_box = processed_meta_data["resized_mask_region_box"]
        
        # Read sample
        sample_GT_frame = cv2.imread(os.path.join(sub_folder_path, "gt_padded_frame0.png"))
        canvas_height, canvas_width, _  = sample_GT_frame.shape 

        # NOTE: Define the new size, which is based on the region box to the target width and height
        (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) = resized_mask_region_box        # already resized
        region_height = bottom_right_y - top_left_y
        region_width = bottom_right_x - top_left_x
        scale_w = region_target_width / region_width
        scale_h = region_target_height / region_height
        scaled_canvas_width = int(canvas_width * scale_w)
        scaled_canvas_height = int(canvas_height * scale_h)

        # Calculate the MAX nubmer of frames in video folder
        total_gen_num_frames_one_video, total_gt_num_frames_one_video = 0, 0
        for file_name in sorted(os.listdir(os.path.join(data_parent_path, "instance0"))):
            if file_name.find("gen_frame") != - 1:
                total_gen_num_frames_one_video += 1
        
            if file_name.find("gt_frame") != - 1:
                total_gt_num_frames_one_video += 1
        print("We have total gen and gt number in one video of frames of ", total_gen_num_frames_one_video, total_gt_num_frames_one_video)

        
        # Get the index
        gen_indices = np.linspace(0, total_gen_num_frames_one_video - 1, test_num_frames, dtype=int)     
        gt_indices = np.linspace(0, total_gt_num_frames_one_video - 1, test_num_frames, dtype=int)
        assert(len(gen_indices) == test_num_frames)
        assert(len(gt_indices) == test_num_frames)



        # Collect Gen frames and GT frames
        gen_frames, gt_frames = [], []
        for order_idx in range(test_num_frames):

            # Read the path
            gen_frame_path = os.path.join(sub_folder_path, "gen_frame"+str(gen_indices[order_idx])+".png")       # This will be arbitrary resolution and needs to be resized
            assert(os.path.exists(gen_frame_path))
            gt_frame_path = os.path.join(sub_folder_path, "gt_frame"+str(gt_indices[order_idx])+".png")       # This will be arbitrary resolution and needs to be resized
            assert(os.path.exists(gt_frame_path))

            # Read and Transforms
            gen_frame = cv2.cvtColor(cv2.imread(gen_frame_path), cv2.COLOR_RGB2BGR)
            gt_frame = cv2.cvtColor(cv2.imread(gt_frame_path), cv2.COLOR_RGB2BGR)

            # Resize
            gen_frame = cv2.resize(gen_frame, (region_target_width, region_target_height))        # The height and width for tracking, all methods is uniformed on the same
            gt_frame = cv2.resize(gt_frame, (region_target_width, region_target_height))

            # Updates
            gen_frames.append(gen_frame)
            gt_frames.append(gt_frame)

        # Convert to tensor
        gen_tensor = torch.tensor(gen_frames).to(device)
        gt_tensor = torch.tensor(gt_frames).to(device)
        # Permute convert
        gen_video_inputs = [gen_tensor.to(torch.uint8).permute(0, 3, 1, 2)]
        gt_video_inputs = [gt_tensor.to(torch.uint8).permute(0, 3, 1, 2)]


        
        ################################################### Process Gen video ####################################################################

        # Preproces
        gen_inputs = processor(
                                    text = [llm_text],
                                    images = None,              # NOTE: This is not needed when there is video inputs
                                    videos = gen_video_inputs,
                                    fps = llm_fps,
                                    padding = True,
                                    return_dict = True,
                                    return_tensors = "pt"
                                )
        second_per_grid_ts = gen_inputs.pop('second_per_grid_ts')
        second_per_grid_ts = [float(s) for s in second_per_grid_ts]
        gen_inputs.update({
                            'second_per_grid_ts': second_per_grid_ts
                        })
        gen_inputs = gen_inputs.to("cuda")
        # breakpoint()



        # Generate
        generated_ids = model.generate(**gen_inputs, max_new_tokens=128)
        generated_ids_trimmed = [
                                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(gen_inputs.input_ids, generated_ids)
                                ]
                                
        gen_output_text = processor.batch_decode(
                                                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                                                )[0]

        gen_output_text = gen_output_text.strip().lower()
        print("Gen Output text is ", gen_output_text)

        ####################################################################################################################################



        ################################################### Process GT video #####################################################################
        
        # Vision Processor
        gt_inputs = processor(
                                text = [llm_text],
                                images = None,              # NOTE: This is not needed when there is video inputs
                                videos = gt_video_inputs,
                                fps = llm_fps,
                                padding = True,
                                return_dict = True,
                                return_tensors = "pt"
                            )
        second_per_grid_ts = gt_inputs.pop('second_per_grid_ts')
        second_per_grid_ts = [float(s) for s in second_per_grid_ts]
        gt_inputs.update({
                            'second_per_grid_ts': second_per_grid_ts
                        })
        gt_inputs = gt_inputs.to("cuda")



        # Generate
        generated_ids = model.generate(**gt_inputs, max_new_tokens=128)
        generated_ids_trimmed = [
                                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(gt_inputs.input_ids, generated_ids)
                                ]
                                
        gt_output_text = processor.batch_decode(
                                                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                                                )[0]
        
        gt_output_text = gt_output_text.strip().lower()
        print("GT Output text is ", gt_output_text)

        ##################################################################################################################################

        

        # Justify the Solution
        if gt_output_text not in ["yes", "no"]:
            print("One invalid GT case for ", instance_idx, " which write ", gt_output_text)
            continue
        

        if gt_output_text == gen_output_text:
            score = 1
        else:
            score = 0
        

        # Per Video Updates
        all_video_score.append(score)

        print("Finish Instance ", instance_idx, " with score ", score)
        print("Time spent is ", (time.time() - start_time)/60)
        print()

    print("Effective length is ", len(all_video_score))
    average_score = sum(all_video_score) / len(all_video_score)
    return average_score

