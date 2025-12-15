'''
    Video Caption by Qwen-VL 32B/72B; Aim for multiple section in 
'''

import os, sys, shutil
import csv
import gc
import time
import argparse
import signal
import ffmpeg
import json
import numpy as np
import math
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
csv.field_size_limit(sys.maxsize)       # Default setting is 131072, 10x expand should be enough
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"



# Handler function that raises TimeoutError
def timeout_handler(signum, frame):
    raise TimeoutError("Time exceeded for function execution")



def single_process(input_csv_folder_path, store_csv_folder_path, GPU_offset):
    

    # Setting
    store_freq = 10
    device = 'cuda'



    # Read the csv file
    csv_idx = GPU_offset
    csv_file_path = os.path.join(input_csv_folder_path, "sub" + str(csv_idx) + ".csv")
    print("CSV file we read is ", csv_file_path)


    # Prepare the store file path
    store_file_path = os.path.join(store_csv_folder_path, "sub" + str(csv_idx) + ".csv")
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
            

    # Init the model
    processor = AutoProcessor.from_pretrained(model_path)
    # bnb_config = BitsAndBytesConfig(
    #                                     load_in_4bit=True,
    #                                     bnb_4bit_compute_dtype=torch.float16,  # Use float16 for computations
    #                                     bnb_4bit_use_double_quant=True,
    #                                     bnb_4bit_quant_type='nf4',  # NormalFloat4 quantization
    #                                 )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                                                                    model_path, torch_dtype="auto", device_map="auto",
                                                                    # quantization_config=bnb_config,
                                                                )
    

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

                
                new_addition_content = [store_prompt_name]
                print("The first row is ", row + new_addition_content)

                # Store the first row to csv
                if not resume:
                    with open(store_file_path, 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        writer.writerows([row + new_addition_content])
                continue


            # Read important information
            video_path = row[elements["video_path"]]
            valid_duration = json.loads(row[elements["valid_duration"]])
            Panoptic_Info_all = json.loads(row[elements["Panoptic_Segmentation"]])
            

            # Resume mode will execute until we have the last store row matched
            if resume:
                if video_path == last_store_row[elements["video_path"]] and valid_duration == json.loads(last_store_row[elements["valid_duration"]]):     # Check Video Path and the valid duration
                    print("We find resume at", row_idx)        # In caption, this row should match len(store_rows).  
                    find_resume = True      # We find the exact row we want
                    continue        # Should continue; else, we repeat the same one again.

            if not find_resume:
                continue

            
            try:

                # Messages containing a local video path and a text query
                messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "video",
                                            "video": video_path,
                                            "max_pixels": 256 * 384,            # The video information here should be deprecated
                                        },
                                        {
                                            "type": "text", 
                                            "text": instruction_prompt,
                                        },
                                    ],
                                }
                            ]


                # Read the video by ffmpeg, not decod
                resolution = str(target_width) + "x" + str(target_height)
                video_stream, err = ffmpeg.input(
                                                    video_path
                                                ).output(
                                                    "pipe:", format = "rawvideo", pix_fmt = "rgb24", s = resolution, vsync = 'passthrough',
                                                ).run(
                                                    capture_stdout = True, capture_stderr = True    # If there is bug, command capture_stderr
                                                )    # The resize is already included
                video_full_np = np.frombuffer(video_stream, np.uint8).reshape(-1, target_height, target_width, 3)
                
                # Fetch the valid duration
                video_np = video_full_np[valid_duration[0] : valid_duration[1]]
                video_tensor = torch.tensor(video_np).to(device)
                num_frames = len(video_np)


                output_text_all = []
                for (panoptic_start_frame_idx, _) in Panoptic_Info_all:

                    # NOTE: panoptic_start_frame_idx这个应该是根据valid curation crop以后开始算的，所以这里的0就是crop以后的0

                    # Define the Start End range 
                    end_frame_idx = min(num_frames, panoptic_start_frame_idx + max_frames_consider)


                    # Crop the video to the needed duration
                    crop_video_inputs = [video_tensor[panoptic_start_frame_idx : end_frame_idx : sample_frame_freq].permute(0, 3, 1, 2)]
                    print("Number of frames process is ", len(crop_video_inputs[0]))



                    # In Qwen 2.5 VL, frame rate information is also input into the model to align with absolute time.
                    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    # image_inputs, video_inputs = process_vision_info(messages)        # HACK: deprecated, memory leak occurs


                    # Final Pre-Process
                    inputs = processor(
                                            text = [text],
                                            images = None,              # NOTE: This is not needed when there is video inputs
                                            videos = crop_video_inputs,
                                            padding = True,
                                            return_tensors = "pt"
                                        )
                    second_per_grid_ts = inputs.pop('second_per_grid_ts')
                    second_per_grid_ts = [float(s) for s in second_per_grid_ts]
                    inputs.update({
                                        'second_per_grid_ts': second_per_grid_ts
                                    })
                    inputs = inputs.to("cuda")


                    # Generate
                    generated_ids = model.generate(**inputs, max_new_tokens=128)
                    generated_ids_trimmed = [
                                                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                                            ]
                    output_text = processor.batch_decode(
                                                            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                                                        )[0]
                    print("Output text is ", output_text, " for the video ", video_path, " of the range " + str(panoptic_start_frame_idx) + "-" + str(end_frame_idx) + " for valid duration of ", valid_duration, "\n")


                    # Append to the list
                    output_text_all.append(output_text)


                # Update the text prompt
                info_lists.append(row + [json.dumps(output_text_all)])
                print("Finished Instance", str(row_idx), "\n")


                # Clean cache
                gc.collect()


                # Log update (The update will be quite random for this file, because we may skip earlier on)
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


            except Exception as error:
                print("There is exception case", error)
                continue   


        # Final Log update
        with open(store_file_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(info_lists)




if __name__ == "__main__":

    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_offset', type=int, default=0)
    args = parser.parse_args()


    # Model and inputs outputs Setting       
    model_path = "Qwen/Qwen2.5-VL-32B-Instruct"      # Qwen2.5-VL-7B-Instruct  Qwen2.5-VL-72B-Instruct.  It seems that 32B is newer and competitive compared to 72B version
    input_csv_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_camera_estimation_left"                  # Input 
    store_csv_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_caption"          # Output
    GPU_offset = args.GPU_offset
    resume = True


    # Video Processing Setting
    store_prompt_name = "Structured_Text_Prompt"
    target_height = 256
    target_width = 384
    max_frames_consider = 160             # About 81 * 2
    sample_frame_freq = 16                # 原来是1fps，大改就是24个step; 目前更加dense一点的吧，设置16


    # Instruction Prompt
    # instruction_prompt = "Please describe the video in 30 words. \
    #                         Only describe the temporal change of the video provided without describing the spatial information in the first frame provided.\
    #                         Only show the information with the highest confidence. Don't use any words like gesture, gesturing."     # Don't use any words like gesture, gesturing.

    # Structured Instruction Prompt
    instruction_prompt = "Please describe the video in 20 to 40 words. Describe in the content of style, abstract, and then detailed description. The response should be a paragraph of natural language for diffusion model text prompt, and don't be too obvious in noting the structure like using 'The video captures ...', 'In the scene', 'abstractly' or 'detailedly'. Prefer temporal change of the video rather than the spatial information in the first frame provided. Don't use any words like gesture, gesturing. Only show the information with high confidence."


    if not os.path.exists(store_csv_folder_path):
        os.makedirs(store_csv_folder_path)


    # Inferece Process
    single_process(input_csv_folder_path, store_csv_folder_path, GPU_offset)


    print("Finished!")


