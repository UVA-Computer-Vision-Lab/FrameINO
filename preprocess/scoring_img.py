'''
    Estimate the image level quality of the video by image quality assessment (IQA), ICA, Text Volume, and Aesthetic.
    Will create a **temporary folder** (tmp) and a pretrained weight folder
'''

import os, sys, shutil
import pandas as pd
import time
import csv
import collections
from multiprocessing import Process
import multiprocessing
import cv2
import ffmpeg
import numpy as np
from torchvision import transforms
import random
from PIL import Image
import argparse
import torch
import json
csv.field_size_limit(sys.maxsize)


# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)



def polygon_area(coordinates):
    n = len(coordinates)
    area = 0
    for i in range(n):
        j = (i + 1) % n
        area += coordinates[i][0] * coordinates[j][1]
        area -= coordinates[j][0] * coordinates[i][1]
    return abs(area) / 2



@torch.no_grad
def single_process( csv_folder_path,
                    store_folder_path,
                    GPU_offset,
                    samples_on_video,
                    scoring_criteria,
                    text_area_crop
                ):

    # Setting
    debug = False        # Will store 
    store_freq = 10


    # Read the csv file
    csv_file_path = os.path.join(csv_folder_path, "sub" + str(GPU_offset) + ".csv")
    print("CSV file we read is ", csv_file_path)


    # Prepare the store file path
    store_file_path = os.path.join(store_folder_path, "sub" + str(GPU_offset) + ".csv")
    if os.path.exists(store_file_path):
        # Remove existing csv
        os.remove(store_file_path)


    # Make the tmp path
    tmp_folder_path = os.path.join(tmp_folder_name, "instance" + str(GPU_offset))
    if os.path.exists(tmp_folder_path):
        shutil.rmtree(tmp_folder_path)
    print("mkdir at ", tmp_folder_path)
    os.makedirs(tmp_folder_path)



    # Init model with different Device
    if not torch.cuda.is_available():
        raise Exception("We should have a cuda machine available!")
    device = torch.device("cuda")

    if "Text_Area" in scoring_criteria:
        import easyocr
        lang_choices = ["en", "ch_sim"]
        OCR_reader = easyocr.Reader(lang_choices)

    if "Image_Quality_Assessment" in scoring_criteria:
        import pyiqa
        iqa_metric = pyiqa.create_metric('clipiqa+', device=device)
    
    if "Aesthetic" in scoring_criteria:
        aesthetic_metric = pyiqa.create_metric('nima', device=device)
    
    if "Image_Complexity" in scoring_criteria:
        from preprocess.auxiliary.ICNet import ICNet
        img_complexity_model = ICNet()
        img_complexity_path = '../pretrained/ck.pth'  # wget https://huggingface.co/incantor/image_complexity_ic9600/resolve/main/ck.pth?download=true
        
        if not os.path.exists(img_complexity_path):
            if not os.path.exists("../pretrained/"):
                os.makedirs("../pretrained/")
            os.system("wget https://huggingface.co/incantor/image_complexity_ic9600/resolve/main/ck.pth && mv ck.pth ../pretrained/ ")
        
        img_complexity_model.load_state_dict(torch.load(img_complexity_path, map_location=torch.device('cpu')))
        img_complexity_model.eval()
        img_complexity_model.to(device)

        # Setup Transform needed
        IC_inference_transform = transforms.Compose([
            transforms.Resize((512, 512)),      # Default: 512x512
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    # Nothing for Clarity for now
    # if "First_Frame_Clarity" in scoring_criteria:



    # Read all row in the csv file
    start_time = time.time()
    info_lists = []       # The order will be follow automatically
    with open(csv_file_path) as file_obj: 
    
        reader_obj = csv.reader(file_obj) 
        
        # Iterate over each row in the csv  
        cur_idx = 0
        for idx, row in enumerate(reader_obj): 

            # Initialize per iter
            temp_store = collections.defaultdict(list)
            exception_case = 0

            # For the first row case (With all title content)
            if idx == 0:    # The first line is the title of content
                elements = dict()
                for element_idx, key in enumerate(row):
                    elements[key] = element_idx

                info_lists.append(row + scoring_criteria)
                print("The first row is ", info_lists[0])

                # Store the csv
                with open(store_file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(info_lists)
                continue


            # Read the important information
            video_path = row[elements["video_path"]]
            height = int(row[elements["height"]])
            width = int(row[elements["width"]])
            fps = float(row[elements["fps"]])
            valid_duration = json.loads(row[elements["valid_duration"]])
            

            try:

                # Read the video by ffmpeg
                resolution = str(width) + "x" + str(height)
                video_stream, err = ffmpeg.input(
                                                    video_path
                                                ).output(
                                                    "pipe:", format = "rawvideo", pix_fmt = "rgb24", s = resolution, vsync = 'passthrough',
                                                ).run(
                                                    capture_stdout = True, capture_stderr = True
                                                )      # The resize is already included
                video_np = np.frombuffer(video_stream, np.uint8).reshape(-1, height, width, 3)
                video_np = video_np[valid_duration[0] : valid_duration[1]]
                num_frames = len(video_np)

            except Exception:
                print("There is error reading ", video_path)
                continue


            # Select frame
            for iter_idx in range(len(samples_on_video)):

                if not debug:
                    full_output_image_path = os.path.join(tmp_folder_path, "tmp_full"+str(iter_idx)+".png")
                    cropped_output_image_path = os.path.join(tmp_folder_path, "tmp_crop"+str(iter_idx)+".png")

                else:
                    full_output_image_path = os.path.join(tmp_folder_path, format(cur_idx, '08') + ".png")
                    cur_idx += 1


                # Write to frames, Will continuously iterate until sucessfully fetch one frame that can be read 
                iter_times = 0
                while True:

                    if iter_times >= 10:
                        print("There are too many times that fail, we skip this case with a whole white place holder")
                        frame = np.zeros((256, 384, 3))
                        exception_case += 1
                        break

                    try:
                        
                        frame_idx = int(samples_on_video[iter_idx] * num_frames)
                        frame = video_np[frame_idx]

                        # Store the sample (full and cropped)
                        cv2.imwrite(full_output_image_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        if text_area_crop:
                            cropped_frame = frame[:int(height * 0.57), :, :]    # Crop the image for the tallest 60% and the width now is 80% for a reasonable aspect ratio
                            cv2.imwrite(cropped_output_image_path, cropped_frame)

                    except Exception:
                        print("There is exception. We continue find a new frame index that may work ", video_path)
                        iter_times += 1
                        continue
                
                    break
                
                if iter_times >= 10:    # Too many exception cases occur
                    print("Too many exception occurs")
                    continue
                

                ####################### Upon this line, we have a valid frame read...

                # Text bounding box detection
                if "Text_Area" in scoring_criteria:
                    if text_area_crop:
                        bounds = OCR_reader.readtext(cropped_output_image_path)
                    else:
                        bounds = OCR_reader.readtext(full_output_image_path)
                    total_area = 0

                    for bound in bounds:
                        coordinates, content, confidence = bound
                        # top_left, top_right, bottom_right, bottom_left = coordinates
                        # area = (bottom_right[0] - top_left[0]) * (bottom_right[1] - top_left[1])
                        total_area += polygon_area(coordinates)
                    # We should calculate the ratio with respect to the whole image, because the resolution to each is different
                    text_ratio = total_area / (height * width)
                    temp_store["Text_Area"].append(text_ratio)


                # Image Quality Assessment
                if "Image_Quality_Assessment" in scoring_criteria:
                    iqa_score = iqa_metric(full_output_image_path).detach().cpu().numpy()[0][0]
                    temp_store["Image_Quality_Assessment"].append(iqa_score)
                

                # Aesthetic Assessment
                if "Aesthetic" in scoring_criteria:
                    aesthetic_score = aesthetic_metric(full_output_image_path).detach().cpu().numpy()[0][0]
                    temp_store["Aesthetic"].append(aesthetic_score)
                    # print("Aesthetic score is ", aesthetic_score)


                # Image Complexity Assessment
                if "Image_Complexity" in scoring_criteria:
                    ori_img = Image.open(full_output_image_path).convert("RGB")
                    img = IC_inference_transform(ori_img)
                    img = img.cuda()
                    img = img.unsqueeze(0)
                    ic_score, _ = img_complexity_model(img)
                    ic_score = ic_score.item()
                    
                    temp_store["Image_Complexity"].append(ic_score)
                    # print("IC score is ", ic_score)


                # Clarity
                if frame_idx == 0 and "First_Frame_Clarity" in scoring_criteria:

                    # Read img
                    # print("We do First_Frame_Clarity at frame", frame_idx)
                    img = cv2.imread(full_output_image_path, cv2.IMREAD_GRAYSCALE)

                    # Process img
                    clarity_score = cv2.Laplacian(img, cv2.CV_64F).var()        # Higher Better

                    # Append to the list
                    temp_store["First_Frame_Clarity"].append(clarity_score)


                # After using the tmp image, we should delete it, else the storage is crashed
                if not debug:
                    os.remove(full_output_image_path)
                    if text_area_crop:
                        os.remove(cropped_output_image_path)

            # Average result
            organized_values = row
            for key in scoring_criteria:
                value = sum(temp_store[key]) / len(temp_store[key])
                organized_values.append(value)
            info_lists.append(organized_values)


            # Log update
            if idx % store_freq == 0:
                print("We have processed ", float(idx/1000), "K video")
                full_time_spent = int(time.time() - start_time)
                print("Time spent is %d min %d s" %(full_time_spent//60, full_time_spent%60))

                # Store the csv
                with open(store_file_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(info_lists[-1*store_freq:])
                

        # Last append for the rest; the following might raise bugs
        # with open(store_file_path, 'a', newline='') as csvfile:
        #     writer = csv.writer(csvfile)
        #     left_amount = idx % store_freq
        #     writer.writerows(info_lists[-1*left_amount:])


    # Clean the tmp at the end
    shutil.rmtree(tmp_folder_path)



if __name__ == "__main__":

    # Argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU_offset', type=int, default=0)
    args = parser.parse_args()


    # Fundamental Setting
    csv_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_scoring_SceneCut_left"       # Input
    store_folder_path = "/PATH/TO/CSV_FOLDER/general_dataset_scoring_img"               # Output
    scoring_criteria = ["Text_Area", "Image_Quality_Assessment", "Aesthetic", "Image_Complexity", "First_Frame_Clarity"]        #  First_Frame_Clarity only do the first frame
    samples_on_video = [0.0, 0.5, 0.95]         # Ratio of process across whole video duration
    text_area_crop = False                      # True for Webvid, we crop the watermark region out by empiricaly region
    tmp_folder_name = "tmp_img_scoring/"        # temporary folder to store intermediate result
    GPU_offset = args.GPU_offset



    # Prepare the csv file
    if not os.path.exists(store_folder_path):
        # shutil.rmtree(store_folder_path)
        os.makedirs(store_folder_path)


    # Prepare return items
    manager = multiprocessing.Manager()
    return_dict = manager.dict()


    # Parallel process
    start_time = time.time()
    single_process(csv_folder_path, store_folder_path, GPU_offset, samples_on_video, scoring_criteria, text_area_crop)
    full_time_spent = int(time.time() - start_time)
    print("Total time spent for this video is %d min %d s" %(full_time_spent//60, full_time_spent%60))


