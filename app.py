import os, sys, shutil
import csv
import numpy as np
import ffmpeg
import cv2
import collections
import json
import math
import time
import imageio
import random
import ast
import gradio as gr
from omegaconf import OmegaConf
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
# from diffusers import AutoencoderKLCogVideoX
# from transformers import T5EncoderModel
from diffusers.utils import export_to_video, load_image


# Import files from the local fodler
root_path = os.path.abspath('.')
sys.path.append(root_path)
# from pipelines.pipeline_cogvideox_i2v_motion_FrameINO import CogVideoXImageToVideoPipeline
# from architecture.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from data_loader.video_dataset_motion import VideoDataset_Motion
from architecture.transformer_wan import WanTransformer3DModel
from pipelines.pipeline_wan_i2v_motion_FrameINO import WanImageToVideoPipeline
from architecture.autoencoder_kl_wan import AutoencoderKLWan



MARKDOWN = \
    """
    <div align='center'> 
        <h1> Frame In-N-Out </h1> \
            <h2 style='font-weight: 450; font-size: 1rem; margin-bottom: 1rem;'>\
                <a href='https://kiteretsu77.github.io/BoyangWang/'>Boyang Wang</a>,  <a href='https://xuweiyichen.github.io/'>Xuweiyi Chen</a>,   <a href='http://mgadelha.me/'>Matheus Gadelha</a>,  <a href='https://sites.google.com/site/zezhoucheng/'>Zezhou Cheng</a>\
            </h2> \
        
        <div style="display: flex; flex-wrap: wrap; justify-content: center; gap: 2rem; margin-bottom: 1rem;">
            <a href="https://arxiv.org/abs/2505.21491" target="_blank"
            style="display: inline-flex; align-items: center; padding: 0.5rem 1rem; background-color: #f0f0f0; /* 浅灰色背景 */ color: #333; /* 深色文字 */ text-decoration: none; border-radius: 9999px; font-weight: 500; transition: background-color 0.3s;">
                <span style="margin-right: 0.5rem;">📄</span> 
                <span>Paper</span>
            </a>
            <a href="https://github.com/UVA-Computer-Vision-Lab/FrameINO" target="_blank"
            style="display: inline-flex; align-items: center; padding: 0.5rem 1rem; background-color: #f0f0f0; color: #333; text-decoration: none; border-radius: 9999px; font-weight: 500; transition: background-color 0.3s;">
                <span style="margin-right: 0.5rem;">💻</span> 
                <span>GitHub</span>
            </a>
            <a href="https://uva-computer-vision-lab.github.io/Frame-In-N-Out" target="_blank"
            style="display: inline-flex; align-items: center; padding: 0.5rem 1rem; background-color: #f0f0f0; color: #333; text-decoration: none; border-radius: 9999px; font-weight: 500; transition: background-color 0.3s;">
                <span style="margin-right: 0.5rem;">🤖</span>
                <span>Project Page</span>
            </a>
            <a href="https://huggingface.co/collections/uva-cv-lab/frame-in-n-out" target="_blank"
            style="display: inline-flex; align-items: center; padding: 0.5rem 1rem; background-color: #f0f0f0; color: #333; text-decoration: none; border-radius: 9999px; font-weight: 500; transition: background-color 0.3s;">
                <span style="margin-right: 0.5rem;">🤗</span>
                <span>HF Model and Data</span>
            </a>
        </div>

       
    </div>

    Frame In-N-Out extends the first-frame conditioning to a larger spatial canvas by specifying top-left and bottom-right expansion offsets. 
    Further, it allows users to assign motion trajectories to existing objects, introduce new identities that enter the scene with their own trajectories, or both.<br>
    The model we used here is [<b>Wan2.2-5B</b> V1.6](https://huggingface.co/uva-cv-lab/FrameINO_Wan2.2_5B_Stage2_MotionINO_v1.6) trained on our Frame In-N-Out control mechanism.


    <br>
    <b>Easiest way:</b> 
        Choose one example and then simply click <b>Generate</b>.

    <br>
    <br>
    ❗️❗️❗️Instruction Steps:<br>
    1️⃣ Upload your first frame image. Set the size you want to resize to for <b>Resized Height for Input Image</b> and <b>Resized Width for Input Image</b>.  <br> 
    2️⃣ Set your <b>canvas top left</b> and <b>bottom right expansion</b>. The combined height and width should be the multiplier of 32. <br>
        Recommend <b>Canvas HEIGHT = 704</b> and <b>Canvas WIDTH = 1280</b> for the best performance (Pre-trained training Resolution). <br>
    3️⃣ Click <b>Build the Canvas</b>.  <br>
    4️⃣ Provide the trajectory of the main object in the canvas by clicking on the <b>Expanded Canvas</b>. <br>
    5️⃣ Provide the ID reference image and its trajectory (optional). Also, write a detailed <b>text prompt</b>. <br>
    Click the <b>Generate</b> button to start the Video Generation. <br>

    
    If **Frame In-N-Out** is helpful, please help star the [GitHub Repo](https://github.com/UVA-Computer-Vision-Lab/FrameINO?tab=readme-ov-file). Thanks! 
    
    """



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





######################################################## CogVideoX #################################################################

# # Path Setting
# model_code_name = "CogVideox"
# base_model_id = "zai-org/CogVideoX-5b-I2V"  
# transformer_ckpt_path = "uva-cv-lab/FrameINO_CogVideoX_Stage2_MotionINO_v1.0"

# # Load Model
# transformer = CogVideoXTransformer3DModel.from_pretrained(transformer_ckpt_path, torch_dtype=torch.float16)
# text_encoder = T5EncoderModel.from_pretrained(base_model_id, subfolder="text_encoder", torch_dtype=torch.float16)
# vae = AutoencoderKLCogVideoX.from_pretrained(base_model_id, subfolder="vae", torch_dtype=torch.float16)

# # Create pipeline and run inference
# pipe = CogVideoXImageToVideoPipeline.from_pretrained(
#             base_model_id,
#             text_encoder = text_encoder,
#             transformer = transformer,
#             vae = vae,
#             torch_dtype = torch.float16,
#         )
# pipe.enable_model_cpu_offload()

#####################################################################################################################################




######################################################## Wan2.2 5B #################################################################

# Path Setting
model_code_name = "Wan"
base_model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"  
transformer_ckpt_path = "uva-cv-lab/FrameINO_Wan2.2_5B_Stage2_MotionINO_v1.6"


# Load model
print("Loading the model!")
transformer = WanTransformer3DModel.from_pretrained(transformer_ckpt_path, torch_dtype=torch.float16)
vae = AutoencoderKLWan.from_pretrained(base_model_id, subfolder="vae", torch_dtype=torch.float32)

# Create the pipeline
print("Loading the pipeline!")
pipe = WanImageToVideoPipeline.from_pretrained(base_model_id, transformer=transformer, vae=vae, torch_dtype=torch.bfloat16)
pipe.to("cuda")
pipe.enable_model_cpu_offload()

#####################################################################################################################################




########################################################## Other Auxiliary Func #################################################################

# # Init SAM model
model_type = "vit_h"        #vit-h has the most number of paramter
sam_pretrained_path = "pretrained/sam_vit_h_4b8939.pth"
if not os.path.exists(sam_pretrained_path):
    os.system("wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P pretrained/")
sam = sam_model_registry[model_type](checkpoint = sam_pretrained_path).to(device="cuda")
sam_predictor = SamPredictor(sam)     # There is a lot of setting here

#####################################################################################################################################




# Examples Sample
def get_example():
    case = [
                [
                    '__assets__/horse.jpg',
                    480,
                    736,
                    128,
                    224,
                    96,
                    320,
                    '__assets__/sheep.png',
                    "A brown horse with a black mane walks to the right on a wooden path in a green forest, and then a white sheep enters from the left and walks toward it. Natural daylight, realistic texture, smooth motion, cinematic focus, 4K detail.",
                    [[[[299, 241], [390, 236], [461, 245], [521, 249], [565, 240], [612, 246], [666, 245]], [[449, 224], [488, 212], [512, 206], [531, 209], [552, 202], [581, 204], [609, 210], [657, 206], [703, 202], [716, 211]]], [[[24, 305], [104, 300], [167, 299], [219, 303], [270, 296], [295, 304]]]],
                ],

                [
                    '__assets__/cup.jpg',
                    448,
                    736,
                    256,
                    64,
                    0,
                    480,
                    '__assets__/hand2.png',
                    "A human hand reaches into the frame, gently grabbing the black metal cup with a golden character design on the front, lifting it off the table and taking it away.",
                    [[[[565, 324], [473, 337], [386, 345], [346, 340], [339, 324], [352, 212], [328, 114], [328, 18], [348, 0]]]],
                ],

                [   
                    '__assets__/grass.jpg',
                    512,
                    800,
                    64,
                    64,
                    160, 
                    416,
                    '__assets__/dog.png',
                    "A fluffy, adorable puppy joyfully sprints onto the bright green grass, its fur bouncing with each step as sunlight highlights its soft coat. The scene takes place in a peaceful park filled with tall trees casting gentle shadows across the lawn. After dashing forward with enthusiasm, the puppy slows to a happy trot, continuing farther ahead into the deeper area of the park, disappearing toward the more shaded grass beneath the trees.",
                    [[[[600, 412], [512, 394], [408, 358], [333, 336], [270, 313], [259, 260], [236, 222], [231, 180]], [[592, 392], [295, 305], [256, 217], [243, 163]]]],
                ],

                [
                    '__assets__/man_scene.jpg',
                    576,
                    1024,
                    64,
                    32,
                    64,
                    224,
                    None,
                    "A single hiker, equipped with a backpack, walks toward the right side of a rugged mountainside trail. The bright sunlight highlights the pale rocky terrain around him, while massive stone cliffs loom in the background. Sparse patches of grass and scattered boulders sit along the path, emphasizing the isolation and vastness of the mountain environment as he steadily continues his journey.",
                    [[[[342, 247], [415, 247], [478, 262], [518, 271], [570, 275], [613, 283], [646, 308], [690, 307], [705, 325]], [[349, 227], [461, 232], [536, 254], [595, 252], [638, 269], [691, 289], [715, 291]], [[341, 283], [415, 291], [500, 316], [590, 317], [632, 354], [675, 362], [711, 372]]]],
                ]

            ]
    return case




def on_example_click(
                        input_image, resized_height, resized_width, 
                        top_left_height, top_left_width, bottom_right_height, bottom_right_width, 
                        identity_image, text_prompt, traj_lists,
                    ):

    # Convert
    traj_lists = ast.literal_eval(traj_lists)
    # Note: No need for the rest like resized_width and resized_height, because these will be replaced in function


    # Sequentially build the canvas (We don't accept the empty traj_lists & traj_instance_idx returned by build_canvas)
    visual_canvas, initial_visual_canvas, inference_canvas, _, _ = build_canvas(input_image, resized_height, resized_width, top_left_height, top_left_width, bottom_right_height, bottom_right_width)


    # Sequentially load the Trajs of all instances on the canvas
    visual_canvas, traj_instance_idx = fn_vis_all_instance_traj(visual_canvas, traj_lists)


    return visual_canvas, initial_visual_canvas, inference_canvas, traj_instance_idx




def build_canvas(input_image_path, resized_height, resized_width, top_left_height, top_left_width, bottom_right_height, bottom_right_width):

    # Init
    canvas_color = (250, 249, 246)      # This color is like white color used in painting paper 
    

    # Convert the string to integer
    if not resized_height.isdigit():
        raise gr.Error("resized_height must be integer input!")
    resized_height = int(resized_height)
    
    if not resized_width.isdigit():
        raise gr.Error("resized_width must be integer input!")
    resized_width = int(resized_width)

    if not top_left_height.isdigit():
        raise gr.Error("top_left_height must be integer input!")
    top_left_height = int(top_left_height)

    if not top_left_width.isdigit():
        raise gr.Error("top_left_width must be integer input!")
    top_left_width = int(top_left_width)

    if not bottom_right_height.isdigit():
        raise gr.Error("bottom_right_height must be integer input!")
    bottom_right_height = int(bottom_right_height)

    if not bottom_right_width.isdigit():
        raise gr.Error("bottom_right_width must be integer input!")
    bottom_right_width = int(bottom_right_width)



    # Read the original image and preprare the placeholder
    first_frame_img = np.uint8(np.asarray(Image.open(input_image_path)))          # NOTE: this is BGR form, be careful for the later cropping process for ID Reference
    print("first_frame_img shape is ", first_frame_img.shape)


    # Resize to a uniform resolution
    first_frame_img = cv2.resize(first_frame_img, (resized_width, resized_height), interpolation = cv2.INTER_AREA)
    print("first_frame_img is resized to", first_frame_img.shape)


    # Expand to Outside Region to form the Canvas
    expand_height = resized_height + top_left_height + bottom_right_height
    expand_width = resized_width + top_left_width + bottom_right_width
    inference_canvas = np.uint8(np.zeros((expand_height, expand_width, 3)))       # Whole Black Canvas, same as other inference
    visual_canvas = np.full((expand_height, expand_width, 3), canvas_color, dtype=np.uint8)
    print("Init Visual Canvas shape is", visual_canvas.shape)
    print("Init Inference Canvs shape is", inference_canvas.shape)


    # Sanity Check 
    if expand_height % 32 != 0:
        raise gr.Error("The Height of resized_height + top_left_height + bottom_right_height must be divisible by 32!")
    if expand_width % 32 != 0:
        raise gr.Error("The Width of resized_width + top_left_width + bottom_right_width must be divisible by 32!")


    # Sanity Check 
    if expand_height % 32 != 0:
        raise gr.Error("The Height of resized_height + top_left_height + bottom_right_height must be divisible by 32!")
    if expand_width % 32 != 0:
        raise gr.Error("The Width of resized_width + top_left_width + bottom_right_width must be divisible by 32!")


    # Draw the Region Box Region (Original Resolution)
    bottom_len = inference_canvas.shape[0] - bottom_right_height
    right_len = inference_canvas.shape[1] - bottom_right_width
    inference_canvas[top_left_height:bottom_len, top_left_width:right_len, :] = first_frame_img
    visual_canvas[top_left_height:bottom_len, top_left_width:right_len, :] = first_frame_img


    # Resize to the uniform height and width
    visual_canvas = cv2.resize(visual_canvas, (uniform_width, uniform_height), interpolation = cv2.INTER_AREA)
    print("Visual Canvas resized to", visual_canvas.shape)


    # Return the visual_canvas (for visualizaiton) and canvas map
    # Corresponds to: visual_canvas, initial_visual_canvas, inference_canvas, traj_instance_idx, traj_lists
    return visual_canvas, visual_canvas.copy(), inference_canvas, 0, [ [ [] ] ]     # The last two is initialized with the trajectory instance idx and trajectory list




def process_points(traj_list, num_frames=81):


    if len(traj_list) < 2:     # First point
        return [traj_list[0]] * num_frames

    elif len(traj_list) >= num_frames:
        raise gr.Info("The number of trajectory points is more than limits, we will do cropping!")
        skip = len(traj_list) // num_frames
        return traj_list[::skip][: num_frames - 1] + traj_list[-1:]

    else:

        insert_num = num_frames - len(traj_list)
        insert_num_dict = {}
        interval = len(traj_list) - 1
        n = insert_num // interval
        m = insert_num % interval

        for i in range(interval):
            insert_num_dict[i] = n

        for i in range(m):
            insert_num_dict[i] += 1

        res = []
        for i in range(interval):
            insert_points = []
            x0, y0 = traj_list[i]
            x1, y1 = traj_list[i + 1]

            delta_x = x1 - x0
            delta_y = y1 - y0
            for j in range(insert_num_dict[i]):
                x = x0 + (j + 1) / (insert_num_dict[i] + 1) * delta_x
                y = y0 + (j + 1) / (insert_num_dict[i] + 1) * delta_y
                insert_points.append([int(x), int(y)])

            res += traj_list[i : i + 1] + insert_points
        res += traj_list[-1:]

        # return
        return res



def fn_vis_realtime_traj(visual_canvas, traj_list, traj_instance_idx):  # Visualize the traj on canvas

    # Process Points
    points = process_points(traj_list)

    # Draw straight line to connect
    for i in range(len(points) - 1):
        p = points[i]
        p1 = points[i + 1]
        cv2.line(visual_canvas, p, p1, all_color_codes[traj_instance_idx], 5)

    return visual_canvas


def fn_vis_all_instance_traj(visual_canvas, traj_lists):  # Visualize all traj from all instances on canvas

    for traj_instance_idx, traj_list_instance in enumerate(traj_lists):
        for traj_list_line in traj_list_instance:
            visual_canvas = fn_vis_realtime_traj(visual_canvas, traj_list_line, traj_instance_idx)

    return visual_canvas, traj_instance_idx     # Also return the instance idx


def add_traj_point(
                    visual_canvas,
                    traj_lists,
                    traj_instance_idx, 
                    evt: gr.SelectData,
                ):  # Add new Traj and then visualize

    # Convert
    traj_lists = ast.literal_eval(traj_lists)

    # Mark New Trajectory Key Point
    hotizontal, vertical = evt.index

    # traj_lists data structure is: (Num of Instnace, Num of Trajecotries, Num of Points, [X, Y])
    traj_lists[-1][-1].append( [int(hotizontal), int(vertical)] )

    # Draw new trajectory on the Canvas image
    visual_canvas = fn_vis_realtime_traj(visual_canvas, traj_lists[-1][-1], traj_instance_idx)


    # Return New Traj Marked Canvas image
    return visual_canvas, traj_lists



def clear_traj_points(initial_visual_canvas):


    return initial_visual_canvas.copy(), 0, [ [ [] ] ]         # 1sr One is the initial state canvas; 2nd one is the traj instance idx; 3rd one is the traj list (with the same data structure)


def traj_point_update(traj_lists):

    # Convert
    traj_lists = ast.literal_eval(traj_lists)

    # Append on the last trajecotry line
    traj_lists[-1].append([])

    return traj_lists



def traj_instance_update(traj_instance_idx, traj_lists):

    # Convert
    traj_lists = ast.literal_eval(traj_lists)

    # Update one index
    if traj_instance_idx >= len(all_color_codes):
        raise gr.Error("The trajectory instance number is over the limit!")

    # Add one for the traj instance
    traj_instance_idx = traj_instance_idx + 1

    # Append a new empty list to the traj lists
    traj_lists.append([[]])

    # Reutn
    return traj_instance_idx, traj_lists



def sample_traj_by_length(points, num_samples):
    # Sample points evenly from traj based on the euclidean distance

    pts = np.array(points, dtype=float)  # shape (M, 2)

    # 1) 每段长度
    seg = pts[1:] - pts[:-1]
    seg_len = np.sqrt((seg**2).sum(axis=1))  # shape (M-1,)
    
    # 2) 累积长度
    cum = np.cumsum(seg_len)
    total_length = cum[-1]
    
    # 3) 目标等距长度位置
    target = np.linspace(0, total_length, num_samples)
    
    res = []
    for t in target:
        # 4) 找到它落在哪一段
        idx = np.searchsorted(cum, t)
        if idx == 0:
            prev = 0.
        else:
            prev = cum[idx-1]
        
        # 5) 在该段内插值
        ratio = (t - prev) / seg_len[idx]
        p = pts[idx] * ratio + pts[idx+1] * (1-ratio)  # careful: direction reversed?
        # Actually want: start*(1-ratio) + end*ratio
        p = pts[idx] * (1 - ratio) + pts[idx+1] * ratio
        res.append(p)
    return np.array(res)



def inference(inference_canvas, visual_canvas, text_prompt, traj_lists, main_reference_img, 
                resized_height, resized_width, top_left_height, top_left_width, bottom_right_height, bottom_right_width):

    # TODO: enhance the text prompt by Qwen3-VL-32B?


    # Convert
    resized_height = int(resized_height)
    resized_width  = int(resized_width)
    top_left_height = int(top_left_height)
    top_left_width  = int(top_left_width)
    bottom_right_height = int(bottom_right_height)
    bottom_right_width  = int(bottom_right_width)
    traj_lists = ast.literal_eval(traj_lists)
    


    # Init Some Fixed Setting
    if model_code_name == "Wan":
        config_path = "config/train_wan_motion_FrameINO.yaml"
        dot_radius = 7
        num_frames = 81
    elif model_code_name == "CogVideoX":
        config_path = "config/train_cogvideox_i2v_motion_FrameINO.yaml"
        dot_radius = 6
        num_frames = 49
    config = OmegaConf.load(config_path)
    

    # Prepare tmp folders
    print()
    store_folder_path = "tmp_app_example_" + str(int(time.time()))
    if os.path.exists(store_folder_path):
        shutil.rmtree(store_folder_path)
    os.makedirs(store_folder_path)


    # Write the visual canvas
    visual_canvas_store_path = os.path.join(store_folder_path, "visual_canvas.png")
    cv2.imwrite( visual_canvas_store_path, cv2.cvtColor(visual_canvas, cv2.COLOR_BGR2RGB) )
    


    # Resize the map
    canvas_width = resized_width + top_left_width + bottom_right_width
    canvas_height = resized_height + top_left_height + bottom_right_height
    # inference_canvas = cv2.resize(visual_canvas, (canvas_width, canvas_height), interpolation = cv2.INTER_AREA)
    print("Canvas Shape is", str(canvas_height) + "x" + str(canvas_width) )


    # TODO: 还要去enhance这个text prompt要跟QWen的保持一致的complexity的感觉。。。
    
    # Save the text prompt
    print("Text Prompt is", text_prompt)
    with open(os.path.join(store_folder_path, 'text_prompt.txt'), 'w') as file:
        file.write(text_prompt)


    ################################################## Motion Trajectory Condition #####################################################

    # #Prepare the points in the linear way
    full_pred_tracks = [[] for _ in range(num_frames)]
    ID_tensor = None

    # Iterate all tracking information for all objects
    print("traj_lists is", traj_lists)
    for instance_idx, traj_list_per_object in enumerate(traj_lists):

        # Iterate all trajectory lines in one instance
        for traj_idx, single_trajectory in enumerate(traj_list_per_object):
            
            # Sanity Check
            if len(single_trajectory) < 2:
                raise gr.Error("One of the trajectory provided is too short!")


            # Sampled the point based on the Euclidean distance
            sampled_points = sample_traj_by_length(single_trajectory, num_frames)


            # Iterate all points
            temporal_idx = 0
            for (raw_point_x, raw_point_y) in sampled_points:

                # Scale the point coordinate to the Infernece Size (Realistic Canvas size)
                point_x, point_y = int(raw_point_x * canvas_width / uniform_width), int(raw_point_y * canvas_height / uniform_height)       # Clicking on the board is with respect to the Uniform Preset Height and Width

                if traj_idx == 0:       # Needs to init the list in list
                    full_pred_tracks[temporal_idx].append( [] ) 
                full_pred_tracks[temporal_idx][-1].append( (point_x, point_y) )        # [-1] and [instance_idx] should have the same effect
                temporal_idx += 1

            
    # Create the traj tensor
    traj_tensor, traj_imgs_np, _, img_with_traj = VideoDataset_Motion.prepare_traj_tensor(
                                                                                            full_pred_tracks, canvas_height, canvas_width, 
                                                                                            [], dot_radius, canvas_width, canvas_height, 
                                                                                            idx=0, first_frame_img = inference_canvas
                                                                                        )


    # Store Trajectory
    imageio.mimsave(os.path.join(store_folder_path, "traj_video.mp4"),  traj_imgs_np, fps=8)

    ######################################################################################################################################################

    
    
    ########################################## Prepare the Identity Reference Condition #####################################################


    # ID reference preparation
    if main_reference_img is not None:
        print("We have an ID reference being used!")

        # Fetch
        ref_h, ref_w, _ = main_reference_img.shape


        # Using breakpoint to extract the points
        sam_predictor.set_image(np.uint8(main_reference_img))


        # Define the sample point
        sam_points = [(ref_w//2, ref_h//2)] # We don't need that many points to express       [:len(traj_points)//2]
        

        # Reverse traj_points
        positive_point_cords = np.array(sam_points)
        positive_point_labels = np.ones(len(positive_point_cords))

        # Predict the mask based on the point and bounding box designed
        masks, scores, logits = sam_predictor.predict(  
                                                        point_coords = positive_point_cords,
                                                        point_labels = positive_point_labels,
                                                        multimask_output = False,
                                                    )
        mask = masks[0]
        main_reference_img[mask == False] = 0   # Merge the mask the first first frame


        # Resize to the same resolution as the first frame 
        scale_h = canvas_height / max(ref_h, ref_w)
        scale_w = canvas_width / max(ref_h, ref_w)
        new_h, new_w = int(ref_h * scale_h), int(ref_w * scale_w)
        main_reference_img = cv2.resize(main_reference_img, (new_w, new_h), interpolation = cv2.INTER_AREA)

        # Calculate padding amounts on all direction
        pad_height1 = (canvas_height - main_reference_img.shape[0]) // 2
        pad_height2 = canvas_height - main_reference_img.shape[0] - pad_height1
        pad_width1 = (canvas_width - main_reference_img.shape[1]) // 2
        pad_width2 = canvas_width - main_reference_img.shape[1] - pad_width1

        # Apply padding to same resolution as the training farmes
        main_reference_img = np.pad(
                                        main_reference_img, 
                                        ((pad_height1, pad_height2), (pad_width1, pad_width2), (0, 0)), 
                                        mode = 'constant', 
                                        constant_values = 0
                                    )

        cv2.imwrite(os.path.join(store_folder_path, "ID.png"), cv2.cvtColor(main_reference_img, cv2.COLOR_BGR2RGB))

    elif main_reference_img is None:
        # Whole Black Color placeholder
        main_reference_img = np.uint8(np.zeros((canvas_height, canvas_width, 3)))


    # Convert to tensor
    ID_tensor = torch.tensor(main_reference_img)
    ID_tensor = train_transforms(ID_tensor).permute(2, 0, 1).contiguous()

    if model_code_name == "Wan":        # Needs to be the shape  (B, C, F, H, W) 
        ID_tensor = ID_tensor.unsqueeze(0).unsqueeze(2)

    ###############################################################################################################################################



    ############################################# Call the Inference Pipeline ##########################################################

    image = Image.fromarray(inference_canvas)

    if model_code_name == "Wan":
        video = pipe(
                        image = image, 
                        prompt = text_prompt, negative_prompt = "",     # Empty string as negative text prompt 
                        traj_tensor = traj_tensor,      # Should be shape (F, C, H, W)
                        ID_tensor = ID_tensor,          # Should be shape (B, C, F, H, W) 
                        height = canvas_height, width = canvas_width, num_frames = num_frames,
                        num_inference_steps = 50,       # 38 is also ok
                        guidance_scale = 5.0,    
                    ).frames[0]     

    elif model_code_name == "CogVideoX":
        video = pipe(
                        image = image, 
                        prompt = text_prompt, 
                        traj_tensor = traj_tensor, 
                        ID_tensor = ID_tensor,
                        height = canvas_height, width = canvas_width, num_frames = len(traj_tensor),
                        guidance_scale = 6, use_dynamic_cfg = False, 
                        num_inference_steps = 50, 
                        add_ID_reference_augment_noise = True,           
                    ).frames[0]
     


    # Store the reuslt
    export_to_video(video, os.path.join(store_folder_path, "generated_video_padded.mp4"), fps=8)



    # Save frames
    print("Writing as Frames")
    video_file_path = os.path.join(store_folder_path, "generated_video.mp4")
    writer = imageio.get_writer(video_file_path, fps = 8)
    for frame_idx, frame in enumerate(video):

        # Extract Unpadded version
        # frame = np.uint8(frame)
        if model_code_name == "CogVideoX":
            frame = np.asarray(frame)        # PIL to RGB
        bottom_right_y = frame.shape[0] - bottom_right_height
        bottom_right_x = frame.shape[1] - bottom_right_width
        cropped_region_frame = np.uint8(frame[top_left_height: bottom_right_y, top_left_width : bottom_right_x] * 255)
        writer.append_data(cropped_region_frame)

    writer.close()

    #####################################################################################################################################

    
    return gr.update(value = video_file_path, width = uniform_width, height = uniform_height)   




if __name__ == '__main__':

    
    # Global Setting 
    uniform_height = 480        # Visual Canvas as 480x720 is decent
    uniform_width = 720


    # Draw the Website
    block = gr.Blocks().queue(max_size=10)
    with block:


        with gr.Row():
            gr.Markdown(MARKDOWN)

        with gr.Row(elem_classes=["container"]):

            with gr.Column(scale=2):
                # Input image
                input_image = gr.Image(type="filepath", label="Input Image 🖼️ ")
                # uploaded_files = gr.Gallery(label="Your images", visible=False, columns=5, rows=1, height=200)

            with gr.Column(scale=2):

                # Input image
                resized_height = gr.Textbox(label="Resized Height for Input Image")
                resized_width = gr.Textbox(label="Resized Width for Input Image")
                # gr.Number(value=unit_height, label="Fixed", interactive=False)
                # gr.Number(value=unit_height * 1.77777, label="Fixed", interactive=False)

                # Input the expansion factor
                top_left_height = gr.Textbox(label="Top-Left Expand Height")
                top_left_width = gr.Textbox(label="Top-Left Expand Width")
                bottom_right_height = gr.Textbox(label="Bottom-Right Expand Height")
                bottom_right_width = gr.Textbox(label="Bottom-Right Expand Width")

                # Button
                build_canvas_btn = gr.Button(value="Build the Canvas")


        with gr.Row():

            with gr.Column(scale=3):
                with gr.Row(scale=3):
                    visual_canvas = gr.Image(height = uniform_height, width = uniform_width, type="numpy", label='Expanded Canvas 🖼️ ')
                    # inference_canvas = gr.Image(height = uniform_height, width = uniform_width, type="numpy")
                    # inference_canvas = None

                with gr.Row(scale=1):
                    # TODO: 还差clear traj的选择
                    add_point = gr.Button(value = "Add New Traj Line (Same Obj)", visible = True)     # Add new trajectory for the same instance
                    add_traj = gr.Button(value = "Add New Instance (New Obj, including new ID)", visible = True)
                    clear_traj_button = gr.Button("Clear All Traj", visible=True)

            with gr.Column(scale=2):

                with gr.Row(scale=2):
                    identity_image = gr.Image(type="numpy", label="Identity Reference (SAM on center point only) 🖼️ ")
                    
                with gr.Row(scale=2):
                    text_prompt = gr.Textbox(label="Text Prompt", lines=3)


        with gr.Row():

            # Button
            generation_btn = gr.Button(value="Generate")


        with gr.Row():
            generated_video = gr.Video(value = None, label="Generated Video", show_label = True, height = uniform_height, width = uniform_width)



        ################################################################## Click + Select + Any Effect Area ###########################################################################

        # Init some states that will be supporting purposes
        traj_lists = gr.Textbox(label="Trajectory", visible = False)    # gr.State(None)       # Data Structure is: (Number of Instance, Number of Trajectories, Points)       Init as [ [ [] ] ] 
        inference_canvas = gr.State(None)
        traj_instance_idx = gr.State(0)
        initial_visual_canvas = gr.State(None)      # gr.Image(height = uniform_height, width = uniform_width, type="numpy", label='Canvas Expanded Image (Initial State)')       # This is the initila visual, used to load back in clearing


        # Canvas Click
        build_canvas_btn.click(
                                build_canvas, 
                                inputs = [input_image, resized_height, resized_width, top_left_height, top_left_width, bottom_right_height, bottom_right_width], 
                                outputs = [visual_canvas, initial_visual_canvas, inference_canvas, traj_instance_idx, traj_lists]       # inference_canvas is used for inference; visual_canvas is for gradio visualization
                            )


        # Draw Trajectory for each click on the canvas
        visual_canvas.select(
                                fn = add_traj_point, 
                                inputs = [visual_canvas, traj_lists, traj_instance_idx], 
                                outputs = [visual_canvas, traj_lists]
                            )


        # Add new Trajectory
        add_point.click(
                            fn = traj_point_update, 
                            inputs = [traj_lists], 
                            outputs = [traj_lists],
                        )
        add_traj.click(
                        fn = traj_instance_update, 
                        inputs = [traj_instance_idx, traj_lists], 
                        outputs = [traj_instance_idx, traj_lists],
                    )
                    
        # Clean all the traj points
        clear_traj_button.click(
                                    clear_traj_points,
                                    [initial_visual_canvas],
                                    [visual_canvas, traj_instance_idx, traj_lists],
                                )


        # Inference Generation
        generation_btn.click(
                                inference,
                                inputs = [inference_canvas, visual_canvas, text_prompt, traj_lists, identity_image, resized_height, resized_width, top_left_height, top_left_width, bottom_right_height, bottom_right_width],
                                outputs = [generated_video],
                            )




        # Load Examples 
        with gr.Row(elem_classes=["container"]):
            gr.Examples(
                            examples = get_example(),
                            inputs = [input_image, resized_height, resized_width, top_left_height, top_left_width, bottom_right_height, bottom_right_width, identity_image, text_prompt, traj_lists],
                            run_on_click = True,
                            fn = on_example_click,
                            outputs = [visual_canvas, initial_visual_canvas, inference_canvas, traj_instance_idx],
                        )
        

    block.launch(share=True)




