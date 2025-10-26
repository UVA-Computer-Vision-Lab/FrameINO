'''
    This is for the WAN2.2 5B (TI2V) version VDM training with Frame In-N-Out Control.
'''

# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import imageio
import logging
import math
import os, sys, shutil
import random
import csv
import cv2
import numpy as np
import shutil
from datetime import timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union
from omegaconf import OmegaConf
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel



import diffusers
# from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.image_processor import PipelineImageInput
from diffusers.training_utils import compute_loss_weighting_for_sd3
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, free_memory
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    export_to_video,
    is_wandb_available,
    load_image,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module


if is_wandb_available():
    import wandb
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")
logger = get_logger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"



# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from architecture.transformer_wan import WanTransformer3DModel
from architecture.autoencoder_kl_wan import AutoencoderKLWan
from architecture.noise_sampler import DiscreteSampling
from pipelines.pipeline_wan_i2v_motion_FrameINO import WanImageToVideoPipeline
from data_loader.sampler import MixedBatchSampler
from data_loader.video_dataset_motion_FrameINO import VideoDataset_Motion_FrameINO




def filter_kwargs(cls, kwargs):
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {'self', 'cls'}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    return filtered_kwargs



def save_tensor_as_video(tensor, save_path, fps=10):
    """
    Save a tensor of shape (N, C, H, W) as a video file.
    Args:
        tensor: Tensor of shape (N, C, H, W) in range [0, 1]
        save_path: Path to save the video file
        fps: Frames per second for the video
    """
    # Convert tensor to numpy array and adjust dimensions
    video = tensor.cpu().detach().numpy()  # (N, C, H, W)
    video = np.transpose(video, (0, 2, 3, 1))  # (N, H, W, C)
    
    # Scale to [0, 255] and convert to uint8
    video = (video * 255).astype(np.uint8)
    
    # Get video dimensions
    N, H, W, C = video.shape
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (W, H))
    
    # Write frames
    for frame in video:
        # Convert from RGB to BGR (OpenCV uses BGR)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    # Release video writer
    out.release()




def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
                            "--config_path",
                            type = str,
                            default = "config/train_wan_motion_FrameINO.yaml",
                            # required = True,
                            help = "Path to the config.",
                        )
    parser.add_argument(
                            "--use_8BitAdam",
                            type = bool,
                            default = False,
                            help = "Whether we use the 8BitAdam.",
                        )
    parser.add_argument(
                            "--debug",
                            type = bool,
                            default = False,
                            help = "Whether we are in the debug model: don't use torch.compile",
                        )

    args = parser.parse_args()
    return args




def log_validation(
                        base_model_path,
                        transformer,
                        vae,
                        text_encoder,
                        accelerator,
                        val_dataloader,
                        validation_store_path,
                        num_inference_steps,
                    ):


    ################################################# The following should be the same as test_code section ##############################################

    # Create pipeline and run inference
    pipe = WanImageToVideoPipeline.from_pretrained(
                                                    base_model_path,
                                                    text_encoder = text_encoder,
                                                    transformer = accelerator.unwrap_model(transformer),
                                                    vae = accelerator.unwrap_model(vae),
                                                    torch_dtype = torch.float16,
                                                )
    pipe = pipe.to(accelerator.device)
    # pipe.set_progress_bar_config(disable=True)



    # Iterate the validation dataset
    for idx, batch in enumerate(val_dataloader):

        if idx != accelerator.process_index:        # We want each process to be different index to do inference
            continue
        print("This Process Idx is", accelerator.process_index)
        

        # Prepare the store folder here such that only one process come here each time
        store_folder_path = os.path.join(validation_store_path, "Process"+str(idx))
        if os.path.exists(store_folder_path):        # Should not have the path exists (unless there is a conflict)
            os.system("rm -rf ", store_folder_path)
        os.makedirs(store_folder_path, exist_ok=True)  


        # Fetch and Only the first batch size
        ## Critical Info: First Frame + Traj (Tensor) + ID (Tensor) + Text Prompt
        first_frame_np = np.asarray(batch["first_frame_np"][0])
        traj_tensor = batch["traj_tensor"][0]   
        ID_tensor = batch["ID_tensor"].unsqueeze(2)     # The shape will be (B, C, F, H, W) 
        text_prompt = batch["text_prompt"][0]
        ## Auxiliary Info
        gt_video_path = batch["gt_video_path"][0]
        merge_frames = batch["merge_frames"][0]
        traj_imgs_np = np.asarray(batch["traj_imgs_np"][0])
        processed_meta_data = batch["processed_meta_data"]
        (top_left_x, top_left_y), (bottom_right_x, bottom_right_y) = processed_meta_data["resized_mask_region_box"]      # Already adjusted the resolution size in the data loader


        # Chcek
        gen_height, gen_width, _ = first_frame_np.shape


        # Save the GT video
        shutil.copyfile(gt_video_path, os.path.join(store_folder_path, "gt_video.mp4"))


        # Convert the first frame numpy (the masked image with 0-value out of BBox region) to image form
        first_frame_path = os.path.join(store_folder_path, "first_frame.png")
        cv2.imwrite(first_frame_path, cv2.cvtColor(first_frame_np, cv2.COLOR_RGB2BGR))


        # Save the traj images
        for traj_idx, traj_img_np in enumerate(traj_imgs_np):

            # Store the traj images (Traj Condition)
            # traj_img_store_path = os.path.join(store_folder_path, "traj_cond" + str(traj_idx) + ".png")
            # cv2.imwrite(traj_img_store_path, cv2.cvtColor(traj_img_np, cv2.COLOR_RGB2BGR))

            # Store the merge frame for better visualization
            merge_img_store_path = os.path.join(store_folder_path, "merge_cond" + str(traj_idx).zfill(2) + ".png")
            cv2.imwrite(merge_img_store_path, cv2.cvtColor( np.uint8(merge_frames[traj_idx]), cv2.COLOR_RGB2BGR))
        

        # Save the video
        imageio.mimsave(os.path.join(store_folder_path, "merge_cond.mp4"), merge_frames, fps=12)


        # Save the text prompt
        with open(os.path.join(store_folder_path, 'text_prompt.txt'), 'w') as file:
            file.write(text_prompt)



        ################################################ Call Validation ################################################################
        
        image = load_image(first_frame_path)
        negative_prompt = ""     # Empty Negative Prompt looks slightly better "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
        video = pipe(
                        image = image, 
                        prompt = text_prompt, negative_prompt = negative_prompt, 
                        traj_tensor = traj_tensor, 
                        ID_tensor = ID_tensor,
                        height = gen_height, width = gen_width, num_frames = len(traj_tensor),
                        num_inference_steps = num_inference_steps,
                        guidance_scale = 5.0      
                    ).frames[0]     # use_dynamic_cfg = True, 


        # Store the reuslt
        export_to_video(video, os.path.join(store_folder_path, "generated_video_padded.mp4"), fps=8)


        # Write as Frames for Generated Video (Padded version)
        for frame_idx, frame_np in enumerate(video):

            # Save the GEN Padded frame
            save_path = os.path.join(store_folder_path, "gen_padded_frame"+str(frame_idx)+".png")
            cv2.imwrite(save_path, cv2.cvtColor(np.uint8(frame_np*255), cv2.COLOR_BGR2RGB))

            # Extract Unpadded version
            cropped_region_frame = frame_np[top_left_y:bottom_right_y, top_left_x:bottom_right_x]

            # Save the region frame (Crop the padding)
            save_path = os.path.join(store_folder_path, "gen_frame"+str(frame_idx)+".png")
            cv2.imwrite(save_path, cv2.cvtColor(np.uint8(cropped_region_frame*255), cv2.COLOR_RGB2BGR))
            
        ################################################################################################################################


        # Only consider one case
        break

    # Delete and clean
    del pipe
    torch.cuda.empty_cache()

    return



def _get_t5_prompt_embeds(
    tokenizer: AutoTokenizer,
    text_encoder: UMT5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    
    # HACK: tokenizer is inlcuded in this function; No need to write it in the data loader
    # prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt)

    if tokenizer is not None:
        text_inputs = tokenizer(
                                    prompt,
                                    padding = "max_length",
                                    max_length = max_sequence_length,
                                    truncation = True,
                                    add_special_tokens = True,
                                    return_tensors = "pt",
                                )
        text_input_ids = text_inputs.input_ids
        # prompt_attention_mask = text_inputs.attention_mask

    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    # Encode Text by T5; 
    # seq_lens = prompt_attention_mask.gt(0).sum(dim=1).long()
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    # prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    # prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]            # This is to fetch the effective prompt length, which is varied for different lang inputs

    return prompt_embeds



def encode_prompt(
    tokenizer: AutoTokenizer,
    text_encoder: UMT5EncoderModel,
    prompt: Union[str, List[str]],
    num_videos_per_prompt: int = 1,
    max_sequence_length: int = 226,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    text_input_ids=None,
):
    
    # prompt = [prompt] if isinstance(prompt, str) else prompt
    prompt_embeds = _get_t5_prompt_embeds(
                                            tokenizer,
                                            text_encoder,
                                            prompt = prompt,
                                            num_videos_per_prompt = num_videos_per_prompt,
                                            max_sequence_length = max_sequence_length,
                                            device = device,
                                            dtype = dtype,
                                            text_input_ids = text_input_ids,
                                        )

    return prompt_embeds



def compute_prompt_embeddings(
    tokenizer, text_encoder, prompt, max_sequence_length, device, dtype, requires_grad: bool = False
):
    
    if requires_grad:
        prompt_embeds = encode_prompt(
                                        tokenizer,
                                        text_encoder,
                                        prompt,
                                        num_videos_per_prompt = 1,
                                        max_sequence_length = max_sequence_length,
                                        device = device,
                                        dtype = dtype,
                                    )
    else:   # For most cases, we don't require grad

        with torch.no_grad():
            prompt_embeds = encode_prompt(
                                            tokenizer,
                                            text_encoder,
                                            prompt,
                                            num_videos_per_prompt = 1,
                                            max_sequence_length = max_sequence_length,
                                            device = device,
                                            dtype = dtype,
                                        )
    return prompt_embeds



def get_optimizer(config, params_to_optimize, use_8BitAdam, use_deepspeed: bool = False):

    optimizer = config["optimizer"]
    use_8bit_adam = use_8BitAdam


    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr = config["learning_rate"],
            betas = (config["adam_beta1"], config["adam_beta2"]),
            eps = config["adam_epsilon"],
            weight_decay = config["adam_weight_decay"],
        )

    # Optimizer creation
    supported_optimizers = ["adam", "adamw", "prodigy"]
    if optimizer not in supported_optimizers:
        logger.warning(
            f"Unsupported choice of optimizer: {optimizer}. Supported optimizers include {supported_optimizers}. Defaulting to AdamW"
        )
        optimizer = "adamw"

    if use_8bit_adam and optimizer.lower() not in ["adam", "adamw"]:
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'Adam' or 'AdamW'. Optimizer was "
            f"set to {optimizer.lower()}"
        )

    if use_8bit_adam:
        try:
            from bitsandbytes.optim import AdamW8bit
            # from torchao.prototype.low_bit_optim import AdamW4bit, AdamW8bit

        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

    if optimizer.lower() == "adamw":    # Most cases we should use AdamW with 8Bit adam to accelerate
        optimizer_class = AdamW8bit if use_8bit_adam else torch.optim.AdamW

        optimizer = optimizer_class(
                                        params_to_optimize,
                                        betas = (config["adam_beta1"], config["adam_beta2"]),
                                        eps = config["adam_epsilon"],
                                        weight_decay = config["adam_weight_decay"],
                                    )

    elif optimizer.lower() == "adam":
        optimizer_class = bnb.optim.Adam8bit if use_8bit_adam else torch.optim.Adam

        optimizer = optimizer_class(
                                        params_to_optimize,
                                        betas = (config["adam_beta1"], config["adam_beta2"]),
                                        eps = config["adam_epsilon"],
                                        weight_decay = config["adam_weight_decay"],
                                    )

    elif optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            betas = (config["adam_beta1"], config["adam_beta2"]),
            beta3 = config["prodigy_beta3"],
            weight_decay = config["adam_weight_decay"],
            eps = config["adam_epsilon"],
            decouple = config["prodigy_decouple"],
            use_bias_correction = config["prodigy_use_bias_correction"],
            safeguard_warmup = config["prodigy_safeguard_warmup"],
        )

    return optimizer



# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")




def video_tensor_to_vae_latent(video_tensor, first_frame_tensor, vae, device):
    
    # NOTE: video_tensor is fully info without any mask; first_frame_tensor will have the mask outside BBox

    # Prepare Mean and variance
    latents_mean = (
                        torch.tensor(vae.config.latents_mean)
                        .view(1, vae.config.z_dim, 1, 1, 1)
                        .to(device, video_tensor.dtype)
                    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, video_tensor.dtype)



    #################################################  Video Latent Prepare ####################################################

    # Preprocess
    video_tensor = video_tensor.to(device=device, dtype=vae.dtype)      
    video_tensor = video_tensor.to(device, dtype=vae.dtype)     #.unsqueeze(0)
    video_tensor = video_tensor.permute(0, 2, 1, 3, 4)          # Output shape is: [B, C, F, H, W]

    # VAE Encode
    video_latents = retrieve_latents(vae.encode(video_tensor), sample_mode="argmax")        # The output shape is different from CogVideoX, which should be [B, C, F, H, W]

    # Normalize based on Mean & Variance
    video_latents = (video_latents - latents_mean) * latents_std

    # Final Convert
    video_latents = video_latents.to(memory_format = torch.contiguous_format).float()

    ##############################################################################################################################



    ##############################################  First Frame Latent Prepare ################################################################

    # Prepare the first frame   (Masked with 0-Value)
    first_frame = first_frame_tensor.unsqueeze(1).permute(0, 2, 1, 3, 4).to(device, dtype=vae.dtype).clone()
    first_frame = first_frame.to(device=device, dtype=vae.dtype)        # NOTE: no padding is needed for Wan2.2

    # VAE Encode
    first_frame_latent = retrieve_latents(vae.encode(first_frame), sample_mode="argmax")

    # Normalize based on Mean & Variance
    first_frame_latent = (first_frame_latent - latents_mean) * latents_std

    # Random Full Zero Mask
    if random.random() < config["noised_image_dropout"]:
        first_frame_latent = torch.zeros_like(first_frame_latent)

    # Final Convert
    first_frame_latent = first_frame_latent.to(memory_format = torch.contiguous_format).float()
    
    ##########################################################################################################################################


    # Return the image and video latents
    return video_latents, first_frame_latent 



def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value


def traj_tensor_to_vae_latent(traj_tensor, vae, device):


    # Prepare Mean and variance
    latents_mean = (
                        torch.tensor(vae.config.latents_mean)
                        .view(1, vae.config.z_dim, 1, 1, 1)
                        .to(device, traj_tensor.dtype)
                    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, traj_tensor.dtype)


    #################################################  Video Latent Prepare ####################################################

    # Preprocess
    traj_tensor = traj_tensor.to(device=device, dtype=vae.dtype)
    traj_tensor = traj_tensor.to(device, dtype=vae.dtype)     #.unsqueeze(0)
    traj_tensor = traj_tensor.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

    # VAE encode
    traj_latents = retrieve_latents(vae.encode(traj_tensor), sample_mode="argmax")

    # Extract Mean and Variance
    traj_latents = (traj_latents - latents_mean) * latents_std

    ##############################################################################################################################


    # Final Convert
    traj_latents = traj_latents.to(memory_format = torch.contiguous_format).float()


    # Return the image and video latents
    return traj_latents 



def ID_tensor_to_vae_latent(ID_tensors, vae, device):

    # Prepare Mean and Variance of the latents
    latents_mean = (
                        torch.tensor(vae.config.latents_mean)
                        .view(1, vae.config.z_dim, 1, 1, 1)
                        .to(device, ID_tensors.dtype)
                    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, ID_tensors.dtype)


    #################################################  Video Latent Prepare ####################################################

    # Preprocess Transform
    ID_tensors = ID_tensors.to(device=device, dtype=vae.dtype)
    ID_tensors = ID_tensors.to(device, dtype=vae.dtype)     


    # VAE encode
    ID_latents = []
    for frame_idx in range(ID_tensors.shape[2]):

        # Fetch
        ID_tensor = ID_tensors[:, :, frame_idx].unsqueeze(2)

        # Single Frame Encode, which will be single frame token
        ID_latent = retrieve_latents(vae.encode(ID_tensor), sample_mode="argmax")        # The output shape is different from CogVideoX, which should be [B, C, F, H, W]

        # Extract Mean and Variance
        ID_latent = (ID_latent - latents_mean) * latents_std

        # Append
        ID_latents.append(ID_latent)


    # Final Convert
    ID_latents = torch.cat(ID_latents, dim = 2)         # Final shape is (B, C, F, H, W)
    ID_latents = ID_latents.to(memory_format = torch.contiguous_format).float()

    ##############################################################################################################################


    # Return the image and video latents
    return ID_latents 


def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value



def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module



def main(config, args):

    # Process args
    debug = args.debug
    use_8BitAdam = args.use_8BitAdam


    # Read Frequently Used config
    resume_from_checkpoint = config["resume_from_checkpoint"]
    output_folder = config["output_folder"]
    experiment_name = config["experiment_name"]
    mixed_precision = config["mixed_precision"]
    report_to = config["report_to"]
    seed = config["seed"]
    base_model_path = config["base_model_path"]
    pretrained_transformer_path = config["pretrained_transformer_path"]
    download_folder_path = config["download_folder_path"]
    train_csv_relative_path = config["train_csv_relative_path"]
    validation_csv_relative_path = config["validation_csv_relative_path"]
    gradient_checkpointing = config["gradient_checkpointing"]         
    learning_rate = config["learning_rate"]
    train_batch_size = config["train_batch_size"]
    dataloader_num_workers = config["dataloader_num_workers"] if not debug else 1       # In debug mode, only has 1 worker
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    max_train_steps = config["max_train_steps"]
    lr_warmup_steps = config["lr_warmup_steps"]
    checkpointing_steps = config["checkpointing_steps"]
    scale_lr = config["scale_lr"]
    checkpoints_total_limit = config["checkpoints_total_limit"]
    revision = config["revision"]
    lr_scheduler = config["lr_scheduler"]
    validation_step = config["validation_step"]
    first_iter_validation = config["first_iter_validation"]
    num_inference_steps = config["num_inference_steps"]
    max_grad_norm = config["max_grad_norm"]


    # Organize
    use_FrameIn = True      


    # Value Check
    if max_train_steps is None:
        print("max_train_steps must be set")
        os.exit(0)

    if torch.backends.mps.is_available() and mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    output_dir = os.path.join(output_folder, experiment_name)
    logging_dir = Path(output_dir, config["logging_name"])

    accelerator_project_config = ProjectConfiguration(project_dir = output_dir, logging_dir = logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)        # HACK: his find_unused_parameters is said to increase the computation cost
    init_kwargs = InitProcessGroupKwargs(backend="nccl", timeout=timedelta(seconds=config["nccl_timeout"]))
    accelerator = Accelerator(
                                gradient_accumulation_steps = gradient_accumulation_steps,
                                mixed_precision = mixed_precision,
                                log_with = report_to,
                                project_config = accelerator_project_config,
                                kwargs_handlers = [ddp_kwargs, init_kwargs],
                            )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    if report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        # We will not push to the hub

    # Prepare models and scheduler
    tokenizer = AutoTokenizer.from_pretrained(
                                                base_model_path, subfolder = "tokenizer", revision = revision
                                            )

    text_encoder = UMT5EncoderModel.from_pretrained(
                                                        base_model_path, 
                                                        subfolder = "text_encoder", 
                                                        torch_dtype = torch.bfloat16,  
                                                        # quantization_config = TorchAoConfig("int8wo"),
                                                    )


    # HACK: Only load BF16 in 5B model, else load FP16
    # Read the Wan3D model based on if the pretrained_transformer_path is available
    if pretrained_transformer_path is not None:
        print("Load Pretrained Transformer model from ", pretrained_transformer_path)
        transformer_model_path = pretrained_transformer_path
    else:
        transformer_model_path = base_model_path

    transformer = WanTransformer3DModel.from_pretrained(
                                                            transformer_model_path,
                                                            torch_dtype = torch.bfloat16,       
                                                        )

    # NOTE: Here we don't need to add a new channel dim, which is already updated in the weights in the model


    # Define VAE
    vae = AutoencoderKLWan.from_pretrained(
                                            base_model_path, 
                                            subfolder = "vae", 
                                            torch_dtype = torch.bfloat16,  
                                        )


    # Noise Scheduler, follow other open-source repo
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
                                                        **filter_kwargs(
                                                                            FlowMatchEulerDiscreteScheduler, 
                                                                            OmegaConf.to_container(config['noise_scheduler_kwargs'])
                                                                        )
                                                    )


    # Adjust the trainable property of each module
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    transformer.requires_grad_(True)       


    # For mixed precision training we CAST ALL non-trainable weights (vae, text_encoder) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.state.deepspeed_plugin:
        # DeepSpeed is handling precision, use what's in the DeepSpeed config
        if (
            "fp16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["fp16"]["enabled"]
        ):
            weight_dtype = torch.float16
        if (
            "bf16" in accelerator.state.deepspeed_plugin.deepspeed_config
            and accelerator.state.deepspeed_plugin.deepspeed_config["bf16"]["enabled"]
        ):
            weight_dtype = torch.float16
    else:
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
                            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
                        )

    text_encoder.to(accelerator.device)    
    transformer.to(accelerator.device).to(weight_dtype)       # HACK: whether transformer use BF16 didn't speed up a lot    
    vae.to(accelerator.device)


    if gradient_checkpointing:
        transformer.enable_gradient_checkpointing()




    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):

        if accelerator.is_main_process:

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "transformer"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()


    def load_model_hook(models, input_dir):

        transformer_ = None

        while len(models) > 0:
            model = models.pop()

            load_model = WanTransformer3DModel.from_pretrained(input_dir, subfolder = "transformer",)
            model.register_to_config(**load_model.config)

            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)



    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config["allow_tf32"] and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if scale_lr:        # We can consdier to scale the LR
        learning_rate = (
                            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
                        )


    # Define trainable parameters
    parameters_list = []
    total_params_for_training = 0
    for name, param in transformer.named_parameters():  # Full Finetune
        parameters_list.append(param)    
        param.requires_grad = True
        total_params_for_training += param.numel()  
    print("Total parameters that will be trained has ", total_params_for_training/(1024**3), "B parameter")
    


    # Optimization Parameters
    params_to_optimize = [{"params": parameters_list, "lr": learning_rate}]

    use_deepspeed_optimizer = (
        accelerator.state.deepspeed_plugin is not None
        and "optimizer" in accelerator.state.deepspeed_plugin.deepspeed_config
    )
    use_deepspeed_scheduler = (
        accelerator.state.deepspeed_plugin is not None
        and "scheduler" in accelerator.state.deepspeed_plugin.deepspeed_config
    )

    optimizer = get_optimizer(config, params_to_optimize, use_8BitAdam, use_deepspeed = use_deepspeed_optimizer)



    ############################################# Function in Function ###############################################
    def collate_fn(examples):

        # Load tensor
        video_tensor = [example["video_tensor"].unsqueeze(0) for example in examples]   
        video_tensor = torch.cat(video_tensor)  # The shape should be (B, F, C, H, W)

        # Load Traj tensor
        traj_tensor = [example["traj_tensor"].unsqueeze(0) for example in examples]   
        traj_tensor = torch.cat(traj_tensor)  # The shape should be (B, F, C, H, W)

        # Load ID_area tensor
        ID_tensor = [example["ID_tensor"].unsqueeze(0) for example in examples]   
        ID_tensor = torch.cat(ID_tensor)

        # Load Main Reference tensor    (With Masked Region with 0-value)
        first_frame_tensor = [example["first_frame_tensor"].unsqueeze(0) for example in examples]   
        first_frame_tensor = torch.cat(first_frame_tensor)   

        # Load all prompts
        prompts = [example["text_prompt"] for example in examples]

        return {
                    "video_tensor" : video_tensor,
                    "traj_tensor" : traj_tensor,
                    "ID_tensor": ID_tensor,
                    "first_frame_tensor": first_frame_tensor,
                    "prompts" : prompts,
                }
    ###################################################################################################################


    ################################################# Prepare Dataset and DataLoaders ########################################################
    
    # Init Training Dataset
    print("Preparing Train Dataloader!")
    train_datasets = [
                        VideoDataset_Motion_FrameINO(   
                                                        config, download_folder_path, train_csv_relative_path, 
                                                        config["train_video_relative_path"], config["train_ID_relative_path"], 
                                                    )
                    ]   # Was designed to support multiple dataset loading; for the simplicity of expression, we just use one dataset.
    concat_dataset = ConcatDataset(train_datasets)
    
    
    # Mix dataset Training (Because the key index for each csv dataset file is different)
    mixed_sampler = MixedBatchSampler(
                                        src_dataset_ls = train_datasets,
                                        batch_size = train_batch_size,
                                        drop_last = True,
                                        shuffle = True,
                                        # Don't Use generator, it will fixed dataset order in resume
                                    )
    train_dataloader = DataLoader(
                                    concat_dataset,
                                    collate_fn = collate_fn,
                                    batch_sampler = mixed_sampler,
                                    num_workers = dataloader_num_workers,
                                )       # NOTE: Don't use shuffle: it will be fixed dataset order in resume

    # Prepare the validation dataloader
    print("Preparing Val Dataloader!")
    val_dataset = VideoDataset_Motion_FrameINO(
                                                    config, download_folder_path, validation_csv_relative_path, 
                                                    config["validation_video_relative_path"], config["validation_ID_relative_path"],
                                                )
    val_dataloader = DataLoader(
                                    val_dataset,
                                    batch_size = 1,
                                    shuffle = True,
                                    num_workers = 1,
                                )   # No need for the collate in Validation

    ##########################################################################################################################################



    # Prepare LR scheduler
    if use_deepspeed_scheduler:
        from accelerate.utils import DummyScheduler

        lr_scheduler = DummyScheduler(
                                        name = lr_scheduler,
                                        optimizer = optimizer,
                                        total_num_steps = max_train_steps * accelerator.num_processes,
                                        num_warmup_steps = lr_warmup_steps * accelerator.num_processes,
                                    )
    else:
        lr_scheduler = get_scheduler(
                                        lr_scheduler,
                                        optimizer = optimizer,
                                        num_warmup_steps = lr_warmup_steps * accelerator.num_processes,
                                        num_training_steps = max_train_steps * accelerator.num_processes,
                                        # num_cycles = config["lr_num_cycles"],
                                        # power = config["lr_power"],
                                    )


    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
                                                                                    transformer, optimizer, train_dataloader, lr_scheduler
                                                                                )


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = config["tracker_name"] or "Wan-I2V"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps


    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {total_params_for_training}")
    logger.info(f"  Num videos = {len(concat_dataset)}")
    logger.info(f"  Batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0



    # Potentially load in the weights and states from a previous save
    if not resume_from_checkpoint:
        initial_global_step = 0
    else:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            print("Resume the latest checkpoints!")
            # Get the mos recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch


    progress_bar = tqdm(
                            range(0, max_train_steps),
                            initial = initial_global_step,
                            desc = "Steps",
                            # Only show the progress bar once on each machine.
                            disable = not accelerator.is_local_main_process,
                        )


    # Flow Matching Denoise function needs
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
    
    idx_sampling = DiscreteSampling(config["train_sampling_steps"], uniform_sampling=True) # By default we use uniform sampling




    # Read the model config
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    for epoch in range(first_epoch, num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]

            with accelerator.accumulate(models_to_accumulate):
                

                # Fetch the info
                video_tensor = batch["video_tensor"]        # Shape is (B, F, C, H, W)
                traj_tensor = batch["traj_tensor"]          # Shape is (B, F, C, H, W)
                prompts = batch["prompts"]
                first_frame_tensor = batch["first_frame_tensor"]
                if use_FrameIn:
                    ID_tensor = batch["ID_tensor"]
                batch_size, pixel_num_frames, num_channels, pixel_height, pixel_width = video_tensor.shape
                

                # Encode Video and Traj tensor to latnets
                with torch.no_grad():

                    # VAE encode for video and image (masked) and traj latents
                    video_latents, first_frame_latent = video_tensor_to_vae_latent(video_tensor, first_frame_tensor, vae, accelerator.device)
                    traj_latents = traj_tensor_to_vae_latent(traj_tensor, vae, accelerator.device)

                    # Mask First Frame of video_latents with Zero masked First Frame Condition
                    video_latents[:, :, :1] = first_frame_latent         # Let the 1st frame of GT to be aligned with noisy_video_latents.  


                    # Refernece Tensor Adjustment for FrameIn cases. 
                    if use_FrameIn:

                        # VAE encode the ID reference latents
                        ID_tensor = ID_tensor.unsqueeze(2)      # The output shape will be aligned to (B, C, F, H, W)
                        ID_latent = ID_tensor_to_vae_latent(ID_tensor, vae, accelerator.device)       

                        # Increase the Frame Dimension of the Traj latents and the first frame latent
                        ID_latent_padding = video_latents.new_zeros(ID_latent.shape)                   # Zero latent values  
                        traj_latents = torch.cat([traj_latents, ID_latent_padding], dim = 2)
                
                # Fetch the shape
                batch_size, num_channels, num_gen_frames, latent_height, latent_width = video_latents.shape


                # Prepare Prompt
                prompt_embeds = compute_prompt_embeddings(
                                                            tokenizer,
                                                            text_encoder,
                                                            prompts,
                                                            config["max_text_seq_length"],
                                                            accelerator.device,
                                                            weight_dtype,
                                                            requires_grad = False,
                                                        )


                # Sample a random timestep for each image
                indices = idx_sampling(batch_size, device=video_latents.device)
                indices = indices.long().cpu()
                timesteps = noise_scheduler.timesteps[indices].to(device=video_latents.device)     
    

                # Add noise according to Flow Matching
                noise = torch.randn(video_latents.size(), device=video_latents.device, dtype=weight_dtype)
                sigmas = get_sigmas(timesteps, n_dim=video_latents.ndim, dtype=video_latents.dtype)
                noisy_video_latents = (1.0 - sigmas) * video_latents + sigmas * noise


                # Merge the first frame latent into the noisy_video_latents
                noisy_video_latents[:, :, :1] = first_frame_latent


                # Deal with ID if needed
                if use_FrameIn:
                    # Append the reference tensor to the Frame-Wise Dimension, other conditioning like Motion, should be None 
                    noisy_model_input = torch.cat([noisy_video_latents, ID_latent], dim = 2)
                else:
                    noisy_model_input = noisy_video_latents
                

                # Channel-Wise Concatenation for Noisy latents and others (channel is the dim = 1, not 2 as before)
                noisy_model_input = torch.cat([noisy_model_input, traj_latents], dim = 1)     # The channel dim 48 + 48 = 96



                # Predict the noise by Transformer model
                with torch.cuda.amp.autocast(dtype=weight_dtype):
                    noise_pred = transformer(
                                                hidden_states = noisy_model_input,
                                                timestep = timesteps,
                                                encoder_hidden_states = prompt_embeds,
                                                return_dict = False,
                                            )[0]
                

                # Drop the ID Reference part
                if use_FrameIn:
                    noise_pred = noise_pred[:, :, :num_gen_frames]       # the shape is (B, C, F, H, W)


                
                # Denoise Scheduler Process
                target = noise - video_latents
                target = target.float()
                noise_pred = noise_pred.float()


                # Loss
                loss = F.mse_loss(noise_pred, target, reduction = "mean")
                

                # Back-propagate
                accelerator.backward(loss)


                # Other Sync Process
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, 1.0)


                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()



            ##################################### Validation per XXX iterations #######################################

            # For all process, each needs to process one file
            if accelerator.gradient_state.sync_gradients and \
                ((global_step == 0 and first_iter_validation) or (global_step != 0 and global_step % validation_step == 0)):
                

                # try:
                    
                # Prepare the validation folder
                validation_store_path = os.path.join(output_dir, "validation", "ckpt" + str(global_step))
                if accelerator.is_main_process:
                    if os.path.exists(validation_store_path):
                        shutil.rmtree(validation_store_path)
                    os.makedirs(validation_store_path)

                # Run Validation
                log_validation(
                                base_model_path,
                                transformer,
                                vae,
                                text_encoder,
                                accelerator,
                                val_dataloader,
                                validation_store_path,
                                num_inference_steps,
                            )
                
                # except Exception:
                #     print("There is an exception for the validation!")
                    
            ###############################################################################################################



            # Store the Checkpoint
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1    # Skip the first term validation

                if accelerator.is_main_process or accelerator.distributed_type == DistributedType.DEEPSPEED:
                    if global_step % checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if checkpoints_total_limit is not None:
                            checkpoints = os.listdir(output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        # Save to the folder
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            # Updates
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step = global_step)

            if global_step >= max_train_steps:
                break
 

    # End training
    accelerator.end_training()




if __name__ == "__main__":
    args = get_args()

    config = OmegaConf.load(args.config_path)
    main(config, args)