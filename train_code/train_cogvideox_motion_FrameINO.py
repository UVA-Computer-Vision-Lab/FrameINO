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
import logging
import math
import imageio
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
from torch import nn
from torch.utils.data import ConcatDataset

import torch
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer


import diffusers
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXDPMScheduler,
)
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



# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from architecture.cogvideox_transformer_3d import CogVideoXTransformer3DModel
from pipelines.pipeline_cogvideox_i2v_motion_FrameINO import CogVideoXImageToVideoPipeline, get_resize_crop_region_for_grid
from architecture.embeddings import get_3d_rotary_pos_embed
from data_loader.sampler import MixedBatchSampler
from data_loader.video_dataset_motion_FrameINO import VideoDataset_Motion_FrameINO

if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.33.0.dev0")

logger = get_logger(__name__)
os.environ["TOKENIZERS_PARALLELISM"] = "false"



################################################################################################################################
def get_args():
    parser = argparse.ArgumentParser(description="Simple example of a ControlNet training script.")
    parser.add_argument(
                            "--config_path",
                            type=str,
                            default="config/train_cogvideox_motion_FrameINO.yaml",
                            # required=True,
                            help="Path to the config.",
                        )
    parser.add_argument(
                            "--use_8BitAdam",
                            type=bool,
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
                        use_FrameIn,
                    ):


    ################################################# The following should be the same as test_code section ##############################################

    # Create pipeline and run inference
    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
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
        first_frame_np = np.asarray(batch["first_frame_np"][0])
        text_prompt = batch["text_prompt"][0]
        traj_tensor = batch["traj_tensor"][0] 
        traj_imgs_np = np.asarray(batch["traj_imgs_np"][0])
        ID_tensor = batch["ID_tensor"][0]  
        ID_np = np.asarray(batch["ID_np"][0])
        gt_video_path = batch["gt_video_path"][0]
        merge_frames = batch["merge_frames"][0]
        

        # Fetch Generation Resulotion
        gen_height, gen_width, _ = first_frame_np.shape


        # Save the GT video
        shutil.copyfile(gt_video_path, os.path.join(store_folder_path, "gt_video.mp4"))
        
        # Convert first frame np to image form
        first_frame_path = os.path.join(store_folder_path, "first_frame.png")
        cv2.imwrite(first_frame_path, cv2.cvtColor(first_frame_np, cv2.COLOR_RGB2BGR))

        # Save the ID Reference Img
        reference_img_path = os.path.join(store_folder_path, "Main_Reference.png")
        cv2.imwrite(reference_img_path, cv2.cvtColor(ID_np, cv2.COLOR_RGB2BGR))

        # Save the traj images
        for temporal_idx, traj_img_np in enumerate(traj_imgs_np):
            traj_img_store_path = os.path.join(store_folder_path, "traj_cond" + str(temporal_idx) + ".png")
            cv2.imwrite(traj_img_store_path, cv2.cvtColor(traj_img_np, cv2.COLOR_RGB2BGR))

            # Store the merge frame for better visualization
            merge_img_store_path = os.path.join(store_folder_path, "merge_cond" + str(temporal_idx).zfill(2) + ".png")
            cv2.imwrite(merge_img_store_path, cv2.cvtColor( np.uint8(merge_frames[temporal_idx]), cv2.COLOR_RGB2BGR))
        
        # Save the video
        imageio.mimsave(os.path.join(store_folder_path, "merge_cond.mp4"), merge_frames, fps=12)

        # Save the text prompt
        with open(os.path.join(store_folder_path, 'text_prompt.txt'), 'w') as file:
            file.write(text_prompt)


        ################################################ Call Validation ################################################################

        image = load_image(first_frame_path)
        video = pipe(
                        image = image, 
                        prompt = text_prompt, 
                        traj_tensor = traj_tensor, 
                        ID_tensor = ID_tensor,
                        height = gen_height, width = gen_width,
                        guidance_scale = 6, use_dynamic_cfg = True, num_inference_steps = num_inference_steps        # Make this general setting fixed
                    ).frames[0]     # use_dynamic_cfg = True, 


        # Store the reuslt
        export_to_video(video, os.path.join(store_folder_path, "generated_video.mp4"), fps=8)

        ################################################################################################################################


        # Only consider one case
        break

    # Delete and clean
    del pipe
    torch.cuda.empty_cache()

    return



def _get_t5_prompt_embeds(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
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
        text_inputs = tokenizer(prompt,
                                padding = "max_length",
                                max_length = max_sequence_length,
                                truncation = True,
                                add_special_tokens = True,
                                return_tensors = "pt",
                                )
        text_input_ids = text_inputs.input_ids
    else:
        if text_input_ids is None:
            raise ValueError("`text_input_ids` must be provided when the tokenizer is not specified.")

    # Encode Text by T5
    prompt_embeds = text_encoder(text_input_ids.to(device))[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

    # duplicate text embeddings for each generation per prompt, using mps friendly method
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

    return prompt_embeds



def encode_prompt(
    tokenizer: T5Tokenizer,
    text_encoder: T5EncoderModel,
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



def prepare_rotary_positional_embeddings(
    height: int,
    width: int,
    num_frames: int,
    vae_scale_factor_spatial: int = 8,
    patch_size: int = 2,
    attention_head_dim: int = 64,
    device: Optional[torch.device] = None,
    base_height: int = 480,
    base_width: int = 720,
    grid_height_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    grid_height = height // (vae_scale_factor_spatial * patch_size)
    grid_width = width // (vae_scale_factor_spatial * patch_size)
    base_size_width = base_width // (vae_scale_factor_spatial * patch_size)
    base_size_height = base_height // (vae_scale_factor_spatial * patch_size)

    # HACK: if this 3D RoPE is also bounded by the 
    grid_crops_coords = get_resize_crop_region_for_grid((grid_height, grid_width), base_size_width, base_size_height)   # Different shape has different coords
    freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                                                    embed_dim = attention_head_dim,
                                                    crops_coords = grid_crops_coords,
                                                    grid_size = (grid_height, grid_width),
                                                    temporal_size = num_frames,
                                                    device = device,
                                                )

    return freqs_cos, freqs_sin



def get_optimizer(config, params_to_optimize, use_8BitAdam, use_deepspeed: bool = False):

    optimizer = config["optimizer"]
    use_8bit_adam = use_8BitAdam


    # Use DeepSpeed optimzer
    if use_deepspeed:
        from accelerate.utils import DummyOptim

        return DummyOptim(
            params_to_optimize,
            lr = config["learning_rate"],
            betas = (config["adam_beta1"], config["adam_beta2"], config["adam_beta3"]),
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



def video_tensor_to_vae_latent(video_tensor, first_frame_tensor, vae, device):

    # Preprocess
    video_tensor = video_tensor.to(device, dtype=vae.dtype)     #.unsqueeze(0)
    video_tensor = video_tensor.permute(0, 2, 1, 3, 4)  # [B, C, F, H, W]

    # Prepare the first frame
    first_frame = first_frame_tensor.unsqueeze(1).permute(0, 2, 1, 3, 4).to(device, dtype=vae.dtype).clone()

    # VAE encode
    video_latents = vae.encode(video_tensor).latent_dist

    # Add Augment Noise to the First Frame before enocode
    image_noise_sigma = torch.normal(mean = -3.0, std = 0.5, size = (1,), device = first_frame.device)
    image_noise_sigma = torch.exp(image_noise_sigma).to(dtype = first_frame.dtype)
    noisy_image = first_frame + torch.randn_like(first_frame) * image_noise_sigma[:, None, None, None, None]

    # First frame VAE encode
    first_frame_latent = vae.encode(noisy_image).latent_dist



    ########################## Originally, codes below are from collate_fn ############################
    
    # Multiply with the scaling factor and then reshape
    video_latents = video_latents.sample() * vae.config.scaling_factor
    first_frame_latent = first_frame_latent.sample() * vae.config.scaling_factor
    video_latents = video_latents.permute(0, 2, 1, 3, 4)
    first_frame_latent = first_frame_latent.permute(0, 2, 1, 3, 4)

    # Add the padding
    padding_shape = (video_latents.shape[0], video_latents.shape[1] - 1, *video_latents.shape[2:])
    latent_padding = first_frame_latent.new_zeros(padding_shape)
    first_frame_latent = torch.cat([first_frame_latent, latent_padding], dim=1) # The first frame is padded with zero, Not Repeat

    # Random Full Zero mask
    if random.random() < config["noised_image_dropout"]:
        first_frame_latent = torch.zeros_like(first_frame_latent)

    # Final Convert
    video_latents = video_latents.to(memory_format = torch.contiguous_format).float()
    first_frame_latent = first_frame_latent.to(memory_format = torch.contiguous_format).float()


    # Return the image and video latents
    return video_latents, first_frame_latent 


def traj_tensor_to_vae_latent(traj_tensor, vae, device):

    # Preprocess
    traj_tensor = traj_tensor.to(device, dtype=vae.dtype)     #.unsqueeze(0)
    traj_tensor = traj_tensor.permute(0, 2, 1, 3, 4)  

    # VAE encode
    traj_latents = vae.encode(traj_tensor).latent_dist

    # Scale, Permute, and other conversion
    traj_latents = traj_latents.sample() * vae.config.scaling_factor
    traj_latents = traj_latents.permute(0, 2, 1, 3, 4)
    traj_latents = traj_latents.to(memory_format = torch.contiguous_format).float()

    return traj_latents


def img_tensor_to_vae_latent(img_tensor, vae, device, add_augment_noise = True):


    # Preprocess
    img_tensor = img_tensor.unsqueeze(2).to(device, dtype=vae.dtype)         # Must have 5 dim for the vae to work: [B, C, F, H, W]


    # Add Augment Noise to the image (ID Refernece) before enocode  (NOTE: this is following Concat-ID) But no augment noise in inference
    if add_augment_noise:
        image_noise_sigma = torch.normal(mean = -3.0, std = 0.5, size = (1,), device = img_tensor.device)
        image_noise_sigma = torch.exp(image_noise_sigma).to(dtype = img_tensor.dtype)
        img_tensor = img_tensor + torch.randn_like(img_tensor) * image_noise_sigma[:, None, None, None, None]

    # VAE encode
    image_latent = vae.encode(img_tensor).latent_dist



    ########################## Originally, codes below are from collate_fn ############################
    
    # Multiply with the scaling factor and then reshape
    image_latent = image_latent.sample() * vae.config.scaling_factor
    image_latent = image_latent.squeeze(2)      # Make the shape to [B, C, H, W] no dim on F
    # NOTE: No padding in this branch


    # Final Convert
    image_latent = image_latent.to(memory_format = torch.contiguous_format).float()


    # Return the image and video latents
    return image_latent

    
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
    enable_slicing = config["enable_slicing"]
    enable_tiling = config["enable_tiling"]
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
    variant = config["variant"]
    lr_scheduler = config["lr_scheduler"]
    use_rotary_positional_embeddings = config["use_rotary_positional_embeddings"],
    use_learned_positional_embeddings = config["use_learned_positional_embeddings"]
    validation_step = config["validation_step"]
    first_iter_validation = config["first_iter_validation"]
    num_inference_steps = config["num_inference_steps"]

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

    text_encoder = T5EncoderModel.from_pretrained(
                        base_model_path, 
                        subfolder = "text_encoder", 
                        torch_dtype = torch.bfloat16,  
                        # quantization_config = TorchAoConfig("int8wo"),
                    )



    # HACK: Only load BF16 in 5B model, else load FP16
    # Read the CogVideoX model based on if the pretrained_transformer_path is available
    if pretrained_transformer_path is not None:
        print("Load Pretrained Transformer model from ", pretrained_transformer_path)
        transformer_model_path = pretrained_transformer_path
    else:
        transformer_model_path = base_model_path


    vae = AutoencoderKLCogVideoX.from_pretrained(
                                                    base_model_path, 
                                                    subfolder = "vae", 
                                                    torch_dtype = torch.bfloat16,  
                                                )
    vae_scale_factor_spatial = 2 ** (len(vae.config.block_out_channels) - 1)


    # Load Transformer with some custom setting
    transformer = CogVideoXTransformer3DModel.from_pretrained(
                                                                transformer_model_path,
                                                                torch_dtype = torch.bfloat16,
                                                                use_FrameIn = use_FrameIn,     
                                                            )

    # NOTE: Here we don't need to add a new channel dim, which is already updated in the weights in the model


    # Load Scheduler
    scheduler = CogVideoXDPMScheduler.from_pretrained(base_model_path, subfolder="scheduler")

    
    # VAE process
    if enable_slicing:
        vae.enable_slicing()
    if enable_tiling:
        vae.enable_tiling()


    # We only train the additional adapter LoRA layers
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

    text_encoder.to(accelerator.device)     # , dtype = weight_dtype
    transformer.to(accelerator.device)      # HACK: whether transformer use BF16 didn't speed up a lot    
    vae.to(accelerator.device)

    if gradient_checkpointing:
        transformer.enable_gradient_checkpointing()


    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):
        # breakpoint()
        if accelerator.is_main_process:

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "transformer"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()


    def load_model_hook(models, input_dir):

        transformer_ = None
        # breakpoint()

        while len(models) > 0:
            model = models.pop()

            load_model = CogVideoXTransformer3DModel.from_pretrained(input_dir, subfolder = "transformer",)
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
        # if name.find("transformer_blocks"):       # This will print all parameters that is not named with transformer_blocks
        #     print(name)
    print("Total parameters that will be trained has ", total_params_for_training/(10**9), "B param")


    # Optimization parameters
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
    
    # Find all dataset
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
                                )        # NOTE: Don't use shuffle; else, there may be a


    # Prepare the validation dataloader
    val_dataset = VideoDataset_Motion_FrameINO(
                                                    config, download_folder_path, validation_csv_relative_path, 
                                                    config["validation_video_relative_path"], config["validation_ID_relative_path"],
                                                )
    val_dataloader = DataLoader(
                                    val_dataset,
                                    batch_size = 1,
                                    shuffle = False,
                                    num_workers = 1,
                                )   # No need for the collate

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
                                        num_cycles = config["lr_num_cycles"],
                                        power = config["lr_power"],
                                    )


    # Prepare everything with our `accelerator`.
    transformer, optimizer, lr_scheduler = accelerator.prepare(
                                                transformer, optimizer, lr_scheduler
                                            )
    if not debug:
        train_dataloader = accelerator.prepare(train_dataloader)


    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_name = config["tracker_name"] or "CogvideoX-I2V"
        accelerator.init_trackers(tracker_name, config=vars(args))

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps


    logger.info("***** Running training *****")
    logger.info(f"  Num trainable parameters = {total_params_for_training}")
    logger.info(f"  Num videos = {len(concat_dataset)}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
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


    # Read the model config
    model_config = transformer.module.config if hasattr(transformer, "module") else transformer.config

    for epoch in range(first_epoch, num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            models_to_accumulate = [transformer]

            with accelerator.accumulate(models_to_accumulate):

                # Fetch the info
                video_tensor = batch["video_tensor"]
                traj_tensor = batch["traj_tensor"]
                prompts = batch["prompts"]
                first_frame_tensor = batch["first_frame_tensor"]
                if use_FrameIn:
                    ID_tensor = batch["ID_tensor"]
                batch_size, pixel_num_frames, num_channels, pixel_height, pixel_width = video_tensor.shape
                    
                
                # Encode video and traj tensor to latents
                video_latents, first_frame_latent = video_tensor_to_vae_latent(video_tensor, first_frame_tensor, vae, accelerator.device)
                traj_latents = traj_tensor_to_vae_latent(traj_tensor, vae, accelerator.device)
                batch_size, num_gen_frames, num_channels, latent_height, latent_width = video_latents.shape       # NOTE: now we set the batch size to be 1, to adapt arbitrary resolution inputs


                # Refernece Tensor Adjustment for FrameIn cases
                if use_FrameIn:

                    # VAE encode the Reference Latents
                    ID_latent = img_tensor_to_vae_latent(ID_tensor, vae, accelerator.device)        # Will add augment noise like the first frame
                    ID_latent = ID_latent.unsqueeze(1)

                    # Increase the frame dimension of the Traj latents and the first frame latent
                    reference_frame_latent_padding = video_latents.new_zeros(ID_latent.shape)                   # Zero latent values
                    first_frame_latent = torch.cat([first_frame_latent, reference_frame_latent_padding], dim=1)     
                    traj_latents = torch.cat([traj_latents, reference_frame_latent_padding], dim=1)



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

                # Cast to correct weight dtpye
                video_latents = video_latents.to(dtype = weight_dtype)              # [B, F, C, H, W]       Might +1  for the ID reference
                traj_latents = traj_latents.to(dtype = weight_dtype)                # [B, F, C, H, W]       Might +1  for the ID reference
                first_frame_latent = first_frame_latent.to(dtype = weight_dtype)    # [B, F, C, H, W]       Might +1  for the ID reference


                # Sample a random timestep for each image
                timesteps = torch.randint(
                                            0, scheduler.config.num_train_timesteps, (video_latents.shape[0],), device = video_latents.device
                                        )
                timesteps = timesteps.long()


                # Add noise to the model input according to the noise magnitude at each timestep
                noise = torch.randn_like(video_latents) # Sample noise that will be added to the latents
                noisy_video_latents = scheduler.add_noise(video_latents, noise, timesteps)

                # Adjust the new video latents after frame-wise concatenation
                # batch_size, num_frames, num_channels, latent_height, latent_width = video_latents.shape     # num_frames should be 14 for FrameIn case


                if use_FrameIn:
                    # Append the reference tensor to the Frame-Wise Dimension, other conditioning like Motion, should be None 
                    # On the NOISE dimension, not the Img-Conditioned Dimension
                    noisy_model_input = torch.cat([noisy_video_latents, ID_latent], dim=1)
                else:
                    noisy_model_input = noisy_video_latents


                # Channel-Wise Concatenation
                noisy_model_input = torch.cat([noisy_model_input, first_frame_latent, traj_latents], dim=2)


                # Prepare rotary embeds
                image_rotary_emb = (
                                        prepare_rotary_positional_embeddings(
                                                                                height = pixel_height,       # NOTE: Original resolution, not latent resolution
                                                                                width = pixel_width,
                                                                                num_frames = num_gen_frames,      # Still 13 frames
                                                                                vae_scale_factor_spatial = vae_scale_factor_spatial,
                                                                                patch_size = model_config.patch_size,
                                                                                attention_head_dim = model_config.attention_head_dim,
                                                                                device = accelerator.device,
                                                                            )
                                        if model_config.use_rotary_positional_embeddings
                                        else None
                                    )


                # Copy the 14th frame with the first frame PE information
                freqs_cos, freqs_sin = image_rotary_emb
                first_frame_token_num = freqs_cos.shape[0] // num_gen_frames
                freqs_cos = torch.cat([freqs_cos, freqs_cos[:first_frame_token_num].clone()], dim=0)      # Hard Code
                freqs_sin = torch.cat([freqs_sin, freqs_sin[:first_frame_token_num].clone()], dim=0)
                image_rotary_emb = (freqs_cos, freqs_sin)



                # Predict the noise by Transformer model
                model_output = transformer(
                                            hidden_states = noisy_model_input,
                                            encoder_hidden_states = prompt_embeds,
                                            timestep = timesteps,
                                            image_rotary_emb = image_rotary_emb,
                                            return_dict = False,
                                        )[0]


                # Drop the ID Reference part
                if use_FrameIn:
                    model_output = model_output[:, :num_gen_frames]


    
                # Denoise Scheduler Process
                model_pred = scheduler.get_velocity(model_output, noisy_video_latents, timesteps)
                

                alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                weights = 1 / (1 - alphas_cumprod)
                while len(weights.shape) < len(model_pred.shape):
                    weights = weights.unsqueeze(-1)

                # Target latents
                target = video_latents
                

                loss = torch.mean((weights * (model_pred - target) ** 2).reshape(batch_size, -1), dim=1)
                loss = loss.mean()
                accelerator.backward(loss)


                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, config["max_grad_norm"])

                if accelerator.state.deepspeed_plugin is None:
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
                                use_FrameIn,
                            )

                # except Exception:
                #     print("There is an exception for the validation!")
            ###############################################################################################################


            # Checks if the accelerator has performed an optimization step behind the scenes
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

                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        # Save to the folder
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

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