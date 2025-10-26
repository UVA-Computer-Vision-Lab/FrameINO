
<p align="center">
    <img src="__assets__/logo.png" height="100">
</p>


# Frame In-N-Out: Unbounded Controllable Image-to-Video Generation (NeurIPS 2025)

<div align="center">

[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2505.21491)
[![Website](https://img.shields.io/badge/Project-Website-pink?logo=googlechrome&logoColor=white)](https://uva-computer-vision-lab.github.io/Frame-In-N-Out/)
<a href="https://huggingface.co/collections/uva-cv-lab/frame-in-n-out"><img src="https://img.shields.io/static/v1?label=%F0%9F%A4%97%20Hugging%20Face&message=Model+with+Data&color=orange"></a>

</div>



We propose Frame In-N-Out, a controllable Image-to-Video generation Diffusion Transformer model where objects can enter or exit the scene along user-specified motion trajectories and ID reference. Our method introduces a new dataset curation pattern recognition, evaluation protocol, and a **motion-controllable**, **identity-preserving**, **unbounded canvas** Video Diffusion Transformer, to achieve Frame In and Frame Out in the cinematic domain.



🔥 [Update](#Update) **|** 👀 [**Visualization**](#Visualization)  **|** 🔧 [Installation](#installation) **|** ⚡ [Test](#fast_inference)  **|** 🧩 [Dataset Curation](#dataset_curation)  **|** 🔥[Train](#training)  **|** 💻 [evaluation](#evaluation)



## <a name="Update"></a> Update 🔥🔥🔥
- [x] Release the paper
- [x] Release the paper weights (CogVideoX-5B Stage1 Motion + Stage2 Motion with In-N-Out capbility)
- [x] Release the improved model weights (Wan2.2-5B on higher resolution, more datasets, and improved curation)
- [x] Gradio App demo 
- [x] Release the Evaluation Code and Metrics
- [x] Release the Training Code with a short sample dataset
- [ ] HF Space Demo.
- [ ] Release A variable resolution trained weights as V1.6.
- [ ] Release the Pre-Processing Code and possibly the full Processed metadata.

:star: **If you like Frame In-N-Out, please help star this repo. Thanks!** :hugs:





## <a name="Visualization"></a> Brief Intro Video 👀
---

https://github.com/user-attachments/assets/0fabd2a4-9d3b-4148-bc04-6fc03c53caca

---






## <a name="installation"></a> Installation 🔧
```shell
conda create -n FINO python=3.10
conda activate FINO
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
conda install ffmpeg
pip install -r requirements.txt
```







## <a name="fast_inference"></a> Fast Inference ⚡⚡⚡
Gradio Interactive demo is available by 
```shell
  python app.py
```
<!-- The Gradio Demo online is availabe at: -->
NOTE: This will automatically download pretrained weight to the HF cache and use our v1.5 Wan2.2-5B weight by default.
NOTE: We recommend to open **Running on public URL:** choice, which is more stable compared to the local URL option.






## <a name="dataset_curation"></a> Dataset Curation 🧩

TBD. We might use a separate github repo to collect all solutions because curation involves too many different packages and setup.

For a small quick mini-dataset (including training + validation + evaluation), you can download by: 
```shell
  # Recommend to set --local-dir as ../FrameINO_data, which is the default fixed dir in most files
  hf download uva-cv-lab/FrameINO_data --repo-type dataset --local-dir ../FrameINO_data
```





## <a name="training"></a> Train 🔥 

Though we provide a short sample training dataset (~300 videos), the full dataset needs to be prepared by yourself. 
This is just for illustration and as an example.

The training is slightly different on the dataloader part from what is stated in the paper. 
We also modified and trained a Wan2.2-5B version. 
This is because we found that WAN2.2-5B might be an interesting model and we spent quite a lot of time after the submission to optimize the training and, more importantly, curation stage. We prefer the version presented below and will be based on this.


### Download Pretrained weight 

```shell
huggingface-cli download THUDM/CogVideoX-5b-I2V --local-dir ../pretrained/CogVideoX_5B_I2V      # CogVideoX-5B

huggingface-cli download Wan-AI/Wan2.2-TI2V-5B-Diffusers --local-dir ../pretrained/Wan2.2-TI2V-5B-Diffusers     # Wan2.2-5B
```
If you download a different directory, you might need to edit **base_model_path** from **config/XXX.yaml**.
Meanwhile, **config/XXX.yaml** control all paramters needed to change. Please read it before executing the following code.


### Stage1 Motion Training 

For Wan2.2-5B:
```shell
# 1 GPU
python train_code/train_wan_motion.py

# 4GPU (Our experiment Setting). Change the XXXXX to your port (like 32214)
accelerate launch --config_file config/accelerate_config_4GPU.json --main_process_port XXXXX train_code/train_wan_motion.py
```

For CogVideoX:
```shell
# 1 GPU
python train_code/train_cogvideox_motion.py

# 4GPU (Our experiment Setting).  Change the XXXXX to your port (like 32214)
accelerate launch --config_file config/accelerate_config_4GPU.json --main_process_port XXXXX train_code/train_cogvideox_motion.py
```

Use **--use_8BitAdam True** for 8Bit Adam (based on your hardware support)



### Stage2 Frame In-N-Out Training (Motion + Unbounded Canvas + ID reference)

For Wan2.2-5B:
```shell
# 1 GPU
python train_code/train_wan_motion_FrameINO.py    

# 4GPU (Our experiment Setting).  Change the XXXXX to your port (like 32214)
accelerate launch --config_file config/accelerate_config_4GPU.json --main_process_port XXXXX train_code/train_wan_motion_FrameINO.py
```


For CogVideoX:
```shell
# 1 GPU
python train_code/train_cogvideox_motion_FrameINO.py    

# 4GPU (Our experiment Setting).  Change the XXXXX to your port (like 32214)
accelerate launch --config_file config/accelerate_config_4GPU.json --main_process_port XXXXX train_code/train_cogvideox_motion_FrameINO.py
```

Use **--use_8BitAdam True** for 8Bit Adam (based on your hardware support)



## <a name="evaluation"></a> Evaluation 💻

The evaluation dataloader is slightly different from the training version before. 
The dataloader we use in this stage is based on our paper setting (using the v1.0 paper weight at the same time). 
<!-- As stated in the previous section: after submission, we improve quite a lot on the dataset curaiton and this leads to slight difference. -->

For Frame In:
```shell
  python test_code/run_cogvideox_FrameIn_mass_evaluation.py
```

 
For Frame Out:
```shell
  python test_code/run_cogvideox_FrameOut_mass_evaluation.py
```


For the evaluation metrics, we provide our modified version of Traj Error, Video Segmentation on Mean Absolute Error, Relative DINO matching, and VLM-judged In-N-Out success rate.
Check **evaluation/mass_evalution.py** and then modify the setting there (like the number of frames, path, metrics for Frame In/Out) based on the needs.






## Disclaimer
This project is released for academic use only. We disclaim responsibility for the distribution of the model weight and sample data. Users are solely liable for their actions. The project contributors are not legally affiliated with, nor accountable for, users' behaviors.


## 📚 Citation
```bibtex
@article{wang2025frame,
  title={Frame In-N-Out: Unbounded Controllable Image-to-Video Generation},
  author={Wang, Boyang and Chen, Xuweiyi and Gadelha, Matheus and Cheng, Zezhou},
  journal={arXiv preprint arXiv:2505.21491},
  year={2025}
}
```

## 🤗 Acknowledgment
The current version of **Frame In-N-Out** is built on [diffusers](https://github.com/huggingface/diffusers).
We appreciate the authors for sharing their awesome codebase.
