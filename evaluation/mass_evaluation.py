'''
    Test all evluation with generated and gt folder
'''
import os, shutil, sys
import json
import gc
import cv2
import torch

# Import files from the local folder
root_path = os.path.abspath('.')
sys.path.append(root_path)
from evaluation.evaluate_INO_Traj import INO_Traj_evaluation
from evaluation.evaluate_INO_VSeg_MAE import INO_VSeg_MAE_evaluation
from evaluation.evaluate_INO_VLM import INO_VLM_evaluation
from evaluation.evaluate_INO_DINO import INO_DINO_evaluation



def mass_evaluation(data_parent_path, evaluation_metrics, common_target_height=256, 
                        common_target_width=384, test_num_frames=49, is_frame_in=None, store_json_path="results.json"):


    # Sanity Check
    assert(is_frame_in is not None)


    # FID FVD Setting
    Frechet_height, Frechet_width = common_target_height, common_target_width      

    
    # Evaluation
    print("Start Evaluation")
    result_collections = {}
    for evaluation_metric in evaluation_metrics:
        print("START EVALUATION of ", evaluation_metric)

        if evaluation_metric == "INO_TrajError":        # Will Download CoTracker3 automatically
            result_collections["INO_TrajError"] = INO_Traj_evaluation(data_parent_path, common_target_height, common_target_width, test_num_frames)

        elif evaluation_metric == "INO_VSeg_MAE":       # Will Download SAM2 automatically
            result_collections["INO_VSeg_MAE"] = INO_VSeg_MAE_evaluation(data_parent_path, common_target_height, common_target_width, test_num_frames)
        
        elif evaluation_metric == "Relative_DINO":      # Will Downlaod Dino-V2 automatically
            result_collections["Relative_DINO"] = INO_DINO_evaluation(data_parent_path, common_target_height, common_target_width, test_num_frames)

        elif evaluation_metric == "INO_VLM":            # Will Use Qwen2.5VL-32B from "../pretrained" folder; If not exists, will download Qwen/Qwen2.5-VL-32B-Instruct to HF cache
            result_collections["INO_VLM"] = INO_VLM_evaluation(data_parent_path, common_target_height, common_target_width, is_frame_in=is_frame_in)        # No need to write number of frames (because VLM is limited)

        else:
            raise NotImplementedError

        print("Evaluation for metrics", evaluation_metric, "is", result_collections[evaluation_metric])

        # Clean after each Run
        gc.collect()


    # Final Write the results to a json
    print("Evaluation Result is", result_collections)
    if os.path.exists(store_json_path):   # Remove at the end to avoid miss-running error
        os.remove(store_json_path)

    json_data = json.dumps(result_collections, indent=4) # Add indent
    with open(store_json_path, "w") as f: 
        f.write(json_data)




if __name__ == "__main__":

    # Basic Setting
    data_parent_path = "../FINO3/results_FrameIn"               # Path to the generated results
    common_target_height, common_target_width, test_num_frames = 256, 384, 49       # test_num_frames we use 49 for FrameIn; 14 for Frame Out (due to base model constrained) ; BTW, VLM only use 14 frames (due to compute concern)
    is_frame_in = True     # If it is not FrameIn, then it is FrameOut Setting; Please set this based on the needs


    # Evaluation Setting
    store_json_path = "results_FrameIn.json"
    evaluation_metrics = ["INO_TrajError", "INO_VSeg_MAE", "Relative_DINO", "INO_VLM"]           
    # For Frame In: "INO_TrajError", "INO_VSeg_MAE", "Relative_DINO", "INO_VLM"
    # For Frame Out: "INO_TrajError", "INO_VSeg_MAE", "INO_VLM"


    # Do the inference
    mass_evaluation(data_parent_path, evaluation_metrics, common_target_height, common_target_width, test_num_frames=test_num_frames, is_frame_in=is_frame_in, store_json_path = store_json_path)