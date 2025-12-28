import torch
import cv2
import json
import os
import numpy as np
from PIL import Image
import argparse
import pandas as pd
import torch.nn.functional as F
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.regression import MeanSquaredError
from urllib.request import urlretrieve 
import open_clip
import hpsv2
import ImageReward as RM
import math
from tqdm import tqdm

def rle2mask(mask_rle, shape): # height, width
    starts, lengths = [np.asarray(x, dtype=int) for x in (mask_rle[0:][::2], mask_rle[1:][::2])]
    starts -= 1
    ends = starts + lengths
    binary_mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        binary_mask[lo:hi] = 1
    return binary_mask.reshape(shape)

class MetricsCalculator:
    def __init__(self, device) -> None:
        self.device = device
        # clip
        print("Loading CLIPScore metric...")
        self.clip_metric_calculator = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
        # lpips
        print("Loading LPIPS metric...")
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        
        # aesthetic model
        print("Loading Aesthetic model...")
        self.aesthetic_model = torch.nn.Linear(768, 1)
        aesthetic_model_url = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_vit_l_14_linear.pth?raw=true"
        )
        # Save to local directory to avoid repeated downloads
        ckpt_path = "aesthetic_model.pth"
        if not os.path.exists(ckpt_path):
            print(f"Downloading aesthetic model to {ckpt_path}...")
            urlretrieve(aesthetic_model_url, ckpt_path)
            
        self.aesthetic_model.load_state_dict(torch.load(ckpt_path))
        self.aesthetic_model.to(device)
        self.aesthetic_model.eval()
        
        print("Loading OpenCLIP for Aesthetic score...")
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
        self.clip_model.to(device) # Ensure clip model is on device
        
        # image reward model
        print("Loading ImageReward...")
        try:
            self.imagereward_model = RM.load("ImageReward-v1.0", device=device)
        except Exception as e:
            print(f"Warning: Failed to load ImageReward: {e}")
            self.imagereward_model = None

    def calculate_image_reward(self, image, prompt):
        if self.imagereward_model is None:
            return 0.0
        # ImageReward expects PIL or list of PIL
        reward = self.imagereward_model.score(prompt, [image])
        return reward

    def calculate_hpsv21_score(self, image, prompt):
        # hpsv2.score expects generated image path or PIL
        # It handles device internally typically, or uses cuda if available
        try:
            result = hpsv2.score(image, prompt, hps_version="v2.1")[0]
            return result.item()
        except Exception as e:
            print(f"HPSv2 Error: {e}")
            return 0.0

    def calculate_aesthetic_score(self, img):
        # img: PIL Image
        image = self.clip_preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            # aesthetic_model is a simple linear layer, expects float32
            prediction = self.aesthetic_model(image_features.float())
        return prediction.cpu().item()

    def calculate_clip_similarity(self, img, txt):
        # img: PIL Image
        img = np.array(img) # HWC
        # torchmetrics CLIPScore expects (N, C, H, W) in [0, 255] usually?
        # Actually torchmetrics CLIPScore: 
        # "images (Tensor): images to calculate score for. Expected to be of shape (N, C, H, W) or (C, H, W) and have values in the [0, 255] range."
        
        img_tensor = torch.tensor(img).permute(2,0,1).to(self.device) # C, H, W
        
        score = self.clip_metric_calculator(img_tensor, txt)
        score = score.cpu().item()
        
        return score
    
    def calculate_psnr(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32)/255.
        img_gt = np.array(img_gt).astype(np.float32)/255.

        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask
        
        difference = img_pred - img_gt
        difference_square = difference ** 2
        difference_square_sum = difference_square.sum()
        
        if mask is not None:
            difference_size = mask.sum()
        else:
            difference_size = img_pred.size # Total elements
            
        # Avoid division by zero
        if difference_size == 0:
            return 0.0

        mse = difference_square_sum/difference_size

        if mse < 1.0e-10:
            return 100.0 # Cap max PSNR
            
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    
    def calculate_lpips(self, img_gt, img_pred, mask=None):
        img_pred = np.array(img_pred).astype(np.float32)/255
        img_gt = np.array(img_gt).astype(np.float32)/255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask 
            img_gt = img_gt * mask
            
        img_pred_tensor = torch.tensor(img_pred).permute(2,0,1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2,0,1).unsqueeze(0).to(self.device)
            
        # LPIPS expects input in [-1, 1] usually, but torchmetrics implementation documentation says:
        # "Expected to be of shape (N, C, H, W) and have values in the [0, 1] range." -> Check specific version?
        # Re-checking evaluate_brushnet.py: it uses `img_pred_tensor*2-1`. 
        # Torchmetrics LPIPS docs say "input images should be in range [0, 1]".
        # BUT evaluate_brushnet.py multiplies by 2 and subtracts 1 (mapping to [-1, 1]).
        # I will follow evaluate_brushnet.py exactly to ensure consistency.
        
        score = self.lpips_metric_calculator(img_pred_tensor*2-1, img_gt_tensor*2-1)
        score = score.cpu().item()
        
        return score
    
    def calculate_mse(self, img_pred, img_gt, mask=None):
        img_pred = np.array(img_pred).astype(np.float32)/255.
        img_gt = np.array(img_gt).astype(np.float32)/255.

        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."
        if mask is not None:
            mask = np.array(mask).astype(np.float32)
            img_pred = img_pred * mask
            img_gt = img_gt * mask
        
        difference = img_pred - img_gt
        difference_square = difference ** 2
        difference_square_sum = difference_square.sum()
        
        if mask is not None:
             difference_size = mask.sum()
        else:
             difference_size = img_pred.size

        if difference_size == 0:
            return 0.0

        mse = difference_square_sum/difference_size

        return mse.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True, help="Path to generated images")
    parser.add_argument("--gt_dir", type=str, required=True, help="Path to ground truth images")
    parser.add_argument("--prompt_json", type=str, help="JSON file mapping filenames to prompts/masks")
    parser.add_argument("--mask_key", type=str, default="inpainting_mask", help="Key for mask in json")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    metrics_calculator = MetricsCalculator(device)
    
    # Load mapping file
    mapping = {}
    if args.prompt_json and os.path.exists(args.prompt_json):
        try:
            with open(args.prompt_json, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
        except Exception as e:
            print(f"Error loading prompt json: {e}")
            
    files = os.listdir(args.results_dir)
    image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    
    evaluation_df = pd.DataFrame(columns=['Image ID', 'Image Reward', 'HPS V2.1', 'Aesthetic Score', 'PSNR', 'LPIPS', 'MSE', 'CLIP Similarity'])
    
    print(f"Found {len(image_files)} images. Starting evaluation...")
    
    for i, filename in enumerate(tqdm(image_files)):
        # print(f"Processing {filename} ({i+1}/{len(image_files)})...")
        
        # Determine Keys
        key = filename
        if key not in mapping:
            key_no_ext = os.path.splitext(filename)[0]
            if key_no_ext in mapping:
                key = key_no_ext
                
        # Get Prompt and Mask
        prompt = ""
        mask_rle = None
        if key in mapping:
            item = mapping[key]
            # Handle prompt/caption
            if isinstance(item, dict):
                prompt = item.get("caption", item.get("prompt", ""))
                mask_rle = item.get(args.mask_key, None)
            else:
                prompt = str(item)
        
        # Load Images
        res_path = os.path.join(args.results_dir, filename)
        
        # Find GT
        gt_path = os.path.join(args.gt_dir, filename)
        if not os.path.exists(gt_path):
             base_name = os.path.splitext(filename)[0]
             for ext in ['.png', '.jpg', '.jpeg']:
                 if os.path.exists(os.path.join(args.gt_dir, base_name + ext)):
                     gt_path = os.path.join(args.gt_dir, base_name + ext)
                     break
        
        if not os.path.exists(gt_path):
            print(f"GT not found for {filename}, skipping paired metrics.")
            continue
            
        try:
            # Resize to 512x512 as in BrushNet evaluation to be consistent
            tgt_image = Image.open(res_path).convert("RGB").resize((512,512))
            src_image = Image.open(gt_path).convert("RGB").resize((512,512))
        except Exception as e:
            print(f"Error loading images for {filename}: {e}")
            continue

        # Prepare Mask
        mask = None
        if mask_rle is not None:
            # BrushNet logic: mask = 1 - mask (to focus on inpainting area? or keep area?)
            # In evaluate_brushnet.py:
            # mask = rle2mask(mask,(512,512))
            # mask = 1 - mask[:,:,np.newaxis]
            # calculate_psnr(src_image, tgt_image, mask)
            # If 1 is masked (to be inpainted), then 1-mask means 0 is inpainted area??
            # Usually we want to measure error in the inpainted area?
            # Or maybe they measure reconstruction of the WHOLE image or just the hole?
            # Let's trust evaluate_brushnet.py logic for consistency.
            try:
                mask_arr = rle2mask(mask_rle, (512, 512))
                mask_arr = 1 - mask_arr[:, :, np.newaxis] # Shape (512, 512, 1)
                mask = mask_arr
            except Exception as e:
                print(f"Error processing mask for {filename}: {e}")
                mask = None
        
        evaluation_result = [filename]
        
        # Calculate Metrics
        # 1. Image Reward
        ir = metrics_calculator.calculate_image_reward(tgt_image, prompt)
        evaluation_result.append(ir)
        
        # 2. HPS v2.1
        hps = metrics_calculator.calculate_hpsv21_score(tgt_image, prompt)
        evaluation_result.append(hps)
        
        # 3. Aesthetic Score
        aes = metrics_calculator.calculate_aesthetic_score(tgt_image)
        evaluation_result.append(aes)
        
        # 4. PSNR
        psnr = metrics_calculator.calculate_psnr(src_image, tgt_image, mask)
        evaluation_result.append(psnr)
        
        # 5. LPIPS
        lpips_val = metrics_calculator.calculate_lpips(src_image, tgt_image, mask)
        evaluation_result.append(lpips_val)
        
        # 6. MSE
        mse = metrics_calculator.calculate_mse(src_image, tgt_image, mask)
        evaluation_result.append(mse)
        
        # 7. CLIP Similarity
        clip_sim = metrics_calculator.calculate_clip_similarity(tgt_image, prompt)
        evaluation_result.append(clip_sim)
        
        evaluation_df.loc[len(evaluation_df.index)] = evaluation_result

    print("\nThe averaged evaluation result:")
    averaged_results = evaluation_df.mean(numeric_only=True)
    print(averaged_results)
    
    sum_csv_path = os.path.join(args.results_dir, "metric_summary_new.csv")
    detail_csv_path = os.path.join(args.results_dir, "metric_details_new.csv")
    
    averaged_results.to_csv(sum_csv_path)
    evaluation_df.to_csv(detail_csv_path)
    
    print(f"Saved results to {sum_csv_path} and {detail_csv_path}")
