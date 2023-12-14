import os 
os.environ['CUDA_VISIBLE_DEVICES'] ="0"
import time
from tqdm import tqdm 
import torch
import pandas as pd 
from utils import concat_prompt, log_info

from kandinsky2 import get_kandinsky2
from extract_keyword_fast import extract_keywords
from pad_color import pad
import post_processing

#=======================================
# Initializ models
#=======================================
model = get_kandinsky2('cuda', model_version="2.0", cache_dir ="/code/weights", task_type='text2img')
#=======================================
# Generate
#=======================================

def generate_banners(df_info, output_dir, seed):
    all_predicted_time = []
    
    torch.manual_seed(seed)
    for idx, row in tqdm(df_info.iterrows(), total = len(df_info)):
        t1 = time.time()
        
        # ***************Start model prediction******************
        vie_full_info, is_empty = concat_prompt(guide_text="", prompts=[row["caption"],row["description"],row["moreInfo"]])
        start_ = time.perf_counter()
        keyword_dict = extract_keywords(vie_full_info, temp=0.1)
        qwen_time = time.perf_counter() - start_
        df_info.at[idx, "qwen7B_time"] = str(qwen_time)
        try:
            vie_prompt = keyword_dict["product_type"]+". "+keyword_dict["description"][:100]
        except Exception as e:
            df_info.at[idx, "error"] = str(e)
            vie_prompt = vie_full_info
        df_info.at[idx, "prompt"] = vie_prompt
        start_ = time.perf_counter()
        images_generated = model.generate_text2img(vie_prompt, 
                                                batch_size=2,
                                                h=512, 
                                                w=512, 
                                                num_steps=30, 
                                                denoised_type='dynamic_threshold', 
                                                dynamic_threshold_v=99.5, 
                                                sampler='ddim_sampler', 
                                                ddim_eta=0.05, 
                                                guidance_scale=10,
                                                progress=False)
        k20_time = time.perf_counter() - start_
        df_info.at[idx, "K2.0_time"] = str(k20_time)
        
        generated_path = output_dir +"/"+row["bannerImage"]

        img1 = post_processing.scale_rectangle(images_generated[0], h_scale_ratio=0.8, w_scale_ratio=1.0)
        img2 = post_processing.scale_rectangle(images_generated[1], h_scale_ratio=0.8, w_scale_ratio=1.0)
        post_processed_image = post_processing.concat_images(img1, img2, middle_space=10)
        post_processed_image = post_processing.blend(post_processed_image, ratio=0.8)
        post_processed_image = post_processing.padding(post_processed_image)    
        post_processed_image.save(generated_path)
        # ***************End model prediction******************
        
        t2 = time.time()
        predicted_time = t2 - t1
        all_predicted_time.append((row["bannerImage"], predicted_time))
    
    return all_predicted_time
    
if __name__ == "__main__":
    start = time.perf_counter()
    
    df_info = pd.read_csv("/data/private/info.csv")
    df_info.fillna("", inplace=True)
    df_info["prompt"] = ""
    df_info["error"] = ""
    df_info["qwen7B_time"] = ""
    df_info["K2.0_time"] = ""
    
    output_dir_1 = '/results/submission1'
    output_dir_2 = '/results/submission2'
    print(f"Output to {output_dir_1}")
    print(f"Output to {output_dir_2}")

    os.makedirs(output_dir_1, exist_ok=True)
    os.makedirs(output_dir_2, exist_ok=True)
    generate_banners(df_info, output_dir=output_dir_1, seed=2023456789)
    generate_banners(df_info, output_dir=output_dir_2, seed=2123456789)

    end = time.perf_counter()
