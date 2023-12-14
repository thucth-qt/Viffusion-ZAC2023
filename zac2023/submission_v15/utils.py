from typing import List
import torch 
from datetime import datetime
def normalize_sentence(prompt):
    """Removes dots at the beginning and end of a string"""
    return str(prompt).strip().strip('.').strip()

def concat_prompt(guide_text="hình ành chất lượng cao, hình chụp", prompts:List[str]= [], sep = ". "):
    #check empty prompts 
    prompts_norm = [normalize_sentence(prompt) for prompt in prompts]
    prompts_norm = [prompt for prompt in prompts_norm if len(prompt)>0]
    if len(prompts_norm)==0:
        return "", True
    guide_text=normalize_sentence(guide_text) 
    final_prompt = sep.join([guide_text]+prompts+[""])
    return final_prompt.strip(), False

def log_info(output_dir, start, end):
    with open(output_dir+"/run_info.txt","w") as f:
        current_time = datetime.now().strftime("%H:%M:%S")  # Format: hours:minutes:seconds
        current_date = datetime.now().strftime("%Y-%m-%d")  # Format: year-month-day
        device = torch.cuda.current_device()
        gpu_properties = torch.cuda.get_device_properties(device)
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1e9  # Convert bytes to GB
        gpu_name = gpu_properties.name
        memory_allocated = torch.cuda.memory_allocated(device) / 1024 / 1024  # Convert bytes to megabytes
        memory_cached = torch.cuda.memory_cached(device) / 1024 / 1024  # Convert bytes to megabytes


        f.write(f"Current Date: {current_date}\n")
        f.write(f"Current Time: {current_time}\n\n")
        f.write(f"GPU Name: {gpu_name}\n")
        f.write(f"GPU Memory: {gpu_memory:.2f} GB\n")
        f.write(f"GPU Memory Allocated: {memory_allocated:.2f} MB\n")
        f.write(f"GPU Memory Cached: {memory_cached:.2f} MB\n\n")
        f.write(f"Inference time + write output: {end-start}")