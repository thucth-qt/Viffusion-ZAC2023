from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from parse_json import parse_json_wrapper
# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained("/code/weights/qwen_vie_8bit_fast", trust_remote_code=True,  device_map="cuda", padding="left", )

model = AutoGPTQForCausalLM.from_quantized(
    "/code/weights/qwen_vie_8bit_fast",
    device_map="auto",
    trust_remote_code=True,
    use_safetensors=True,
).eval()

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, tokenizer, stop_words = [], encounters=1):
        super().__init__()
        self.stop_words = stop_words
        self.stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop_id,  stop_word in zip(self.stop_words_ids,self.stop_words):
            if stop_id in input_ids[0]:
                return True

        return False

stop_words = ["<|endoftext|>"]
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stop_words=stop_words, tokenizer=tokenizer)])    


DEFAULT_PROMPT = """<|im_start|>system\nYou are a helpful assistant. You were developed by VillaLabs with the purpose of understanding the Vietnamese language to provide detailed, polite, and helpful answers to questions from humans in Vietnamese.<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"""

USER_MESSAGE = """You are a good assistant in extracting key information from ads in Vietnamese. Your task is to extract product categories from ads or infering product categories that are not directly mentioned in the content.
Ads: "{ads}"
Your response must be in JSON format:
{{
"product_type": "", // product type mentioned in content, this should be noun, you infer product type if not mentioned
"description": "" // description of appearance including shape, colors, materials, details, features, reference images.
}}
"""

def extract_keywords(ads, temp=0.001):
    prompt = DEFAULT_PROMPT.format(user_message=USER_MESSAGE.format(ads=ads))
    prompt_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    response_ids = model.generate(**prompt_ids, max_new_tokens=100, stopping_criteria=stopping_criteria, do_sample=True, temperature=temp)
    response = tokenizer.decode(response_ids[0],skip_special_tokens=True)
    response = str(response.split("assistant")[-1]).strip()

    response_json = parse_json_wrapper(response)
    return response_json