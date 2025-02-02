from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

processor = LlavaNextProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf")

url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
image = Image.open(requests.get(url, stream=True).raw)

conversation = [
    {
        'role': 'system',
        
        'content': [{'type': 'text','text':'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>'}], 
    },
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
print(prompt)
# <|start_header_id|>system<|end_header_id|>


# A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer><|eot_id|><|start_header_id|>user<|end_header_id|>

# <image>
# What is shown in this image?<|eot_id|><|start_header_id|>assistant<|end_header_id|>





inputs = processor(image, prompt, return_tensors="pt").to("cuda:0")

# autoregressively complete prompt
# output = model.generate(**inputs, max_new_tokens=100)

# print(processor.decode(output[0], skip_special_tokens=True))