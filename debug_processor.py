import torch
from transformers import LlavaNextProcessor,AutoProcessor, LlavaNextForConditionalGeneration
from PIL import Image

processing_class = AutoProcessor.from_pretrained('llava-hf/llama3-llava-next-8b-hf', padding_side="right") # during 
# model = LlavaNextForConditionalGeneration.from_pretrained(
#                     "llava-hf/llama3-llava-next-8b-hf", torch_dtype=torch.float16, 
#                     # ignore_mismatched_sizes=True, state_dict=None
#                     )

conversation = [
    {
        'role': 'system',
        
        'content': [{'type': 'text','text':'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>'}], 
    }, 
    {
        'role': 'user',
        'content': [{'type': 'text','text':'<image>\nQuestion: Consider the real-world 3D orientations of the objects. What is the relationship between the orientations of the bicycle and the motorcycle, parallel of perpendicular to each other?\nOptions:\nA. parallel\nB. perpendicular\nPlease select the correct answer from the options above.'}
            ], 
        }
    ]
image_path = "/home/ychou11/LMUData/images/3DSRBench/9122.jpg"
images_input = Image.open(image_path).convert("RGB")
prompts_text = processing_class.apply_chat_template(conversation, add_generation_prompt=True)
# prompt_inputs = processing_class(text=prompts_text, images=images_input, 
#                                               return_tensors="pt", 
#                                               padding=True)
        
print(prompts_text)
# <|start_header_id|>system<|end_header_id|>


# A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer><|eot_id|><|start_header_id|>user<|end_header_id|>


# <image>
# Question: Consider the real-world 3D orientations of the objects. What is the relationship between the orientations of the bicycle and the motorcycle, parallel of perpendicular to each other?
# Options:
# A. parallel
# B. perpendicular
# Please select the correct answer from the options above.<|eot_id|><|start_header_id|>assistant<|end_header_id|>


# From pipeline
# <|start_header_id|>system<|end_header_id|>


# A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer><|eot_id|><|start_header_id|>user<|end_header_id|>


# <image>
# Question: Consider the real-world 3D orientations of the objects. What is the relationship between the orientations of the bicycle and the motorcycle, parallel of perpendicular to each other?
# Options:
# A. parallel
# B. perpendicular
# Please select the correct answer from the options above.<|eot_id|><|start_header_id|>assistant<|end_header_id|>


