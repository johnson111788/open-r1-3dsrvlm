from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor
import base64
from io import BytesIO
from datasets import Dataset, DatasetDict

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

processing_class = AutoProcessor.from_pretrained("llava-hf/llama3-llava-next-8b-hf", padding_side="right") # TODO: during training, one always uses padding on the right

# Format into conversation
def make_conversation(example):
    
    options = [f"{opt}. {example[opt]}" for opt in ["A", "B", "C", "D"] if example[opt]]
    question_text = example["question"]
    options_text = "\n".join(options)
    question = f"<image>\nQuestion: {question_text}\nOptions:\n{options_text}\nPlease select the correct answer from the options above."

    image_path = '/home/ychou11/LMUData/images/3DSRBench/' + str(example['idx']) + '.jpg'
    image = Image.open(image_path)
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    return {
        "prompt": [
            {"role": "system", 'type': 'text', "text": SYSTEM_PROMPT},
            {"role": "user", 'type': 'text', "text": question},
        ],
        "img_str": img_str
    }

if __name__ == "__main__":
    dataset = load_dataset("ccvl/3DSRBench")
    dataset = dataset.map(lambda example, idx: {"idx": idx}, with_indices=True)
    dataset = dataset.map(make_conversation)
    
    for data in dataset['test']:
        import ipdb;ipdb.set_trace()
        prompts_text = processing_class.apply_chat_template(data['prompt'])
        break
