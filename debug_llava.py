from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "What’s shown in this image?"},
            ],
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "This image shows a red stop sign."},]
    },
    {

        "role": "user",
        "content": [
            {"type": "text", "text": "Describe the image in more details."},
        ],
    },
]

text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
