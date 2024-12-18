import os
import re
import time

from qwen_vl_utils import process_vision_info
from showui.processing_showui import ShowUIProcessor

min_pixels = 256*28*28
max_pixels = 1344*28*28

processor = ShowUIProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

img_dir = 'images'
img_list = os.listdir(img_dir)

# save the images with ui_graph visualization
vis_dir = 'display'

for img_url in img_list:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": os.path.join(img_dir, img_url),
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                },
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        ui_graph=True,
        ui_graph_threshold=1,
        ui_graph_vis_dir=vis_dir,
    )