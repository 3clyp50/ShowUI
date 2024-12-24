import os
import json
import random
import numpy as np
import torch
from PIL import Image
from data.data_utils import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

class OmniActDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_image_dir,
        processor,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision="fp32",
        json_data="hf_train",
        inference=False,
        num_turn=1,
        num_history=4,
        interleaved_history='tttt',
        draw_history=0,
        random_sample=False,
        decay_factor=1,
        merge_patch=0,
        merge_threshold=0,
        merge_inference=False,
        merge_random=None,
    ):
        self.base_image_dir = base_image_dir
        self.processor = processor
        self.samples_per_epoch = samples_per_epoch
        self.precision = precision
        self.inference = inference
        self.num_turn = num_turn
        self.num_history = num_history
        self.interleaved_history = interleaved_history
        self.draw_history = draw_history
        self.random_sample = random_sample
        self.decay_factor = decay_factor
        self.merge_patch = merge_patch
        self.merge_threshold = merge_threshold
        self.merge_inference = merge_inference
        self.merge_random = merge_random

        # Load dataset
        self.data = []
        self.load_data(json_data)

    def load_data(self, json_data):
        """Load the OmniAct dataset from JSON files"""
        data_path = os.path.join(self.base_image_dir, "omniact", f"{json_data}.json")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")
            
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

    def __len__(self):
        if self.inference:
            return len(self.data)
        if self.random_sample:
            return self.samples_per_epoch
        return len(self.data)

    def __getitem__(self, idx):
        if self.random_sample and not self.inference:
            idx = random.randint(0, len(self.data) - 1)
            
        item = self.data[idx]
        
        # Process image
        image_path = os.path.join(self.base_image_dir, item['image_path'])
        image = Image.open(image_path).convert('RGB')
        
        # Process text
        conversation = item['conversation']
        if len(conversation) > self.num_turn * 2:
            start_idx = random.randint(0, len(conversation) - self.num_turn * 2)
            conversation = conversation[start_idx:start_idx + self.num_turn * 2]
            
        # Format conversation
        formatted_text = ""
        for i in range(0, len(conversation), 2):
            formatted_text += f"Human: {conversation[i]}\nAssistant: {conversation[i+1]}\n"
            
        # Add image tokens
        text_with_image = (f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n"
                          f"{formatted_text}")
        
        # Encode text and image
        encoding = self.processor(
            text=text_with_image,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.processor.tokenizer.model_max_length,
            return_attention_mask=True
        )
        
        # Process for model input
        input_ids = encoding["input_ids"][0]
        labels = input_ids.clone()
        
        # Mask labels before the assistant responses
        for i in range(len(input_ids)):
            if input_ids[i] == self.processor.tokenizer.encode("Assistant:", add_special_tokens=False)[0]:
                break
            labels[i] = IGNORE_INDEX
            
        # Convert image to appropriate precision
        if self.precision == "bf16":
            pixel_values = encoding["pixel_values"][0].to(torch.bfloat16)
        elif self.precision == "fp16":
            pixel_values = encoding["pixel_values"][0].to(torch.float16)
        else:
            pixel_values = encoding["pixel_values"][0]
            
        data_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_sizes": torch.tensor([image.size[1], image.size[0]]).unsqueeze(0)
        }
        
        return data_dict, {"id": idx, "image_path": item['image_path']}
