# ShowUI Training Guide

This guide provides comprehensive instructions for training the ShowUI model on both single GPU and multi-GPU setups.

## Model Overview

ShowUI is a vision-language model based on Qwen2-VL-2B-Instruct, specifically designed for UI understanding and interaction. The model is trained to:
- Understand UI elements and their relationships
- Ground natural language instructions to UI elements
- Generate accurate coordinates and bounding boxes for UI interactions

The training process uses LoRA (Low-Rank Adaptation) to efficiently fine-tune the model while maintaining its base capabilities.

## Environment Setup

### Linux Prerequisites (for Virtual Machines with GPUs)

1. Install system dependencies:
```bash
# Install required system packages
sudo apt install proot
sudo apt-get install python-dev-is-python3 python-pip gcc
sudo apt-get install linux-headers-$(uname -r)

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh

# Setup HuggingFace CLI
pip install -U huggingface-hub
export PATH="$HOME/.local/bin:$PATH"
source ~/.bashrc
```

2. Create and activate a conda environment:
```bash
conda create -n showui python=3.10
conda activate showui
```

2. Install PyTorch with CUDA support 
- (in our case CUDA 11.8 because we're using a H100):
```bash
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118 --user
```

3. Install required dependencies:
```bash
pip install -r requirements.txt --user
```

## Weights & Biases Setup

1. Create an account at [wandb.ai](https://wandb.ai) if you don't have one
2. Get your API key from [wandb.ai/settings](https://wandb.ai/settings)
3. The training scripts are already configured to use your wandb key for monitoring training progress

## Dataset Information

The training uses two main datasets:

### ShowUI-desktop-8K (Training Dataset)
- 8,000 desktop UI screenshots with annotations
- Diverse collection of desktop applications and interfaces
- Annotations include:
  - UI element locations
  - Element types (buttons, text fields, etc.)
  - Natural language descriptions
  - Interaction points

### ScreenSpot (Evaluation Dataset)
- High-quality UI screenshots for evaluation
- Focused on real-world interface scenarios
- Used to validate the model's:
  - Element localization accuracy
  - Understanding of UI contexts
  - Grounding capabilities

## Dataset Preparation

1. Move to the data directory (./ShowUI/data):
```bash
cd data
```

2. Download the required datasets:
```bash
# Download training dataset (ShowUI-desktop-8K)
huggingface-cli download showlab/ShowUI-desktop-8K --repo-type dataset --local-dir ./data

# Download evaluation dataset (ScreenSpot)
huggingface-cli download KevinQHLin/ScreenSpot --repo-type dataset --local-dir ./data
```

3. Organize the datasets in the following structure:
```
data/
  ScreenSpot/
    images/
    metadata/
  ShowUI-desktop/
    images/
    metadata/
```

## Metadata Processing

The datasets require preprocessing to create metadata in the correct format. We provide two scripts for this:

1. Process ScreenSpot dataset:
```bash
python data/hf_screenspot.py --data_dir ./data/ScreenSpot
```

2. Process ShowUI-desktop dataset:
```bash
python data/hf_mind2web.py --data_dir ./data/ShowUI-desktop
```

These scripts will update the metadata files in place, converting them to the required format. Each metadata file will follow this structure:
```json
{
    "img_url": "example.png",
    "img_size": [1920, 1080],
    "element": [
        {
            "instruction": "Button Text",
            "bbox": [0.1, 0.2, 0.3, 0.4],
            "data_type": "text",
            "point": [0.2, 0.3]
        }
    ],
    "element_size": 1
}
```

## Hardware Requirements and Training Time

### Single H100 GPU (80GB)
- Memory requirement: ~60GB VRAM
- Expected training time: ~8-12 hours per epoch
- Disk space: ~100GB for datasets and checkpoints

### Single A100 GPU (40GB)
- Memory requirement: ~35GB VRAM
- Expected training time: ~16-20 hours per epoch
- Disk space: ~100GB for datasets and checkpoints
- Common on cloud platforms (GCP, AWS, Azure)

### 8x H100 GPUs
- Memory requirement: ~45GB VRAM per GPU
- Expected training time: ~2-3 hours per epoch
- Disk space: ~200GB for distributed training artifacts

## Training Configuration

We provide training configurations for both full training and quick testing:

### Testing Configuration

For quick validation of the training setup:
```bash
# Windows
train_test.bat

# Linux
chmod +x train/train_test.sh  # First time only
./train/train_test.sh
```

This configuration uses:
- 100 training steps
- Smaller batch size (2) and gradient accumulation steps (2)
- Fewer worker processes (4)
- Single GPU with DeepSpeed ZeRO-2
- Useful for:
  - Validating the training setup
  - Testing data pipeline
  - Checking GPU memory usage
  - Verifying model configuration

### Full Training Configurations

We provide two configurations optimized for different hardware setups:

### Single H100 GPU (80GB)

Optimized for training on a single H100 GPU with the following key settings:
- Batch size: 4
- Gradient accumulation steps: 4
- DeepSpeed ZeRO-2 optimization
- Flash Attention 2
- 8 worker processes

To start training on a single H100:

Windows:
```bash
train_single_h100.bat
```

Linux:
```bash
# Make scripts executable (first time only)
chmod +x train/train_single_h100.sh train/train_8x_h100.sh

# Run training
./train/train_single_h100.sh
```

### Single A100 GPU (40GB)

Optimized for training on a single A100 GPU with memory-efficient settings:
- Batch size: 2 (reduced for 40GB VRAM)
- Gradient accumulation steps: 8 (increased to maintain effective batch size)
- DeepSpeed ZeRO-3 optimization (more aggressive memory savings)
- Flash Attention 2
- 4 worker processes (reduced for memory efficiency)

To start training on a single A100:

Windows:
```bash
train_single_a100.bat
```

Linux:
```bash
# Make scripts executable (first time only)
chmod +x train/train_single_a100.sh

# Run training
./train/train_single_a100.sh
```

Note: The A100 configuration maintains training quality while adapting to lower VRAM by:
- Using smaller per-GPU batch size
- Compensating with increased gradient accumulation
- Leveraging ZeRO-3 for additional memory optimization
- Reducing worker processes to free up system memory

### 8x H100 GPUs

Optimized for distributed training across 8 H100 GPUs with these settings:
- Batch size: 8
- Gradient accumulation steps: 4
- DeepSpeed ZeRO-3 optimization
- Flash Attention 2
- 16 worker processes
- Distributed across all GPUs

To start distributed training:

Windows:
```bash
train_8x_h100.bat
```

Linux:
```bash
./train/train_8x_h100.sh
```

## Training Parameters

Key training parameters in both configurations:

- Model: Qwen2-VL-2B-Instruct
- Training epochs: 1 (adjust as needed)
- Steps per epoch: 100
- Learning rate: 0.0001
- Precision: bfloat16
- LoRA parameters:
  - r: 8
  - alpha: 64
- Visual tokens:
  - Min: 256
  - Max: 1344
- History settings:
  - Number of history items: 4
  - Turn count: 1
  - Interleaved history: "tttt"
- Crop settings:
  - Min: 0.5
  - Max: 1.5

## Monitoring Training

1. Training progress can be monitored in real-time through:
   - Weights & Biases dashboard at wandb.ai
   - Local tensorboard logs in the ./logs directory

2. The following metrics are tracked:
   - Training loss
   - Validation metrics on ScreenSpot dataset
   - GPU utilization and memory usage
   - Training speed (samples/second)

## Checkpoints and Model Conversion

Training checkpoints are automatically saved in the ./logs directory:
- Single GPU training: ./logs/showui_single_h100/
- Multi-GPU training: ./logs/showui_8x_h100/

### Converting DeepSpeed Checkpoints

To convert DeepSpeed checkpoints to a format usable with Hugging Face:

1. For LoRA weights:
```bash
python -c "from peft import PeftModel; \
model = PeftModel.from_pretrained('Qwen/Qwen2-VL-2B-Instruct', './logs/showui_8x_h100/checkpoint-latest/'); \
model.save_pretrained('./showui_lora')"
```

2. For full model weights:
```bash
python -c "from transformers import AutoModelForCausalLM; \
model = AutoModelForCausalLM.from_pretrained('./logs/showui_8x_h100/checkpoint-latest/'); \
model.save_pretrained('./showui_full')"
```

## Evaluation

The training includes periodic evaluation on the ScreenSpot dataset. If you want to evaluate on your own dataset:

1. Create a new evaluation function in main/eval_X.py
2. Follow the format of existing evaluation functions like main/eval_screenspot.py
3. Update the --val_dataset parameter in the training script to use your new evaluation

## Troubleshooting

Common issues and solutions:

1. Out of memory errors:
   - Reduce batch_size
   - Increase gradient_accumulation_steps
   - Enable gradient_checkpointing (already enabled by default)

2. Slow training:
   - Ensure Flash Attention 2 is properly enabled
   - Check GPU utilization
   - Adjust number of worker processes

3. Distributed training issues:
   - Verify all GPUs are visible (nvidia-smi)
   - Check network connectivity between GPUs
   - Ensure consistent CUDA versions

4. Dataset loading issues:
   - Verify dataset directory structure
   - Check metadata file format
   - Ensure all images are accessible

## Next Steps

After successful training:

1. Convert the model checkpoint to HuggingFace format using the provided conversion scripts
2. Evaluate the model on your specific use case
3. Consider fine-tuning hyperparameters based on your results
4. For deployment:
   - Use the converted HuggingFace model
   - Optimize inference parameters
   - Consider quantization for faster inference

For any customization needs, refer to the original ShowUI paper and codebase for additional guidance.
