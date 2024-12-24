# ShowUI Training

This repository contains the training setup for ShowUI, a vision-language model based on Qwen2-VL-2B-Instruct for UI understanding and interaction.

## Quick Start

1. Set up environment:
```bash
conda create -n showui python=3.10
conda activate showui
pip install -r requirements.txt --user
```

2. Get your [Weights & Biases](https://wandb.ai) API key

3. Choose your training configuration:
- [Single H100 GPU Training](train_single_h100.bat)
- [8x H100 GPUs Training](train_8x_h100.bat)

## Documentation

- [Detailed Training Guide](TRAIN_GUIDE.md) - Comprehensive instructions for training setup and execution
- [Original Training Instructions](TRAIN.md) - Original training documentation from the ShowUI team

## Hardware Requirements

### Single GPU Setup
- 1x NVIDIA H100 (80GB)
- ~100GB disk space
- 32GB+ system RAM

### Multi-GPU Setup
- 8x NVIDIA H100 (80GB)
- ~200GB disk space
- 64GB+ system RAM

## Support

For detailed instructions, troubleshooting, and best practices, please refer to the [Training Guide](TRAIN_GUIDE.md).

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.
