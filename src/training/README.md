# TextFixer Training

This directory contains the training scripts for the TextFixer model.

## Overview

The training process fine-tunes Qwen2.5-0.5B model using SFT (Supervised Fine-Tuning) on the obfuscated text dataset to learn how to convert leetspeak back to normal text.

## Files

- `train_textfixer.py`: Main training script
- `config.py`: Configuration parameters for training

## Training Process

### 1. Dataset Preparation
The training uses the JSONL dataset created by `prepare_dataset.py` with the format:
```json
{"text": "<text_obfuscated>h3ll0 w0rld</text_obfuscated><text>hello world</text>"}
```

### 2. Model Architecture
- **Base Model**: Qwen2.5-0.5B
- **Training Method**: Supervised Fine-Tuning (SFT)
- **Task**: Text-to-text conversion (leetspeak → normal text)

### 3. Training Configuration

#### Hyperparameters
- **Epochs**: 3
- **Batch Size**: 4
- **Learning Rate**: 2e-5
- **Max Length**: 2048 tokens
- **Gradient Accumulation**: 4 steps
- **Mixed Precision**: FP16 (if CUDA available)

#### Hardware Requirements
- **GPU**: Recommended (CUDA compatible)
- **RAM**: Minimum 8GB, recommended 16GB+
- **VRAM**: Minimum 6GB for Qwen2.5-0.5B

## Usage

### Prerequisites
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare the dataset:
```bash
python src/dataset/main.py
python src/dataset/obfuscate_dataset.py
python src/dataset/prepare_dataset.py
```

### Start Training
```bash
python src/training/train_textfixer.py
```

## Output

The trained model will be saved to `models/textfixer/` with:
- Model weights
- Tokenizer
- Training configuration
- Checkpoints (last 3)

## Monitoring

Training progress is logged with:
- Loss metrics
- Learning rate
- Training steps
- Evaluation metrics

## Customization

You can modify training parameters in `config.py`:
- Change model size
- Adjust hyperparameters
- Modify training duration
- Configure hardware settings

## Expected Results

After training, the model should be able to:
- Recognize leetspeak patterns
- Convert obfuscated text to normal text
- Maintain context and meaning
- Handle various leetspeak variations 