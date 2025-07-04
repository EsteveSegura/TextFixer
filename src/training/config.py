#!/usr/bin/env python3
"""
Configuration file for TextFixer training.
"""

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DATASET_PATH = "data/textfixer_dataset.jsonl"
OUTPUT_DIR = "models/textfixer"

# Training hyperparameters
TRAINING_CONFIG = {
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-5,
    "warmup_steps": 100,
    "save_steps": 500,
    "logging_steps": 50,
    "max_length": 2048,
    "gradient_accumulation_steps": 4,
    "save_total_limit": 3,
    "fp16": True,
    "evaluation_strategy": "steps",
    "eval_steps": 500,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False,
}

# Hardware configuration
DEVICE_CONFIG = {
    "use_cuda": True,
    "mixed_precision": True,
    "device_map": "auto",
}

# Dataset configuration
DATASET_CONFIG = {
    "max_length": 2048,
    "padding": True,
    "truncation": True,
    "return_tensors": "pt",
} 