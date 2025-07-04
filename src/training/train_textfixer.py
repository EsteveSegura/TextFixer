#!/usr/bin/env python3
"""
TextFixer Training Script

Trains the Qwen2.5-0.5B model using SFT (Supervised Fine-Tuning)
on the obfuscated text dataset.
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    dataset_path = "data/textfixer_dataset.jsonl"
    output_dir = "models/textfixer"

    logger.info(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()

    logger.info(f"Loading dataset from: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def formatting_func(example):
        return example["text"]

    # SFTConfig for training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        num_train_epochs=6,
        learning_rate=2e-5,
        logging_steps=10,
        save_steps=250,
        save_total_limit=1,
        report_to="none",
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        max_seq_length=256,
        # eos_token="<|im_end|>",  # Uncomment if your model requires it
    )

    # SFTTrainer does not take 'tokenizer' as argument in latest TRL
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        formatting_func=formatting_func,
    )

    logger.info("Training started...")
    trainer.train()
    logger.info(f"Training completed! Model saved in: {output_dir}")

if __name__ == "__main__":
    main() 