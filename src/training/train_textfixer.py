#!/usr/bin/env python3
"""
TextFixer Training Script

Trains Qwen2.5-0.5B model using SFT (Supervised Fine-Tuning)
on the obfuscated text dataset.
"""

import os
import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
import torch
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextFixerTrainer:
    """
    Trainer for TextFixer model using Qwen2.5-0.5B.
    """
    
    def __init__(self, 
                 model_name: str = "Qwen/Qwen2.5-0.5B",
                 dataset_path: str = "data/textfixer_dataset.jsonl",
                 output_dir: str = "models/textfixer"):
        """
        Initialize the trainer.
        
        Args:
            model_name: HuggingFace model name
            dataset_path: Path to the JSONL dataset
            output_dir: Directory to save the trained model
        """
        self.model_name = model_name
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
    def load_dataset(self) -> Dataset:
        """
        Load and prepare the dataset.
        
        Returns:
            HuggingFace Dataset object
        """
        logger.info(f"Loading dataset from: {self.dataset_path}")
        
        # Read JSONL file
        data = []
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        logger.info(f"Loaded {len(data)} samples")
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(data)
        
        return dataset
    
    def load_model_and_tokenizer(self):
        """
        Load the model and tokenizer.
        """
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True
        )
        
        logger.info("Model and tokenizer loaded successfully")
    
    def train(self, 
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 2e-5,
              warmup_steps: int = 100,
              save_steps: int = 500,
              logging_steps: int = 50):
        """
        Entrena el modelo usando SFTTrainer de TRL.
        """
        logger.info("Iniciando entrenamiento con SFTTrainer...")

        # Cargar dataset
        dataset = self.load_dataset()

        # Cargar modelo y tokenizer
        self.load_model_and_tokenizer()

        # Configuración de entrenamiento SFT
        sft_config = SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            save_steps=save_steps,
            logging_steps=logging_steps,
            save_total_limit=3,
            max_seq_length=2048,
            fp16=torch.cuda.is_available(),
            report_to=None,
            eval_steps=save_steps,
            evaluation_strategy="steps",
        )

        # Inicializar SFTTrainer
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=dataset,
            args=sft_config,
            tokenizer=self.tokenizer,
        )

        # Entrenar
        logger.info("Entrenamiento iniciado...")
        trainer.train()

        # Guardar modelo final
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        logger.info(f"Entrenamiento completado! Modelo guardado en: {self.output_dir}")


def main():
    """
    Main function to run the training.
    """
    logger.info("Starting TextFixer training...")
    
    # Initialize trainer
    trainer = TextFixerTrainer()
    
    # Start training
    trainer.train(
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-5,
        warmup_steps=100,
        save_steps=500,
        logging_steps=50
    )
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main() 