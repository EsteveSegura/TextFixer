#!/usr/bin/env python3
"""
TextFixer Dataset Downloader

This script downloads a subset of the FineWeb dataset from HuggingFace
for training the TextFixer LLM model.

Dataset: HuggingFaceFW/fineweb
Source: https://huggingface.co/datasets/HuggingFaceFW/fineweb
"""

import os
import json
from datasets import load_dataset
from typing import List, Dict, Any
import logging
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FineWebDatasetDownloader:
    """
    Downloads and processes the FineWeb dataset for TextFixer training.
    """
    
    def __init__(self, num_samples: int = 300, output_dir: str = "data"):
        """
        Initialize the dataset downloader.
        
        Args:
            num_samples: Number of samples to download (default: 3000)
            output_dir: Directory to save the processed dataset
        """
        self.num_samples = num_samples
        self.output_dir = output_dir
        self.dataset_name = "HuggingFaceFW/fineweb"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
    def download_dataset(self) -> None:
        """
        Download the FineWeb dataset and save a subset of samples.
        """
        try:
            logger.info(f"Loading dataset: {self.dataset_name}")
            
            # Load the dataset from HuggingFace
            # Using the 'default' subset which contains the entire dataset
            dataset = load_dataset(
                self.dataset_name,
                split="train",
                streaming=True  # Use streaming to handle large dataset efficiently
            )
            
            logger.info("Dataset loaded successfully. Processing samples...")
            
            # Collect samples
            samples = []
            for i, sample in enumerate(dataset):
                if i >= self.num_samples:
                    break
                    
                # Cut the text to between 15 and 35 words randomly
                original_text = sample.get("text", "")
                text = sample.get("text", "")
                words = text.split()
                if len(words) > 15:
                    max_words = min(35, len(words))
                    num_words = random.randint(15, max_words)
                    text = " ".join(words[:num_words])
                
                # Extract the text content and metadata
                processed_sample = {
                    "id": i,
                    "original_text": original_text,
                    "text": text,
                    "language": sample.get("language", "en"),
                    "language_score": sample.get("language_score", 0.0),
                    "token_count": sample.get("token_count", 0)
                }
                
                samples.append(processed_sample)
                
                # Log progress every 100 samples
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{self.num_samples} samples")
            
            logger.info(f"Successfully processed {len(samples)} samples")
            
            # Save the dataset
            self._save_dataset(samples)
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {str(e)}")
            raise
    
    def _save_dataset(self, samples: List[Dict[str, Any]]) -> None:
        """
        Save the processed dataset to files.
        
        Args:
            samples: List of processed samples
        """
        # Save as JSON
        json_path = os.path.join(self.output_dir, "fineweb_subset.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Dataset saved to: {json_path}")
        
        # Save as text file (one sample per line)
        txt_path = os.path.join(self.output_dir, "fineweb_subset.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(sample["text"] + "\n\n")
        
        logger.info(f"Text file saved to: {txt_path}")
        
        # Print dataset statistics
        self._print_statistics(samples)
    
    def _print_statistics(self, samples: List[Dict[str, Any]]) -> None:
        """
        Print statistics about the downloaded dataset.
        
        Args:
            samples: List of processed samples
        """
        total_tokens = sum(sample["token_count"] for sample in samples)
        avg_tokens = total_tokens / len(samples) if samples else 0
        total_chars = sum(len(sample["text"]) for sample in samples)
        
        logger.info("Dataset Statistics:")
        logger.info(f"  Total samples: {len(samples)}")
        logger.info(f"  Total tokens: {total_tokens:,}")
        logger.info(f"  Average tokens per sample: {avg_tokens:.1f}")
        logger.info(f"  Total characters: {total_chars:,}")
        logger.info(f"  Average characters per sample: {total_chars / len(samples):.1f}" if samples else "0")


def main():
    """
    Main function to download the FineWeb dataset.
    """
    logger.info("Starting TextFixer dataset download...")
    
    # Initialize the downloader
    downloader = FineWebDatasetDownloader(num_samples=300)
    
    # Download the dataset
    downloader.download_dataset()
    
    logger.info("Dataset download completed successfully!")


if __name__ == "__main__":
    main() 