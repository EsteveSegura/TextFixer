#!/usr/bin/env python3
"""
Script to convert the obfuscated dataset to JSONL format for TextFixer training.
"""

import json

def convert_to_jsonl():
    """Convert the obfuscated dataset to JSONL format."""
    
    # Load the obfuscated dataset
    with open('data/fineweb_subset_obfuscated.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Create JSONL file
    with open('data/textfixer_dataset.jsonl', 'w', encoding='utf-8') as f:
        for sample in dataset:
            text_obfuscated = sample.get('text_obfuscated', '')
            text_original = sample.get('text', '')
            
            # Create the formatted text with the specified format
            formatted_text = f"<text_obfuscated>{text_obfuscated}</text_obfuscated><text>{text_original}</text>"
            
            # Write as JSONL
            json_line = {"text": formatted_text}
            f.write(json.dumps(json_line, ensure_ascii=False) + '\n')
    
    print(f"Dataset converted to JSONL format: data/textfixer_dataset.jsonl")
    print(f"Total samples: {len(dataset)}")

if __name__ == "__main__":
    convert_to_jsonl() 